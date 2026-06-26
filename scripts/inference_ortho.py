import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from diffusers import LongCatImageTransformer2DModel
from train_examples.edit_lora import lora_zoo
from utils.pipeline_longcat_ortho import LongCatImageOrthoPipeline
from utils.task_specific import (
    get_default_prompt,
    normalize_task,
    required_conditions,
    resolve_negative_task,
)


TARGET_MODULES = (
    "attn.to_k",
    "attn.to_q",
    "attn.to_v",
    "attn.to_out.0",
    "attn.add_k_proj",
    "attn.add_q_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "ff.net.0.proj",
    "ff.net.2",
    "ff_context.net.0.proj",
    "ff_context.net.2",
)


def parse_args():
    parser = argparse.ArgumentParser("Universal OrthoTryOn inference")

    parser.add_argument("--task", choices=("vton", "vtoff", "pose"), required=True)
    parser.add_argument(
        "--input_mode",
        choices=("auto", "json", "pose_csv"),
        default="auto",
    )

    parser.add_argument("--vton_root_dir", default=None)
    parser.add_argument("--vtoff_root_dir", default=None)
    parser.add_argument("--pose_root_dir", default=None)

    parser.add_argument("--test_pairs_file", default="test_pairs.txt")
    parser.add_argument("--json_folder", default=None)
    parser.add_argument("--csv_file", default="fasion-resize-pairs-test.csv")

    parser.add_argument("--output_folder", default="./output_results")
    parser.add_argument("--setting", default="unpaired")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--ckpt_step", default=None)
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--num_tasks", type=int, default=3)
    parser.add_argument("--sampler", default="cfg")
    parser.add_argument(
        "--negative_task",
        choices=("vton", "vtoff", "pose"),
        default=None,
    )
    parser.add_argument("--negative_prompt", default=None)
    parser.add_argument(
        "--cfg",
        type=float,
        default=None,
        help="Guidance scale. Default: 2.0 for vton; 1.5 for vtoff and pose.",
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--skip", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)

    return parser.parse_args()


def default_cfg(task):
    return 2.0 if task == "vton" else 1.5


def resolve_mode(args):
    expected_mode = "pose_csv" if args.task == "pose" else "json"

    if args.input_mode == "auto":
        return expected_mode

    if args.input_mode != expected_mode:
        raise ValueError(
            f"Task '{args.task}' requires input_mode='{expected_mode}', "
            f"but received '{args.input_mode}'."
        )

    return args.input_mode


def validate_root_dir(value, argument_name):
    if not value:
        raise ValueError(f"Set --{argument_name}.")

    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(
            f"Directory specified by --{argument_name} does not exist: {root}"
        )

    return root


def resolve_pair_root(args):
    if args.task == "vton":
        return validate_root_dir(args.vton_root_dir, "vton_root_dir")

    if args.task == "vtoff":
        return validate_root_dir(args.vtoff_root_dir, "vtoff_root_dir")

    raise ValueError("Pair records are only supported for vton and vtoff tasks.")


def resolve_pose_root(args):
    return validate_root_dir(args.pose_root_dir, "pose_root_dir")


def resolve_root_path(value, root):
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def resolve_pairs_path(args, root):
    pairs_path = resolve_root_path(args.test_pairs_file, root)

    if not pairs_path.is_file():
        raise FileNotFoundError(f"Test pairs file not found: {pairs_path}")

    return pairs_path


def resolve_vton_prompt_dir(args, root):
    if args.json_folder is None:
        prompt_dir = root / Path("test/edit_instructs")
    else:
        source = resolve_root_path(args.json_folder, root)
        prompt_dir = (
            source
            if any(source.glob("*.json"))
            else source / Path("test/edit_instructs").name
        )

    if not prompt_dir.is_dir():
        raise FileNotFoundError(
            f"VTON prompt directory not found: {prompt_dir}"
        )

    return prompt_dir


def get_first_value(mapping, keys):
    for key in keys:
        value = mapping.get(key)
        if value:
            return value
    return None


def normalize_name(value):
    if not value:
        return None
    return Path(str(value).replace("\\", "/")).name


def load_prompt_index(json_dir):
    pair_prompts = {}
    model_prompts = defaultdict(list)

    for json_path in sorted(json_dir.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        model_name = normalize_name(
            get_first_value(data, ("model_image", "source_image", "image"))
        )
        garment_name = normalize_name(
            get_first_value(data, ("garment_image", "target_image", "cloth"))
        )
        prompt = data.get("edit_instruction") or ""

        if not model_name or not prompt:
            continue

        model_prompts[model_name].append(prompt)
        if garment_name:
            pair_prompts[(model_name, garment_name)] = prompt

    unique_model_prompts = {}
    for model_name, prompts in model_prompts.items():
        unique_prompts = list(dict.fromkeys(prompts))
        if len(unique_prompts) == 1:
            unique_model_prompts[model_name] = unique_prompts[0]

    return pair_prompts, unique_model_prompts


def read_test_pairs(pairs_path):
    with pairs_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                raise ValueError(
                    f"Expected at least two fields at line {line_number} "
                    f"in {pairs_path}: {line}"
                )

            model_name = normalize_name(parts[0])
            garment_name = normalize_name(parts[1])

            if not model_name or not garment_name:
                raise ValueError(
                    f"Invalid pair at line {line_number} in {pairs_path}: {line}"
                )

            yield model_name, garment_name


def resolve_pair_pose_path(root, model_name):
    image_name = Path(model_name)
    return root / Path("test/mmpose_skeleton") / (
        f"{image_name.stem}_skeleton{image_name.suffix}"
    )


def list_vton_fallback_clothes(args):
    if args.vton_root_dir is None:
        return []

    vton_root = validate_root_dir(args.vton_root_dir, "vton_root_dir")
    cloth_dir = vton_root / Path("test/cloth")

    if not cloth_dir.is_dir():
        raise NotADirectoryError(
            f"VTON fallback cloth directory not found: {cloth_dir}"
        )

    valid_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    clothes = sorted(
        path
        for path in cloth_dir.iterdir()
        if path.is_file() and path.suffix.lower() in valid_extensions
    )

    if not clothes:
        raise RuntimeError(
            f"No valid cloth images found in VTON fallback directory: {cloth_dir}"
        )

    return clothes


def choose_cloth(record_cloth, fallback_clothes):
    if record_cloth is not None and record_cloth.is_file():
        return record_cloth

    if fallback_clothes:
        return random.choice(fallback_clothes)

    return record_cloth


def pair_records(args):
    root = resolve_pair_root(args)
    pairs_path = resolve_pairs_path(args, root)

    image_dir = root / Path("test/image")
    cloth_dir = root / Path("test/cloth")

    if not image_dir.is_dir():
        raise NotADirectoryError(f"Image directory not found: {image_dir}")

    if not cloth_dir.is_dir():
        raise NotADirectoryError(f"Cloth directory not found: {cloth_dir}")

    pair_prompts = {}
    model_prompts = {}
    if args.task == "vton":
        prompt_dir = resolve_vton_prompt_dir(args, root)
        pair_prompts, model_prompts = load_prompt_index(prompt_dir)

    for index, (model_name, garment_name) in enumerate(read_test_pairs(pairs_path)):
        reference = image_dir / model_name
        cloth = cloth_dir / garment_name
        pose = resolve_pair_pose_path(root, model_name)

        if args.task == "vton":
            prompt = pair_prompts.get((model_name, garment_name))
            if prompt is None:
                prompt = model_prompts.get(model_name)
            if prompt is None:
                prompt = get_default_prompt("vton")
        else:
            prompt = get_default_prompt("vtoff")

        yield {
            "index": index,
            "name": model_name,
            "reference": reference,
            "cloth": cloth,
            "pose": pose,
            "prompt": prompt,
        }


def pose_records(args):
    root = resolve_pose_root(args)
    csv_path = resolve_root_path(args.csv_file, root)

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    image_dir = root / Path("test_highres")
    pose_dir = root / Path("test_skeleton")

    if not image_dir.is_dir():
        raise NotADirectoryError(f"Image directory not found: {image_dir}")

    if not pose_dir.is_dir():
        raise NotADirectoryError(f"Pose directory not found: {pose_dir}")

    for index, row in pd.read_csv(csv_path).iterrows():
        source = image_dir / row["from"]
        target = image_dir / row["to"]
        pose = pose_dir / f"{target.stem}_skeleton{target.suffix}"

        yield {
            "index": int(index),
            "name": f"{source.stem}_2_{target.stem}_vis.png",
            "reference": source,
            "cloth": None,
            "pose": pose,
            "prompt": get_default_prompt("pose"),
        }


def open_image(path, name):
    if path is None or not path.is_file():
        raise FileNotFoundError(f"Missing {name}: {path}")

    return Image.open(path).convert("RGB")


def load_transformer(args, device):
    transformer = LongCatImageTransformer2DModel.from_pretrained(
        args.model_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    ).to(device)

    if args.lora_path is None:
        return transformer

    checkpoint = Path(args.lora_path)
    if args.ckpt_step is not None:
        checkpoint = checkpoint / f"checkpoints-{args.ckpt_step}"

    if not checkpoint.is_dir():
        raise FileNotFoundError(f"LoRA checkpoint not found: {checkpoint}")

    transformer = lora_zoo.inject_rotated_shared_lora(
        transformer,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        num_tasks=args.num_tasks,
        target_modules=TARGET_MODULES,
    )
    lora_zoo.load_rotated_lora_weights(transformer, str(checkpoint))

    return transformer


def main():
    args = parse_args()
    args.task = normalize_task(args.task)
    args.cfg = default_cfg(args.task) if args.cfg is None else args.cfg

    mode = resolve_mode(args)
    negative_task = resolve_negative_task(
        args.task,
        args.sampler,
        args.negative_task,
    )
    needed = required_conditions(args.task, negative_task)
    fallback_clothes = []
    if "cloth_image" in needed and args.vton_root_dir is not None:
        fallback_clothes = list_vton_fallback_clothes(args)

    label = args.task if negative_task is None else f"{args.task}_neg-{negative_task}"

    output_dir = Path(args.output_folder) / label
    if args.task == 'vton':
        output_dir = output_dir / str(args.setting).strip()
    if args.ckpt_step is not None:
        output_dir = output_dir / str(args.ckpt_step)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    transformer = load_transformer(args, device)
    pipe = LongCatImageOrthoPipeline.from_pretrained(
        args.model_path,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    ).to(device, torch.bfloat16)

    records = pair_records(args) if mode == "json" else pose_records(args)
    if args.max_samples is not None:
        records = list(records)[: args.max_samples]

    for record in tqdm(records, desc=f"{args.task} inference"):
        save_path = output_dir / record["name"]
        if args.skip and save_path.is_file():
            continue

        paths = {
            "ref_image": record["reference"],
            "cloth_image": choose_cloth(
                record["cloth"],
                fallback_clothes,
            ),
            "pose_image": record["pose"],
        }
        missing = [
            name
            for name in needed
            if paths[name] is None or not paths[name].is_file()
        ]
        if missing:
            print(f"[Skip] {record['name']}: missing {', '.join(missing)}")
            continue

        images = {name: open_image(paths[name], name) for name in needed}
        negative_prompt = args.negative_prompt
        if negative_prompt is None:
            negative_prompt = (
                get_default_prompt(negative_task)
                if negative_task is not None
                else ""
            )

        result = pipe(
            image=images.get("ref_image"),
            cloth_image=images.get("cloth_image"),
            pose_image=images.get("pose_image"),
            prompt=record["prompt"],
            negative_prompt=negative_prompt,
            task=args.task,
            sampler=args.sampler,
            negative_task=negative_task,
            guidance_scale=args.cfg,
            num_inference_steps=args.steps,
            generator=torch.Generator("cpu").manual_seed(args.seed),
        ).images[0]
        result.save(save_path)

    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
