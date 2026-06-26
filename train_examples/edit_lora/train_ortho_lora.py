import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path

import diffusers
import torch
import transformers
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models import AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from transformers import AutoModel, AutoProcessor, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from train_examples.edit_lora import lora_zoo
from train_examples.edit_lora.off_dataset import build_dataloader as build_vtoff_dataloader
from train_examples.edit_lora.pose_dataset import build_dataloader as build_pose_dataloader
from train_examples.edit_lora.vton_dataset import build_dataloader as build_vton_dataloader
from longcat_image.models import LongCatImageTransformer2DModel
from longcat_image.utils import LogBuffer, calculate_shift, pack_latents, prepare_pos_ids, unpack_latents
from utils.task_specific import CONDITION_MODALITY_IDS, TASK_IDS, TASKS, get_task_conditions


logger = get_logger(__name__)
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
BATCH_KEYS = {
    "ref_image": "person_images",
    "cloth_image": "cloth_images",
    "pose_image": "pose_images",
}


def cycle(loader):
    while True:
        yield from loader


def set_task(model, task):
    task_id = TASK_IDS[task]
    model = model.module if hasattr(model, "module") else model
    for module in model.modules():
        if hasattr(module, "set_task"):
            module.set_task(task_id)


@torch.no_grad()
def encode_latents(vae, images, device, dtype):
    latents = vae.encode(images.to(device=device, dtype=dtype)).latent_dist.sample()
    return (latents - vae.config.shift_factor) * vae.config.scaling_factor


@torch.no_grad()
def encode_text(batch, text_encoder, args, device, dtype):
    image_grid_thw = batch["image_grid_thw"].to(device)
    if image_grid_thw.ndim == 3:
        image_grid_thw = image_grid_thw.flatten(0, 1)

    output = text_encoder(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        pixel_values=batch["pixel_values"].to(device),
        image_grid_thw=image_grid_thw,
        output_hidden_states=True,
    )
    end = -args.prompt_template_encode_end_idx or None
    return output.hidden_states[-1][:, args.prompt_template_encode_start_idx:end].to(dtype)


def pack(latents):
    return pack_latents(
        latents,
        batch_size=latents.shape[0],
        num_channels_latents=latents.shape[1],
        height=latents.shape[2],
        width=latents.shape[3],
    )


def get_latents(task, batch, vae, device, dtype):
    latents = {
        "target": encode_latents(vae, batch["gt_images"], device, dtype),
        "ref_image": encode_latents(vae, batch["person_images"], device, dtype),
    }
    for condition in get_task_conditions(task):
        if condition == "ref_image":
            continue
        latents[condition] = encode_latents(vae, batch[BATCH_KEYS[condition]], device, dtype)
    return latents


def build_model_input(task, noisy_latents, condition_latents, prompt_length, device):
    height = noisy_latents.shape[-2] // 2
    width = noisy_latents.shape[-1] // 2

    def image_ids(modality_id):
        return prepare_pos_ids(
            modality_id=modality_id,
            type="image",
            start=(prompt_length, prompt_length),
            height=height,
            width=width,
        ).to(device=device, dtype=torch.float64)

    packed_noisy = pack(noisy_latents)
    packed_inputs = [packed_noisy]
    position_ids = [image_ids(1)]
    for condition in get_task_conditions(task):
        packed_inputs.append(pack(condition_latents[condition]))
        position_ids.append(image_ids(CONDITION_MODALITY_IDS[condition]))
    return torch.cat(packed_inputs, dim=1), torch.cat(position_ids, dim=0), packed_noisy


class FisherTracker:
    def __init__(self, beta, update_interval):
        if not 0 <= beta < 1:
            raise ValueError("fisher_beta must be in [0, 1).")
        if update_interval < 1:
            raise ValueError("fisher_update_interval must be positive.")
        self.beta = beta
        self.update_interval = update_interval
        self.values = {task: None for task in TASKS}

    @torch.no_grad()
    def update(self, model, task, step):
        if step % self.update_interval:
            return

        if self.values[task] is None:
            self.values[task] = {
                name: torch.zeros_like(parameter, device="cpu", dtype=torch.float32)
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }

        fisher = self.values[task]
        for value in fisher.values():
            value.mul_(self.beta)

        for name, parameter in model.named_parameters():
            if parameter.requires_grad and parameter.grad is not None:
                gradient = parameter.grad.detach().to(device="cpu", dtype=torch.float32)
                fisher[name].add_(gradient.square(), alpha=1 - self.beta)

    @torch.no_grad()
    def similarity(self, left, right):
        left_fisher = self.values[left]
        right_fisher = self.values[right]
        if left_fisher is None or right_fisher is None:
            return None

        dot = left_norm = right_norm = 0.0
        for name, left_value in left_fisher.items():
            right_value = right_fisher[name]
            dot += torch.sum(left_value * right_value, dtype=torch.float64).item()
            left_norm += torch.sum(left_value.square(), dtype=torch.float64).item()
            right_norm += torch.sum(right_value.square(), dtype=torch.float64).item()

        denominator = (left_norm * right_norm) ** 0.5
        return dot / denominator if denominator else None

    def line(self):
        mapping = []
        for task in TASKS:
            candidates = [other for other in TASKS if other != task and self.similarity(task, other) is not None]
            negative = max(candidates, key=lambda other: self.similarity(task, other)) if candidates else "unavailable"
            mapping.append(f"{task} -> {negative}")
        return "Fisher negative-task mapping: " + "; ".join(mapping) + "."

    def save(self, directory):
        line = self.line()
        Path(directory, "fisher_negative_tasks.txt").write_text(line + "\n", encoding="utf-8")
        return line


def train(
    args,
    accelerator,
    transformer,
    vae,
    text_encoder,
    scheduler,
    optimizer,
    lr_scheduler,
    loaders,
    model_ema,
    fisher_tracker,
    mu,
    dtype,
    global_step,
):
    iterators = {task: cycle(loader) for task, loader in loaders.items()}
    log_buffer = LogBuffer()
    last_log_time = time.time()
    data_start = time.time()
    data_time = 0.0
    optimizer.zero_grad(set_to_none=True)

    while global_step < args.max_train_steps:
        task = random.choice(TASKS)
        set_task(transformer, task)
        batch = next(iterators[task])
        data_time += time.time() - data_start

        latents = get_latents(task, batch, vae, accelerator.device, dtype)
        prompt_embeds = encode_text(batch, text_encoder, args, accelerator.device, dtype)

        with accelerator.accumulate(transformer):
            sigma = torch.sigmoid(
                torch.randn(
                    latents["target"].shape[0],
                    device=accelerator.device,
                    dtype=latents["target"].dtype,
                )
            )
            if args.use_dynamic_shifting:
                sigma = scheduler.time_shift(mu, 1.0, sigma)

            noise = torch.randn_like(latents["target"])
            noisy_latents = (1 - sigma[:, None, None, None]) * latents["target"] + sigma[:, None, None, None] * noise
            model_input, image_ids, packed_noisy = build_model_input(
                task,
                noisy_latents,
                latents,
                prompt_embeds.shape[1],
                accelerator.device,
            )
            text_ids = prepare_pos_ids(
                modality_id=0,
                type="text",
                start=(0, 0),
                num_token=prompt_embeds.shape[1],
            ).to(device=accelerator.device, dtype=torch.float64)

            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
                prediction = transformer(
                    model_input,
                    prompt_embeds,
                    sigma,
                    image_ids,
                    text_ids,
                    None,
                    return_dict=False,
                )[0]

            prediction = prediction[:, : packed_noisy.size(1)]
            prediction = unpack_latents(
                prediction,
                height=latents["target"].shape[2] * 8,
                width=latents["target"].shape[3] * 8,
                vae_scale_factor=16,
            )
            target = noise - latents["target"]
            loss = (prediction.float() - target.float()).square().flatten(1).mean(1).mean()
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                gradients = [parameter.grad.detach().norm(2) for parameter in transformer.parameters() if parameter.grad is not None]
                grad_norm = torch.norm(torch.stack(gradients), 2) if gradients else torch.zeros((), device=accelerator.device)
                if fisher_tracker is not None:
                    fisher_tracker.update(accelerator.unwrap_model(transformer), task, global_step + 1)

            optimizer.step()
            if accelerator.sync_gradients:
                if not accelerator.optimizer_step_was_skipped:
                    lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if model_ema is not None:
                    model_ema.step(transformer.parameters())

        if not accelerator.sync_gradients:
            continue

        global_step += 1
        height, width = batch["person_images"].shape[-2:]
        loss_value = accelerator.gather(loss.detach()).mean().item()
        logs = {
            "loss": loss_value,
            f"loss_{task}": loss_value,
            "task_id": TASK_IDS[task],
            "aspect_ratio": height / width,
            "lr": lr_scheduler.get_last_lr()[0],
            "grad_norm": accelerator.gather(grad_norm).mean().item(),
        }
        log_buffer.update(logs)
        accelerator.log(logs, step=global_step)

        if global_step == 1 or global_step % args.log_interval == 0:
            log_buffer.average()
            elapsed = (time.time() - last_log_time) / args.log_interval
            data_elapsed = data_time / args.log_interval
            shape = latents["target"].shape[1:]
            values = ", ".join(f"{key}:{value:.4f}" for key, value in log_buffer.output.items())
            logger.info(
                f"Step={global_step}, Task={task}, time_all:{elapsed:.3f}, time_data:{data_elapsed:.3f}, "
                f"lr:{logs['lr']:.3e}, s:(ch:{shape[0]},h:{shape[1]},w:{shape[2]}), {values}"
            )
            last_log_time = time.time()
            data_time = 0.0
            log_buffer.clear()

        if global_step % args.save_model_steps == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                checkpoint = Path(args.work_dir) / f"checkpoints-{global_step}"
                lora_zoo.save_rotated_lora_weights(accelerator.unwrap_model(transformer), checkpoint)
                if fisher_tracker is not None:
                    logger.info(fisher_tracker.save(checkpoint))
                logger.info(f"Saved LoRA checkpoint to {checkpoint}.")
            accelerator.wait_for_everyone()

        data_start = time.time()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--report_to", default="tensorboard")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--pretrained_lora_path", default=None)
    parser.add_argument("--num_tasks", type=int, default=len(TASKS))
    parser.add_argument("--fisher_beta", type=float, default=0.99)
    parser.add_argument("--fisher_update_interval", type=int, default=100)
    return parser.parse_args()


def load_config(cli_args):
    config_path = Path(cli_args.config) if cli_args.config else Path(__file__).with_name("train_config.yaml")
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    values = vars(cli_args)
    values.update(config)
    return argparse.Namespace(**values)


def tracker_config(args):
    output = {}
    for key, value in vars(args).items():
        output[key] = value if isinstance(value, (bool, int, float, str)) else str(value)
    return output


def resolve_checkpoint(work_dir, value):
    if not value:
        return None
    if value != "latest":
        path = Path(value)
        if not path.is_dir():
            path = Path(work_dir) / path.name
        return path if path.is_dir() else None

    checkpoints = sorted(
        Path(work_dir).glob("checkpoints-*"),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]),
    )
    return checkpoints[-1] if checkpoints else None


def main():
    args = load_config(parse_args())
    if args.num_tasks != len(TASKS):
        raise ValueError(f"num_tasks must be {len(TASKS)}.")

    os.umask(0o000)
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.work_dir) / "logs"
    log_dir.mkdir(exist_ok=True)
    with (log_dir / "train.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(vars(args), file, sort_keys=False)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=args.work_dir, logging_dir=str(log_dir)),
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(accelerator.mixed_precision, torch.float32)
    transformer_path = args.diffusion_pretrain_weight or Path(args.pretrained_model_name_or_path) / "transformer"
    transformer = LongCatImageTransformer2DModel.from_pretrained(transformer_path, ignore_mismatched_sizes=False)
    transformer = lora_zoo.inject_rotated_shared_lora(
        transformer,
        rank=args.lora_rank,
        alpha=args.lora_rank,
        num_tasks=args.num_tasks,
        target_modules=TARGET_MODULES,
    )
    transformer.requires_grad_(False)
    for name, parameter in transformer.named_parameters():
        if "lora_" in name:
            parameter.requires_grad_(True)

    if args.pretrained_lora_path:
        lora_zoo.load_rotated_lora_weights(transformer, args.pretrained_lora_path)

    model_ema = EMAModel(transformer.parameters(), decay=args.ema_rate) if args.use_ema else None
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=dtype).to(accelerator.device).eval()
    text_encoder = AutoModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(accelerator.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", trust_remote_code=True)
    text_processor = AutoProcessor.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", trust_remote_code=True)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    latent_height, latent_width = int(args.resolution[0]) // 8, int(args.resolution[1]) // 8
    mu = calculate_shift(
        (latent_height // 2) * (latent_width // 2),
        scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len,
        scheduler.config.base_shift,
        scheduler.config.max_shift,
    )

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError as error:
            raise ImportError("Install bitsandbytes to use --use_8bit_adam.") from error
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        (parameter for parameter in transformer.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    loaders = {
        "vton": build_vton_dataloader(args, tokenizer, text_processor, args.resolution),
        "vtoff": build_vtoff_dataloader(args, tokenizer, text_processor, args.resolution),
        "pose": build_pose_dataloader(args, tokenizer, text_processor, args.resolution),
    }

    transformer.to(accelerator.device, dtype=dtype)
    if model_ema is not None:
        model_ema.to(accelerator.device, dtype=dtype)

    transformer, optimizer, loaders["vton"], loaders["vtoff"], loaders["pose"], lr_scheduler = accelerator.prepare(
        transformer,
        optimizer,
        loaders["vton"],
        loaders["vtoff"],
        loaders["pose"],
        lr_scheduler,
    )

    global_step = 0
    checkpoint = resolve_checkpoint(args.work_dir, args.resume_from_checkpoint)
    if checkpoint is not None:
        lora_zoo.load_rotated_lora_weights(accelerator.unwrap_model(transformer), checkpoint)
        global_step = int(checkpoint.name.rsplit("-", 1)[-1])
        logger.info(f"Resumed LoRA weights from {checkpoint}.")
    elif args.resume_from_checkpoint:
        logger.warning(f"Checkpoint not found: {args.resume_from_checkpoint}")

    if accelerator.is_main_process:
        accelerator.init_trackers("sft", tracker_config(args))

    fisher_tracker = (
        FisherTracker(args.fisher_beta, args.fisher_update_interval)
        if accelerator.is_main_process
        else None
    )
    train(
        args,
        accelerator,
        transformer,
        vae,
        text_encoder,
        scheduler,
        optimizer,
        lr_scheduler,
        loaders,
        model_ema,
        fisher_tracker,
        mu,
        dtype,
        global_step,
    )


if __name__ == "__main__":
    main()
