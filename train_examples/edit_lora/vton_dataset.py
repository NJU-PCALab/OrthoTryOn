import json
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from train_examples.edit_lora.dataset_utils import collate, encode_prompt, load_jsonl, resize_crop
from utils.task_specific import PROMPT_TEMPLATE_SUFFIX, get_default_prompt, get_prompt_template


class VTONDataset(Dataset):
    def __init__(self, cfg, tokenizer, text_processor, resolution):
        self.root = Path(cfg.vton_root_path)
        self.records = load_jsonl(self.root / "train_dataset.jsonl")
        prompt_dir = Path(getattr(cfg, "vton_prompt_dir", "train/edit_instructs"))
        self.prompt_dir = prompt_dir if prompt_dir.is_absolute() else self.root / prompt_dir
        if not self.prompt_dir.is_dir():
            raise NotADirectoryError(f"VTON prompt directory not found: {self.prompt_dir}")

        self.resolution = tuple(resolution)
        self.tokenizer = tokenizer
        self.image_processor = text_processor.image_processor
        self.max_length = cfg.text_tokenizer_max_length
        self.null_text_ratio = cfg.null_text_ratio
        self.template = get_prompt_template("vton")

    def __len__(self):
        return len(self.records)

    def _prompt(self, image_path):
        prompt_path = self.prompt_dir / f"{Path(image_path).stem}.json"
        with prompt_path.open("r", encoding="utf-8") as file:
            prompt = json.load(file).get("edit_instruction", "")
        return prompt or get_default_prompt("vton")

    def __getitem__(self, index):
        for _ in range(100):
            try:
                record = self.records[index]
                target_path = self.root / record["image"]
                person_path = self.root / record["condition"]
                cloth_path = self.root / record["cloth"]
                pose_path = Path(str(target_path).replace("/image/", "/mmpose_skeleton/"))
                pose_path = pose_path.with_name(f"{pose_path.stem}_skeleton{pose_path.suffix}")

                prompt = self._prompt(target_path)
                if random.random() < self.null_text_ratio:
                    prompt = ""

                target = Image.open(target_path).convert("RGB")
                person = Image.open(person_path).convert("RGB")
                cloth = Image.open(cloth_path).convert("RGB")
                pose = Image.open(pose_path).convert("RGB")

                person_vl = resize_crop(person, self.resolution, vl=True)
                cloth_vl = resize_crop(cloth, self.resolution, vl=True)
                pose_vl = resize_crop(pose, self.resolution, vl=True)
                input_ids, attention_mask, pixel_values, image_grid_thw = encode_prompt(
                    prompt,
                    [person_vl, cloth_vl, pose_vl],
                    self.tokenizer,
                    self.image_processor,
                    self.max_length,
                    self.template,
                    PROMPT_TEMPLATE_SUFFIX,
                )
                return {
                    "gt_image": resize_crop(target, self.resolution),
                    "person_image": resize_crop(person, self.resolution),
                    "cloth_image": resize_crop(cloth, self.resolution),
                    "pose_image": resize_crop(pose, self.resolution),
                    "prompt": prompt,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "pixel_values": pixel_values,
                    "image_grid_thw": image_grid_thw,
                }
            except Exception as error:
                last_error = error
                index = random.randrange(len(self.records))
        raise RuntimeError(f"Failed to load VTON data after 100 retries: {last_error}")


def build_dataloader(cfg, tokenizer, text_processor, resolution):
    dataset = VTONDataset(cfg, tokenizer, text_processor, resolution)
    return DataLoader(
        dataset,
        batch_size=cfg.train_batch_size,
        num_workers=cfg.dataloader_num_workers,
        collate_fn=collate,
        shuffle=False,
    )