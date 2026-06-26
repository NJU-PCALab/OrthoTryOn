import random
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from train_examples.edit_lora.dataset_utils import collate, encode_prompt, resize_crop
from utils.task_specific import PROMPT_TEMPLATE_SUFFIX, get_default_prompt, get_prompt_template


class PoseDataset(Dataset):
    def __init__(self, cfg, tokenizer, text_processor, resolution):
        self.root = Path(cfg.pose_root_path)
        pairs_path = self.root / "fasion-resize-pairs-train.csv"
        pairs = pd.read_csv(pairs_path)
        self.records = [
            (self.root / "train_highres" / row["from"], self.root / "train_highres" / row["to"])
            for _, row in pairs.iterrows()
        ]
        if not self.records:
            raise RuntimeError(f"No pose pairs found in {pairs_path}.")

        self.resolution = tuple(resolution)
        self.tokenizer = tokenizer
        self.image_processor = text_processor.image_processor
        self.max_length = cfg.text_tokenizer_max_length
        self.null_text_ratio = cfg.null_text_ratio
        self.template = get_prompt_template("pose")
        self.default_prompt = get_default_prompt("pose")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        for _ in range(100):
            try:
                source_path, target_path = self.records[index]
                pose_path = Path(
                    str(target_path)
                    .replace("train_highres", "train_skeleton")
                    .replace(target_path.suffix, f"_skeleton{target_path.suffix}")
                )
                source = Image.open(source_path).convert("RGB")
                target = Image.open(target_path).convert("RGB")
                pose = Image.open(pose_path).convert("RGB")
                prompt = "" if random.random() < self.null_text_ratio else self.default_prompt

                source_vl = resize_crop(source, self.resolution, vl=True)
                pose_vl = resize_crop(pose, self.resolution, vl=True)
                input_ids, attention_mask, pixel_values, image_grid_thw = encode_prompt(
                    prompt,
                    [source_vl, pose_vl],
                    self.tokenizer,
                    self.image_processor,
                    self.max_length,
                    self.template,
                    PROMPT_TEMPLATE_SUFFIX,
                )
                return {
                    "gt_image": resize_crop(target, self.resolution),
                    "person_image": resize_crop(source, self.resolution),
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
        raise RuntimeError(f"Failed to load pose data after 100 retries: {last_error}")


def build_dataloader(cfg, tokenizer, text_processor, resolution):
    dataset = PoseDataset(cfg, tokenizer, text_processor, resolution)
    return DataLoader(
        dataset,
        batch_size=cfg.train_batch_size,
        num_workers=cfg.dataloader_num_workers,
        collate_fn=collate,
        shuffle=False,
    )
