import json
import math
from pathlib import Path

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from longcat_image.utils import encode_prompt_vton


def load_jsonl(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
    if not records:
        raise RuntimeError(f"No records found in {path}.")
    return records


def resize_crop(image, target_size, vl=False):
    target_height, target_width = target_size
    width, height = image.size
    if height / width >= target_height / target_width:
        resized_width = target_width
        resized_height = math.ceil(height * target_width / width)
    else:
        resized_width = math.ceil(width * target_height / height)
        resized_height = target_height

    transforms = [
        T.Resize((resized_height, resized_width), interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop((target_height, target_width)),
    ]
    if vl:
        transforms.append(
            T.Resize((target_height // 2, target_width // 2), interpolation=InterpolationMode.BICUBIC)
        )
    else:
        transforms.extend([T.ToTensor(), T.Normalize([0.5], [0.5])])
    return T.Compose(transforms)(image)


def encode_prompt(prompt, images, tokenizer, image_processor, max_length, template, suffix):
    return encode_prompt_vton(
        prompt,
        images,
        tokenizer,
        image_processor,
        max_length,
        template,
        suffix,
    )


def collate(batch):
    output = {
        "gt_images": torch.stack([item["gt_image"] for item in batch]),
        "person_images": torch.stack([item["person_image"] for item in batch]),
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "image_grid_thw": torch.stack([item["image_grid_thw"] for item in batch]),
        "prompts": [item["prompt"] for item in batch],
    }
    if "cloth_image" in batch[0]:
        output["cloth_images"] = torch.stack([item["cloth_image"] for item in batch])
    if "pose_image" in batch[0]:
        output["pose_images"] = torch.stack([item["pose_image"] for item in batch])
    return output
