import random
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from train_examples.edit_lora.dataset_utils import collate, encode_prompt, load_jsonl, resize_crop
from utils.task_specific import PROMPT_TEMPLATE_SUFFIX, get_default_prompt, get_prompt_template


class VTOFFDataset(Dataset):
    def __init__(self, cfg, tokenizer, text_processor, resolution):
        self.root = Path(cfg.vtoff_root_path)
        self.records = load_jsonl(self.root / "train_dataset.jsonl")
        self.resolution = tuple(resolution)
        self.tokenizer = tokenizer
        self.image_processor = text_processor.image_processor
        self.max_length = cfg.text_tokenizer_max_length
        self.null_text_ratio = cfg.null_text_ratio
        self.template = get_prompt_template("vtoff")
        self.default_prompt = get_default_prompt("vtoff")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        for _ in range(100):
            try:
                record = self.records[index]
                reference_path = self.root / record["image"]
                target_path = self.root / record["cloth"]
                style_path = self.root / self.records[random.randrange(len(self.records))]["cloth"]
                prompt = "" if random.random() < self.null_text_ratio else self.default_prompt

                reference = Image.open(reference_path).convert("RGB")
                target = Image.open(target_path).convert("RGB")
                style = Image.open(style_path).convert("RGB")

                reference_vl = resize_crop(reference, self.resolution, vl=True)
                style_vl = resize_crop(style, self.resolution, vl=True)
                input_ids, attention_mask, pixel_values, image_grid_thw = encode_prompt(
                    prompt,
                    [reference_vl, style_vl],
                    self.tokenizer,
                    self.image_processor,
                    self.max_length,
                    self.template,
                    PROMPT_TEMPLATE_SUFFIX,
                )
                return {
                    "gt_image": resize_crop(target, self.resolution),
                    "person_image": resize_crop(reference, self.resolution),
                    "cloth_image": resize_crop(style, self.resolution),
                    "prompt": prompt,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "pixel_values": pixel_values,
                    "image_grid_thw": image_grid_thw,
                }
            except Exception as error:
                last_error = error
                index = random.randrange(len(self.records))
        raise RuntimeError(f"Failed to load VTOFF data after 100 retries: {last_error}")


def build_dataloader(cfg, tokenizer, text_processor, resolution):
    dataset = VTOFFDataset(cfg, tokenizer, text_processor, resolution)
    return DataLoader(
        dataset,
        batch_size=cfg.train_batch_size,
        num_workers=cfg.dataloader_num_workers,
        collate_fn=collate,
        shuffle=False,
    )
