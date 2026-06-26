import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Refine VTON results with ground-truth backgrounds."
    )
    parser.add_argument("--gen_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mask_suffix", type=str, default="_mask.png")
    parser.add_argument("--skip", action="store_true")
    return parser.parse_args()


def validate_directory(path_value, argument_name):
    path = Path(path_value).expanduser().resolve()

    if not path.is_dir():
        raise NotADirectoryError(
            f"Directory specified by --{argument_name} does not exist: {path}"
        )

    return path


def main():
    args = parse_args()

    gen_dir = validate_directory(args.gen_dir, "gen_dir")
    gt_dir = validate_directory(args.gt_dir, "gt_dir")
    mask_dir = validate_directory(args.mask_dir, "mask_dir")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gen_paths = sorted(
        path
        for path in gen_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not gen_paths:
        raise RuntimeError(f"No generated images found in: {gen_dir}")

    processed_count = 0
    skipped_count = 0

    for gen_path in tqdm(gen_paths, desc="Refining"):
        image_name = gen_path.name
        gt_path = gt_dir / image_name
        mask_path = mask_dir / f"{gen_path.stem}{args.mask_suffix}"
        output_path = output_dir / image_name

        if args.skip and output_path.is_file():
            skipped_count += 1
            continue

        if not gt_path.is_file():
            print(f"[Skip] Missing GT image: {gt_path}")
            skipped_count += 1
            continue

        if not mask_path.is_file():
            print(f"[Skip] Missing mask image: {mask_path}")
            skipped_count += 1
            continue

        gen_image = Image.open(gen_path).convert("RGB")
        gt_image = Image.open(gt_path).convert("RGB")
        mask_image = Image.open(mask_path).convert("L")

        if gt_image.size != gen_image.size:
            gt_image = gt_image.resize(
                gen_image.size,
                resample=Image.BICUBIC,
            )

        if mask_image.size != gen_image.size:
            mask_image = mask_image.resize(
                gen_image.size,
                resample=Image.BICUBIC,
            )

        refined_image = Image.composite(
            gen_image,
            gt_image,
            mask_image,
        )
        refined_image.save(output_path)

        processed_count += 1

    print(f"Processed images: {processed_count}")
    print(f"Skipped images: {skipped_count}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()