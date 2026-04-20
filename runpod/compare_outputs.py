#!/usr/bin/env python3
"""
Build a side-by-side comparison grid:
  rows  = prompts
  cols  = [Original SD v1.4 | ESD Van Gogh-erased]
Picks the first sample image (sample_idx=0) for each case.
"""
import argparse
import os
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def load_image(folder: Path, case: int, sample: int = 0) -> Image.Image:
    path = folder / f"{case}_{sample}.png"
    if not path.exists():
        img = Image.new("RGB", (512, 512), color=(180, 180, 180))
        draw = ImageDraw.Draw(img)
        draw.text((10, 240), f"Missing: {path.name}", fill=(80, 80, 80))
        return img
    return Image.open(path).convert("RGB")


def make_grid(
    original_dir: Path,
    erased_dir: Path,
    prompts_csv: Path,
    out: Path,
    thumb_size: int = 512,
    label_h: int = 40,
    header_h: int = 60,
    padding: int = 8,
) -> None:
    df = pd.read_csv(prompts_csv)
    n_rows = len(df)
    col_labels = ["Original SD v1.4", "ESD — Van Gogh erased"]

    cell_w = thumb_size + padding
    cell_h = thumb_size + label_h + padding
    total_w = 2 * cell_w + padding
    total_h = header_h + n_rows * cell_h + padding

    grid = Image.new("RGB", (total_w, total_h), color=(245, 245, 245))
    draw = ImageDraw.Draw(grid)

    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    font_header = font_label = ImageFont.load_default()
    for fp in font_paths:
        if os.path.exists(fp):
            font_header = ImageFont.truetype(fp, 20)
            font_label = ImageFont.truetype(fp, 13)
            break

    for col_idx, label in enumerate(col_labels):
        x = padding + col_idx * cell_w + thumb_size // 2
        draw.text((x, header_h // 2), label, fill=(40, 40, 40), font=font_header, anchor="mm")

    for row_idx, row in df.iterrows():
        y_top = header_h + row_idx * cell_h
        case = int(row["case_number"])
        prompt = str(row["prompt"])

        for col_idx, folder in enumerate([original_dir, erased_dir]):
            x_left = padding + col_idx * cell_w
            img = load_image(folder, case).resize((thumb_size, thumb_size))
            grid.paste(img, (x_left, y_top))

        draw.text((padding + 4, y_top + thumb_size + 4), f"[{case}] {prompt}", fill=(60, 60, 60), font=font_label)

    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out)
    print(f"Saved comparison grid -> {out}  ({total_w}x{total_h}px)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_dir", required=True)
    parser.add_argument("--erased_dir", required=True)
    parser.add_argument("--prompts_csv", required=True)
    parser.add_argument("--out", default="outputs/comparison_grid.png")
    parser.add_argument("--thumb_size", type=int, default=512)
    args = parser.parse_args()

    make_grid(
        original_dir=Path(args.original_dir),
        erased_dir=Path(args.erased_dir),
        prompts_csv=Path(args.prompts_csv),
        out=Path(args.out),
        thumb_size=args.thumb_size,
    )
