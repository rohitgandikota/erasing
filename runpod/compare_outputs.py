#!/usr/bin/env python3
"""
Comparison grid: shows ALL samples for every prompt.
Layout:
  rows    = prompts
  columns = [orig_0 orig_1 orig_2 orig_3 orig_4 | erased_0 erased_1 erased_2 erased_3 erased_4]
A divider line separates the two halves.
"""
import argparse
import os
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

DIVIDER_W = 6  # px width of the separator between orig and erased


def load_image(folder: Path, case: int, sample: int, thumb: int) -> Image.Image:
    path = folder / f"{case}_{sample}.png"
    if not path.exists():
        img = Image.new("RGB", (thumb, thumb), (200, 200, 200))
        ImageDraw.Draw(img).text((4, thumb // 2 - 8), f"missing\n{case}_{sample}", fill=(80, 80, 80))
        return img
    return Image.open(path).convert("RGB").resize((thumb, thumb))


def best_font(size):
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def make_grid(original_dir, erased_dir, prompts_csv, out, thumb=200, label_h=28, header_h=56, padding=4):
    df = pd.read_csv(prompts_csv)

    # detect how many samples exist
    n_samples = 0
    first_case = int(df.iloc[0]["case_number"])
    while (original_dir / f"{first_case}_{n_samples}.png").exists():
        n_samples += 1
    if n_samples == 0:
        raise FileNotFoundError(f"No images found in {original_dir} for case {first_case}")
    print(f"  Detected {n_samples} samples per prompt, {len(df)} prompts")

    n_cols = n_samples * 2  # orig cols + erased cols
    cell_w = thumb + padding
    cell_h = thumb + label_h + padding

    total_w = n_cols * cell_w + DIVIDER_W + padding * 2
    total_h = header_h + len(df) * cell_h + padding

    grid = Image.new("RGB", (total_w, total_h), (245, 245, 245))
    draw = ImageDraw.Draw(grid)
    font_h = best_font(16)
    font_p = best_font(12)

    # Column headers
    orig_center_x  = padding + (n_samples * cell_w) // 2
    erased_center_x = padding + n_samples * cell_w + DIVIDER_W + (n_samples * cell_w) // 2
    draw.text((orig_center_x,  header_h // 2), "Original SD v1.4",       fill=(30, 30, 30),  font=font_h, anchor="mm")
    draw.text((erased_center_x, header_h // 2), "ESD — Van Gogh erased", fill=(180, 40, 40), font=font_h, anchor="mm")

    # Vertical divider
    div_x = padding + n_samples * cell_w
    draw.rectangle([div_x, 0, div_x + DIVIDER_W, total_h], fill=(100, 100, 100))

    for row_idx, row in df.iterrows():
        case   = int(row["case_number"])
        prompt = str(row["prompt"])
        y_top  = header_h + row_idx * cell_h

        for s in range(n_samples):
            # Original
            x = padding + s * cell_w
            img = load_image(original_dir, case, s, thumb)
            grid.paste(img, (x, y_top))

            # Erased
            x2 = padding + n_samples * cell_w + DIVIDER_W + s * cell_w
            img2 = load_image(erased_dir, case, s, thumb)
            grid.paste(img2, (x2, y_top))

        # Prompt label
        draw.text((padding + 2, y_top + thumb + 3), f"[{case}] {prompt}", fill=(50, 50, 50), font=font_p)

    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out)
    print(f"Saved -> {out}  ({total_w}x{total_h}px)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--original_dir", required=True)
    p.add_argument("--erased_dir",   required=True)
    p.add_argument("--prompts_csv",  required=True)
    p.add_argument("--out",          default="outputs/comparison_grid.png")
    p.add_argument("--thumb",        type=int, default=200)
    args = p.parse_args()

    make_grid(
        original_dir=Path(args.original_dir),
        erased_dir=Path(args.erased_dir),
        prompts_csv=Path(args.prompts_csv),
        out=Path(args.out),
        thumb=args.thumb,
    )
