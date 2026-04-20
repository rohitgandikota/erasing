#!/usr/bin/env python3
"""
3-way comparison grid: Original SD | ESD | SPACE
Layout: rows = prompts, columns = [orig_0..orig_N | esd_0..esd_N | space_0..space_N]
Divider lines separate the three model groups.
"""
import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import os

DIVIDER_W = 6


def load_image(folder: Path, case: int, sample: int, thumb: int) -> Image.Image:
    path = folder / f"{case}_{sample}.png"
    if not path.exists():
        img = Image.new("RGB", (thumb, thumb), (200, 200, 200))
        ImageDraw.Draw(img).text((4, thumb // 2 - 8), f"missing\n{case}_{sample}", fill=(80, 80, 80))
        return img
    return Image.open(path).convert("RGB").resize((thumb, thumb))


def detect_n_samples(folder: Path, first_case: int) -> int:
    n = 0
    while (folder / f"{first_case}_{n}.png").exists():
        n += 1
    return n


def best_font(size):
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def make_grid(dirs, labels, prompts_csv, out, thumb=180, label_h=26, header_h=60, padding=4):
    df = pd.read_csv(prompts_csv)
    first_case = int(df.iloc[0]["case_number"])
    n_samples = detect_n_samples(dirs[0], first_case)
    if n_samples == 0:
        raise FileNotFoundError(f"No images found in {dirs[0]} for case {first_case}")
    print(f"  {len(dirs)} models, {n_samples} samples/prompt, {len(df)} prompts")

    n_groups = len(dirs)
    cell_w = thumb + padding
    cell_h = thumb + label_h + padding
    group_w = n_samples * cell_w

    total_w = n_groups * group_w + (n_groups - 1) * DIVIDER_W + padding * 2
    total_h = header_h + len(df) * cell_h + padding

    grid = Image.new("RGB", (total_w, total_h), (245, 245, 245))
    draw = ImageDraw.Draw(grid)
    font_h = best_font(15)
    font_p = best_font(11)

    header_colors = [(30, 30, 30), (180, 40, 40), (40, 120, 40)]

    for g, (d, label) in enumerate(zip(dirs, labels)):
        x_start = padding + g * (group_w + DIVIDER_W)
        center_x = x_start + group_w // 2
        color = header_colors[g % len(header_colors)]
        draw.text((center_x, header_h // 2), label, fill=color, font=font_h, anchor="mm")
        if g > 0:
            div_x = x_start - DIVIDER_W
            draw.rectangle([div_x, 0, div_x + DIVIDER_W, total_h], fill=(120, 120, 120))

    for row_idx, row in df.iterrows():
        case = int(row["case_number"])
        prompt = str(row["prompt"])
        y_top = header_h + row_idx * cell_h

        for g, d in enumerate(dirs):
            x_start = padding + g * (group_w + DIVIDER_W)
            for s in range(n_samples):
                x = x_start + s * cell_w
                img = load_image(d, case, s, thumb)
                grid.paste(img, (x, y_top))

        draw.text((padding + 2, y_top + thumb + 3), f"[{case}] {prompt}", fill=(50, 50, 50), font=font_p)

    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out)
    print(f"Saved → {out}  ({total_w}x{total_h}px)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dirs",        nargs="+", required=True)
    p.add_argument("--labels",      nargs="+", required=True)
    p.add_argument("--prompts_csv", required=True)
    p.add_argument("--out",         default="outputs/comparison_grid_3way.png")
    p.add_argument("--thumb",       type=int, default=180)
    args = p.parse_args()
    assert len(args.dirs) == len(args.labels), "--dirs and --labels must have the same length"
    make_grid(
        [Path(d) for d in args.dirs],
        args.labels,
        args.prompts_csv,
        Path(args.out),
        thumb=args.thumb,
    )
