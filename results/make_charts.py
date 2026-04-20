#!/usr/bin/env python3
"""
Grouped bar charts comparing Original SD, ESD-x, and SPACE across 4 metrics.

Usage:
    python results/make_charts.py --input results/summary_table.csv --out results/comparison_charts.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DISPLAY_NAMES = {
    "sdv14":                            "Original SD v1.4",
    "diffusers-VanGogh-ESDx1-UNET":    "ESD-x (baseline)",
    "space-Van_Gogh":                   "SPACE (ours)",
}
COLORS = ["#4C72B0", "#DD8452", "#55A868"]

METRICS = [
    ("CLIP ↓",       "CLIP Score ↓\n(lower = more erased)",        True),
    ("ResNet-acc ↓", "ResNet Top-1 Acc ↓\n(lower = more erased)", True),
    ("LPIPS ↑",      "LPIPS ↑\n(higher = more perceptual change)", False),
    ("FID ↓",        "FID ↓\n(lower = less quality drift)",        True),
]


def rename_model(name: str) -> str:
    for k, v in DISPLAY_NAMES.items():
        if k.lower() in name.lower():
            return v
    return name


def to_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/summary_table.csv")
    parser.add_argument("--out",   default="results/comparison_charts.png")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["Model"] = df["Model"].apply(rename_model)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Van Gogh Concept Erasure: ESD vs SPACE", fontsize=14, fontweight="bold", y=1.02)

    x = np.arange(len(df))
    bar_w = 0.6

    for ax, (col, ylabel, lower_better) in zip(axes, METRICS):
        values = [to_float(v) for v in df[col]]
        has_data = [v is not None for v in values]

        if not any(has_data):
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=12, color="gray")
            ax.set_title(ylabel, fontsize=10)
            continue

        bars = ax.bar(
            x[has_data],
            [v for v, h in zip(values, has_data) if h],
            bar_w,
            color=[COLORS[i % len(COLORS)] for i, h in enumerate(has_data) if h],
            edgecolor="white",
            linewidth=0.8,
        )

        # Value labels on bars
        for bar, val in zip(bars, [v for v, h in zip(values, has_data) if h]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0] + 0.001),
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9,
            )

        ax.set_xticks(x[has_data])
        ax.set_xticklabels(
            [df["Model"].iloc[i] for i, h in enumerate(has_data) if h],
            rotation=15, ha="right", fontsize=9,
        )
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        arrow = "↓ better" if lower_better else "↑ better"
        ax.set_xlabel(arrow, fontsize=8, color="gray")

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Chart saved → {out}")


if __name__ == "__main__":
    main()
