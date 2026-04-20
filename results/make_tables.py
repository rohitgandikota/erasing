#!/usr/bin/env python3
"""
Render summary_table.csv as a markdown table and LaTeX table.

Usage:
    python results/make_tables.py --input results/summary_table.csv
"""

import argparse
from pathlib import Path

import pandas as pd

DISPLAY_NAMES = {
    "sdv14":                            "Original SD v1.4",
    "diffusers-VanGogh-ESDx1-UNET":    "ESD-x (Gandikota et al., 2023)",
    "space-Van_Gogh":                   "SPACE (ours)",
}


def rename_model(name: str) -> str:
    for k, v in DISPLAY_NAMES.items():
        if k.lower() in name.lower():
            return v
    return name


def render_markdown(df: pd.DataFrame) -> str:
    df = df.copy()
    df["Model"] = df["Model"].apply(rename_model)
    lines = []
    header = "| " + " | ".join(df.columns) + " |"
    sep    = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    lines.append(header)
    lines.append(sep)
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def render_latex(df: pd.DataFrame) -> str:
    df = df.copy()
    df["Model"] = df["Model"].apply(rename_model)
    cols = list(df.columns)
    n = len(cols)
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\begin{{tabular}}{{l{'c' * (n-1)}}}",
        "\\toprule",
        " & ".join(cols) + " \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(str(v) for v in row) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Van Gogh concept erasure comparison. "
        "CLIP$\\downarrow$ and ResNet-acc$\\downarrow$ measure erasure strength; "
        "LPIPS$\\uparrow$ measures perceptual change; "
        "FID$\\downarrow$ measures overall image quality drift.}",
        "\\label{tab:vangogh_comparison}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/summary_table.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    out_dir = Path(args.input).parent

    md = render_markdown(df)
    md_path = out_dir / "ablation_table.md"
    md_path.write_text(md)
    print(f"Markdown → {md_path}\n")
    print(md)

    latex = render_latex(df)
    tex_path = out_dir / "ablation_table.tex"
    tex_path.write_text(latex)
    print(f"\nLaTeX → {tex_path}\n")
    print(latex)


if __name__ == "__main__":
    main()
