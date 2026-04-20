#!/usr/bin/env python3
"""
Collect all per-metric CSVs from results/ and produce a single summary_table.csv.

Expected input files (all optional — missing ones are noted as N/A):
  results/clip_scores_summary.csv       columns: model, mean_clip_score
  results/classify_sdv14.csv            columns: from imageclassify.py
  results/classify_diffusers-*.csv
  results/classify_space-*.csv
  results/lpips_esd.csv                 columns: mean_lpips (or similar)
  results/lpips_space.csv
  results/fid_scores.csv                columns: model, fid

Output: results/summary_table.csv with columns:
  model | CLIP ↓ | ResNet-acc ↓ | LPIPS ↑ | FID ↓

Usage:
    python runpod/collect_results.py --results_dir results --output_csv results/summary_table.csv
"""

import argparse
import os
from pathlib import Path

import pandas as pd

NA = "N/A"


def safe_read(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def extract_clip(results_dir: Path) -> dict:
    df = safe_read(results_dir / "clip_scores_summary.csv")
    if df.empty:
        return {}
    return dict(zip(df["model"], df["mean_clip_score"].round(4)))


def extract_classify(results_dir: Path) -> dict:
    """Return {model_name: mean_top1_score} by reading imageclassify CSVs.

    imageclassify.py produces columns: category_top1 (str), index_top1, scores_top1 (float).
    We report mean scores_top1 as a proxy for classification confidence.
    """
    out = {}
    for f in results_dir.glob("classify_*.csv"):
        model_name = f.stem.replace("classify_", "")
        df = pd.read_csv(f)
        if "scores_top1" in df.columns:
            out[model_name] = round(float(df["scores_top1"].mean()), 4)
        else:
            # Fallback: numeric columns only
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
                        and ("score" in c.lower() or "acc" in c.lower())]
            if num_cols:
                out[model_name] = round(float(df[num_cols[0]].mean()), 4)
    return out


def extract_lpips(results_dir: Path) -> dict:
    out = {}
    for f in results_dir.glob("lpips_*.csv"):
        label = f.stem.replace("lpips_", "")
        df = pd.read_csv(f)
        lpips_cols = [c for c in df.columns if "lpips" in c.lower() or "loss" in c.lower() or "score" in c.lower()]
        if lpips_cols:
            out[label] = round(df[lpips_cols[0]].mean(), 4)
    return out


def extract_fid(results_dir: Path) -> dict:
    df = safe_read(results_dir / "fid_scores.csv")
    if df.empty:
        return {}
    return dict(zip(df["model"], df["fid"].round(2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--output_csv",  default="results/summary_table.csv")
    args = parser.parse_args()

    rd = Path(args.results_dir)
    clip_map     = extract_clip(rd)
    classify_map = extract_classify(rd)
    lpips_map    = extract_lpips(rd)
    fid_map      = extract_fid(rd)

    # Determine all model names
    all_models = sorted(set(
        list(clip_map) + list(classify_map) + list(fid_map)
    ))
    # Canonical ordering: sdv14 first, then esd, then space
    def sort_key(m):
        if "sdv14" in m.lower() or "sd_v14" in m.lower():
            return 0
        if "esd" in m.lower():
            return 1
        if "space" in m.lower():
            return 2
        return 3
    all_models.sort(key=sort_key)

    rows = []
    for model in all_models:
        # LPIPS: keyed by "esd" or "space" shorthand, not full model name
        lpips_val = NA
        for k, v in lpips_map.items():
            if k.lower() in model.lower() or model.lower() in k.lower():
                lpips_val = v
                break

        rows.append({
            "Model":           model,
            "CLIP ↓":         clip_map.get(model, NA),
            "ResNet-acc ↓":   classify_map.get(model, NA),
            "LPIPS ↑":        lpips_val,
            "FID ↓":          fid_map.get(model, NA),
        })

    df = pd.DataFrame(rows)
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("\n=== Summary Table ===")
    print(df.to_string(index=False))
    print(f"\nSaved → {out}")
    print("\nMetric guide:")
    print("  CLIP ↓       : Lower = Van Gogh style more erased")
    print("  ResNet-acc ↓ : Lower = Erased model less often classified as Van Gogh")
    print("  LPIPS ↑      : Higher = More different from original (more erasure)")
    print("  FID ↓        : Lower = Less drift from original SD quality")


if __name__ == "__main__":
    main()
