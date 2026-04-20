"""
FID evaluation using clean-fid (cleanfid).

Computes FID between a generated image directory and a reference set.
With only 25 images FID is unreliable (needs ~2000+); use --warn_small to note this.

Reference modes:
  --ref_dir PATH         : Compare against another local directory (e.g. outputs/sdv14 as reference).
  --ref_dataset coco     : Use clean-fid's built-in COCO-val-2017 stats (requires download on first run).

Usage:
    # FID of ESD vs original SD (measures drift)
    python evalscripts/fid_eval.py \
        --gen_dir outputs/diffusers-VanGogh-ESDx1-UNET \
        --ref_dir outputs/sdv14 \
        --output_csv results/fid_esd.csv

    # FID of all conditions vs original SD
    python evalscripts/fid_eval.py \
        --gen_dirs outputs/diffusers-VanGogh-ESDx1-UNET outputs/space-Van_Gogh \
        --ref_dir outputs/sdv14 \
        --output_csv results/fid_all.csv

    # FID vs COCO val (general image quality)
    python evalscripts/fid_eval.py \
        --gen_dirs outputs/sdv14 outputs/diffusers-VanGogh-ESDx1-UNET outputs/space-Van_Gogh \
        --ref_dataset coco \
        --output_csv results/fid_coco.csv

Install:  pip install clean-fid
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd

MIN_RELIABLE_IMAGES = 2000


def count_images(d: Path) -> int:
    return len(list(d.glob("*.png")) + list(d.glob("*.jpg")))


def compute_fid_local(gen_dir: Path, ref_dir: Path, device: str) -> float:
    from cleanfid import fid
    return fid.compute_fid(str(gen_dir), str(ref_dir), device=device, use_dataparallel=False)


def compute_fid_coco(gen_dir: Path, device: str) -> float:
    from cleanfid import fid
    return fid.compute_fid(str(gen_dir), dataset_name="coco_val", dataset_res=256,
                            dataset_split="custom", device=device, use_dataparallel=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir",      default=None,       help="Single generated image directory")
    parser.add_argument("--gen_dirs",     nargs="+",          help="Multiple generated dirs")
    parser.add_argument("--ref_dir",      default=None,       help="Reference image directory (local)")
    parser.add_argument("--ref_dataset",  default=None,       help="Built-in reference dataset (e.g. 'coco')")
    parser.add_argument("--output_csv",   default="results/fid_scores.csv")
    parser.add_argument("--device",       default="cuda")
    args = parser.parse_args()

    try:
        import cleanfid  # noqa: F401
    except ImportError:
        raise ImportError("Install clean-fid: pip install clean-fid")

    gen_dirs = args.gen_dirs if args.gen_dirs else ([args.gen_dir] if args.gen_dir else None)
    if not gen_dirs:
        parser.error("Provide --gen_dir or --gen_dirs")
    if not args.ref_dir and not args.ref_dataset:
        parser.error("Provide --ref_dir or --ref_dataset")

    rows = []
    for d in gen_dirs:
        d = Path(d)
        n = count_images(d)
        if n < MIN_RELIABLE_IMAGES:
            warnings.warn(
                f"{d.name}: only {n} images — FID is unreliable below {MIN_RELIABLE_IMAGES}. "
                "Generate more images for meaningful FID. Reporting anyway.",
                stacklevel=2,
            )
        print(f"Computing FID for {d.name} ({n} images)...")
        if args.ref_dir:
            score = compute_fid_local(d, Path(args.ref_dir), args.device)
            ref_label = Path(args.ref_dir).name
        else:
            score = compute_fid_coco(d, args.device)
            ref_label = "coco_val"

        print(f"  FID({d.name} vs {ref_label}) = {score:.2f}  [n={n}]")
        rows.append({"model": d.name, "fid": round(score, 2), "n_images": n, "reference": ref_label})

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nFID scores → {out}")
    if any(r["n_images"] < MIN_RELIABLE_IMAGES for r in rows):
        print(f"NOTE: FID scores with <{MIN_RELIABLE_IMAGES} images are indicative only.")


if __name__ == "__main__":
    main()
