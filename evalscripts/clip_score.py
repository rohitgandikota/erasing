"""
CLIP image-text similarity score for erasure evaluation.

For each image in --image_dir, computes cosine similarity between the image embedding
and the text embedding of --concept_text. Lower score = concept more thoroughly erased.

Usage:
    python evalscripts/clip_score.py \
        --image_dir outputs/sdv14 \
        --concept_text "a painting in the style of Van Gogh" \
        --output_csv results/clip_sdv14.csv

    # Run all three conditions at once:
    python evalscripts/clip_score.py \
        --image_dirs outputs/sdv14 outputs/diffusers-VanGogh-ESDx1-UNET outputs/space-Van_Gogh \
        --concept_text "a painting in the style of Van Gogh" \
        --output_csv results/clip_all.csv
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

try:
    import clip
except ImportError:
    raise ImportError("Install openai-clip: pip install git+https://github.com/openai/CLIP.git")


def load_images(image_dir: Path):
    paths = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
    return paths


def compute_clip_scores(image_paths, concept_text: str, anchor_text: str,
                        model, preprocess, device: str):
    concept_tok = clip.tokenize([concept_text]).to(device)
    anchor_tok  = clip.tokenize([anchor_text]).to(device)
    with torch.no_grad():
        concept_emb = model.encode_text(concept_tok)
        concept_emb = concept_emb / concept_emb.norm(dim=-1, keepdim=True)
        anchor_emb  = model.encode_text(anchor_tok)
        anchor_emb  = anchor_emb / anchor_emb.norm(dim=-1, keepdim=True)

    scores = []
    for path in image_paths:
        img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            img_emb = model.encode_image(img)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            clip_concept = (img_emb @ concept_emb.T).item()
            clip_anchor  = (img_emb @ anchor_emb.T).item()
        # style_delta: how much more like the concept than a generic anchor image.
        # Lower = more erased. Negative = model produces less Van-Gogh-like images
        # than a generic painting baseline, which is ideal.
        scores.append({
            "image":       path.name,
            "clip_score":  round(clip_concept, 4),
            "style_delta": round(clip_concept - clip_anchor, 4),
        })
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir",    default=None, help="Single image directory")
    parser.add_argument("--image_dirs",   nargs="+",    help="Multiple image directories")
    parser.add_argument("--concept_text", default="a painting in the style of Van Gogh")
    parser.add_argument("--anchor_text",  default="a painting",
                        help="Neutral reference text; style_delta = clip_concept - clip_anchor")
    parser.add_argument("--output_csv",   default="results/clip_scores.csv")
    parser.add_argument("--clip_model",   default="ViT-L/14")
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dirs = args.image_dirs if args.image_dirs else ([args.image_dir] if args.image_dir else None)
    if not dirs:
        parser.error("Provide --image_dir or --image_dirs")

    print(f"Loading CLIP {args.clip_model}...")
    model, preprocess = clip.load(args.clip_model, device=args.device)
    model.eval()

    summary_rows = []
    all_rows = []
    for d in dirs:
        d = Path(d)
        paths = load_images(d)
        if not paths:
            print(f"  WARNING: no images found in {d}")
            continue
        scores = compute_clip_scores(paths, args.concept_text, args.anchor_text,
                                     model, preprocess, args.device)
        for s in scores:
            s["model"] = d.name
        all_rows.extend(scores)
        mean_clip  = sum(s["clip_score"]  for s in scores) / len(scores)
        mean_delta = sum(s["style_delta"] for s in scores) / len(scores)
        print(f"  {d.name:45s}  CLIP={mean_clip:.4f}  style_delta={mean_delta:.4f}  (n={len(scores)})")
        summary_rows.append({
            "model":            d.name,
            "mean_clip_score":  round(mean_clip,  4),
            "mean_style_delta": round(mean_delta, 4),
            "n_images":         len(scores),
        })

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out, index=False)

    summary_path = out.parent / (out.stem + "_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\nPer-image scores  → {out}")
    print(f"Summary           → {summary_path}")
    print(f"\nConcept: '{args.concept_text}'  |  Anchor: '{args.anchor_text}'")
    print("style_delta = CLIP(concept) - CLIP(anchor).  Lower/negative = more erased.")


if __name__ == "__main__":
    main()
