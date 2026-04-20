"""
SPACE training for Stable Diffusion v1.x.
Semantically Precise Attribute Concept Erasure uses per-prompt semantic anchors
while keeping the same on-trajectory training setup as ESD-x.

Usage:
    python space_sd.py --erase_concept "Van Gogh"
    python space_sd.py --erase_concept "Van Gogh" --space_pairs_path data/space_pairs/vangogh.json
    python space_sd.py --erase_concept "Van Gogh" --protect_concept "a photograph" --eta 2.0
"""

import argparse

import torch

from utils.esd_trainer import ESDConfig, run_esd_training


def main():
    parser = argparse.ArgumentParser(description="Train SPACE concept erasure for SD v1.x")
    parser.add_argument("--erase_concept",    required=True,  help="Concept to erase (e.g. 'Van Gogh')")
    parser.add_argument("--basemodel_id",     default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--space_pairs_path", default=None,   help="Path to JSON with 20 concept/anchor pairs. Auto-detected from erase_concept if omitted.")
    parser.add_argument("--protect_concept",  default=None,   help="Override the protection concept from the pairs JSON.")
    parser.add_argument("--eta",              type=float, default=2.0, help="Erasure strength η")
    parser.add_argument("--iterations",       type=int,   default=1000)
    parser.add_argument("--lr",               type=float, default=2e-5)
    parser.add_argument(
        "--train_method",
        default="esd-x",
        choices=["esd-x", "esd-x-strict", "xattn", "xattn-strict"],
        help="Trainable cross-attention surface. Defaults to esd-x to match the ESD-x baseline capacity.",
    )
    parser.add_argument("--save_path",        default="esd-models/space/")
    parser.add_argument("--device",           default="cuda:0")
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--pres_lambda",      type=float, default=0.0,
                        help="Weight for preservation loss (0 disables it; high values fight erasure when protection styles resemble erase concept)")
    parser.add_argument("--allow_tf32",       action="store_true")
    args = parser.parse_args()

    config = ESDConfig(
        family="space-sd",
        base_model_id=args.basemodel_id,
        erase_concept=args.erase_concept,
        erase_from=None,
        train_method=args.train_method,
        iterations=args.iterations,
        lr=args.lr,
        negative_guidance=args.eta,      # reused as η for SPACE
        num_inference_steps=50,
        guidance_scale=7.5,
        batch_size=1,
        resolution=None,                 # auto-detected from model config
        save_path=args.save_path,
        device=args.device,
        torch_dtype=torch.bfloat16,
        gradient_clip_norm=args.gradient_clip_norm,
        protect_concept=args.protect_concept,
        space_pairs_path=args.space_pairs_path,
        pres_lambda=args.pres_lambda,
        allow_tf32=args.allow_tf32,
    )

    print(f"SPACE training: erasing '{args.erase_concept}' with η={args.eta}, {args.iterations} steps")
    checkpoint_path = run_esd_training(config)
    print(f"Saved checkpoint → {checkpoint_path}")


if __name__ == "__main__":
    main()
