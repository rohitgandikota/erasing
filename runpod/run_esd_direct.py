"""
Direct ESD inference script — loads the .pt checkpoint with torch.load + unet.load_state_dict,
bypassing apply_esd_checkpoint. Produces the same output directory structure as generate-images.py.
"""
import argparse
import os

import pandas as pd
import torch
from diffusers import StableDiffusionPipeline

torch.set_grad_enabled(False)


def make_generator(device, seed):
    return torch.Generator(device=torch.device(device)).manual_seed(seed)


def generate(pipe, df, save_dir, device, guidance_scale, num_inference_steps, num_samples):
    os.makedirs(save_dir, exist_ok=True)
    for _, row in df.iterrows():
        prompt = [str(row.prompt)] * num_samples
        seed   = int(row.evaluation_seed)
        case   = int(row.case_number)
        images = pipe(
            prompt,
            generator=make_generator(device, seed),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images
        for idx, img in enumerate(images):
            img.save(os.path.join(save_dir, f"{case}_{idx}.png"))
        print(f"  case {case}: {row.prompt}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",          default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--esd_path",            default=None,         help=".pt checkpoint (original format)")
    parser.add_argument("--prompts_path",        required=True)
    parser.add_argument("--save_path",           default="outputs")
    parser.add_argument("--device",              default="cuda:0")
    parser.add_argument("--guidance_scale",      type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int,   default=50)
    parser.add_argument("--num_samples",         type=int,   default=5)
    args = parser.parse_args()

    df = pd.read_csv(args.prompts_path)

    print("==> Loading pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model, torch_dtype=torch.float16, safety_checker=None
    ).to(args.device)
    pipe.set_progress_bar_config(disable=True)

    if args.esd_path is None:
        model_name = "sdv14"
        print("==> Pass 1 — original SD v1.4")
    else:
        model_name = os.path.basename(args.esd_path).split(".")[0]
        print(f"==> Loading ESD weights from {args.esd_path} ...")
        state_dict = torch.load(args.esd_path, map_location="cpu", weights_only=False)
        # Convert to float16 to match the pipeline dtype
        state_dict = {k: v.to(torch.float16) for k, v in state_dict.items()}
        missing, unexpected = pipe.unet.load_state_dict(state_dict, strict=False)
        print(f"  Loaded: {len(state_dict)} keys | missing={len(missing)} unexpected={len(unexpected)}")
        print(f"==> Pass 2 — ESD erased ({model_name})")

    save_dir = os.path.join(args.save_path, model_name)
    generate(pipe, df, save_dir, args.device, args.guidance_scale, args.num_inference_steps, args.num_samples)
    print(f"  Saved to {save_dir}/")


if __name__ == "__main__":
    main()
