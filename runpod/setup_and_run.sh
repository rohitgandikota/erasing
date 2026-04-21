#!/usr/bin/env bash
# ESD Van Gogh replication — RunPod setup script
# Usage:
#   export HF_TOKEN=<your_huggingface_token>
#   bash <(curl -fsSL https://raw.githubusercontent.com/Vedang-P/erasing/main/runpod/setup_and_run.sh)

set -euo pipefail

# ── Validate HF token ────────────────────────────────────────────────────────
if [ -z "${HF_TOKEN-}" ]; then
  echo "ERROR: Set HF_TOKEN first:  export HF_TOKEN=<your_hf_token>"
  exit 1
fi

# ── Config ───────────────────────────────────────────────────────────────────
REPO_URL="https://github.com/Vedang-P/erasing.git"
WEIGHTS_URL="https://erasing.baulab.info/weights/esd_models/art/diffusers-VanGogh-ESDx1-UNET.pt"
BASE_MODEL="CompVis/stable-diffusion-v1-4"
WORKDIR="/workspace/erasing-space"
WEIGHTS_PT="$WORKDIR/esd-models/art/diffusers-VanGogh-ESDx1-UNET.pt"
WEIGHTS_ST="$WORKDIR/esd-models/art/diffusers-VanGogh-ESDx1-UNET.safetensors"
PROMPTS="$WORKDIR/runpod/vangogh_prompts.csv"
OUTPUTS="$WORKDIR/outputs"

# ── Clone repo ───────────────────────────────────────────────────────────────
echo "==> Cloning repository..."
if [ -d "$WORKDIR/.git" ]; then
  git -C "$WORKDIR" pull --ff-only
else
  git clone "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"

# ── Install deps (skip torch/torchvision/torch_xla — use pod's pre-installed versions) ──
# Use pinned versions compatible with torch 2.4–2.8.
echo "==> Installing Python dependencies..."
pip install -q \
  "diffusers==0.30.3" \
  "transformers==4.43.4" \
  "accelerate==0.33.0" \
  "safetensors==0.4.3" \
  pandas Pillow tqdm huggingface_hub

# ── Assert CUDA works ─────────────────────────────────────────────────────────
echo "==> Checking CUDA..."
python3 - << 'PYEOF'
import torch
if not torch.cuda.is_available():
    print(f"ERROR: CUDA not available. torch={torch.__version__}")
    print("  -> Use RunPod 'Pytorch 2.8.0' template (supports Blackwell GPUs).")
    raise SystemExit(1)
gpu = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
sm = f"sm_{cap[0]}{cap[1]}"
print(f"  OK: torch={torch.__version__}, CUDA={torch.version.cuda}, GPU={gpu} ({sm})")
# Fail early if GPU architecture is not supported by this torch build
props = torch.cuda.get_device_properties(0)
try:
    t = torch.zeros(1, device="cuda")
    _ = t + t
except RuntimeError as e:
    print(f"ERROR: GPU not usable: {e}")
    raise SystemExit(1)
PYEOF

# ── Assert diffusers imports cleanly ─────────────────────────────────────────
python3 -c "from diffusers import DiffusionPipeline; print('  diffusers OK')"

# ── HF login ─────────────────────────────────────────────────────────────────
echo "==> Authenticating with Hugging Face..."
python3 - << PYEOF
import os
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
print("  HF login OK")
PYEOF

# ── Download ESD weights (.pt pickle format) ──────────────────────────────────
echo "==> Downloading Van Gogh ESD weights (~3.2 GB)..."
mkdir -p "$(dirname "$WEIGHTS_PT")"
if [ ! -f "$WEIGHTS_PT" ]; then
  wget -q --show-progress -O "$WEIGHTS_PT" "$WEIGHTS_URL"
else
  echo "  Weights already present."
fi

# ── Convert .pt pickle → safetensors (repo now uses safetensors to load) ─────
echo "==> Converting weights to safetensors format..."
python3 - << PYEOF
import os, torch
from safetensors.torch import save_file

pt_path = "$WEIGHTS_PT"
st_path = "$WEIGHTS_ST"

if os.path.exists(st_path):
    print("  Already converted, skipping.")
else:
    print("  Loading .pt checkpoint...")
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    state_dict = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
    save_file(state_dict, st_path, metadata={"format": "pt"})
    print(f"  Saved: {st_path}")
PYEOF

mkdir -p "$OUTPUTS"

# ── Pass 1: original SD v1.4 (skip if already generated) ─────────────────────
if [ -f "$OUTPUTS/sdv14/0_0.png" ]; then
  echo "==> Pass 1 (Original SD) already present — skipping."
else
  echo "==> Pass 1 — original SD v1.4..."
  python3 evalscripts/generate-images.py \
    --base_model "$BASE_MODEL" \
    --prompts_path "$PROMPTS" \
    --save_path "$OUTPUTS" \
    --num_samples 5 \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --device cuda:0
fi

# ── Pass 2: ESD Van Gogh-erased (skip if already generated) ──────────────────
if [ -f "$OUTPUTS/diffusers-VanGogh-ESDx1-UNET/0_0.png" ]; then
  echo "==> Pass 2 (ESD) already present — skipping."
else
  echo "==> Pass 2 — ESD Van Gogh-erased model..."
  python3 evalscripts/generate-images.py \
    --base_model "$BASE_MODEL" \
    --esd_path "$WEIGHTS_ST" \
    --prompts_path "$PROMPTS" \
    --save_path "$OUTPUTS" \
    --num_samples 5 \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --device cuda:0
fi

# ── Comparison grid ───────────────────────────────────────────────────────────
echo "==> Building comparison grid..."
python3 runpod/compare_outputs.py \
  --original_dir "$OUTPUTS/sdv14" \
  --erased_dir "$OUTPUTS/diffusers-VanGogh-ESDx1-UNET" \
  --prompts_csv "$PROMPTS" \
  --out "$OUTPUTS/comparison_grid.png"

echo ""
echo "Done! Download: $OUTPUTS/comparison_grid.png"
