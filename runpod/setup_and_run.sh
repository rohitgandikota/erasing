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
WORKDIR="/workspace/erasing"
WEIGHTS_PATH="$WORKDIR/esd-models/art/diffusers-VanGogh-ESDx1-UNET.pt"
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

# ── Install deps pinned to torch 2.4.x era ───────────────────────────────────
# We do NOT install torch/torchvision/torch_xla — use the pod's pre-installed versions.
# requirements.txt pins versions too new for torch 2.4.0, so we use a curated list.
echo "==> Installing compatible Python dependencies..."
pip install -q \
  "diffusers==0.27.2" \
  "transformers==4.41.2" \
  "accelerate==0.30.1" \
  "safetensors==0.4.3" \
  "pandas" "Pillow" "tqdm" "huggingface_hub"

# ── Assert CUDA works ─────────────────────────────────────────────────────────
echo "==> Checking CUDA..."
python3 - << 'PYEOF'
import torch
if not torch.cuda.is_available():
    print(f"ERROR: CUDA not available. torch={torch.__version__}")
    print("  -> Wrong RunPod template. Use 'RunPod Pytorch 2.4.0' (CUDA 12.4).")
    raise SystemExit(1)
print(f"  OK: torch={torch.__version__}, CUDA={torch.version.cuda}, GPU={torch.cuda.get_device_name(0)}")
PYEOF

# ── Assert diffusers imports cleanly ─────────────────────────────────────────
echo "==> Verifying diffusers import..."
python3 -c "from diffusers import DiffusionPipeline; print('  diffusers OK')"

# ── HF login ─────────────────────────────────────────────────────────────────
echo "==> Authenticating with Hugging Face..."
python3 - << PYEOF
import os
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
print("  HF login OK")
PYEOF

# ── Download ESD weights ──────────────────────────────────────────────────────
echo "==> Downloading Van Gogh ESD weights (~3.2 GB)..."
mkdir -p "$(dirname "$WEIGHTS_PATH")"
if [ ! -f "$WEIGHTS_PATH" ]; then
  wget -q --show-progress -O "$WEIGHTS_PATH" "$WEIGHTS_URL"
else
  echo "  Weights already present."
fi

# ── Clean stale outputs ───────────────────────────────────────────────────────
echo "==> Cleaning previous outputs..."
rm -rf "$OUTPUTS"
mkdir -p "$OUTPUTS"

# ── Pass 1: original SD v1.4 ─────────────────────────────────────────────────
echo "==> Pass 1 — original SD v1.4..."
python3 evalscripts/generate-images.py \
  --base_model "$BASE_MODEL" \
  --prompts_path "$PROMPTS" \
  --save_path "$OUTPUTS" \
  --num_samples 5 \
  --num_inference_steps 50 \
  --guidance_scale 7.5 \
  --device cuda:0

# ── Pass 2: ESD Van Gogh-erased ───────────────────────────────────────────────
echo "==> Pass 2 — ESD Van Gogh-erased model..."
python3 evalscripts/generate-images.py \
  --base_model "$BASE_MODEL" \
  --esd_path "$WEIGHTS_PATH" \
  --prompts_path "$PROMPTS" \
  --save_path "$OUTPUTS" \
  --num_samples 5 \
  --num_inference_steps 50 \
  --guidance_scale 7.5 \
  --device cuda:0

# ── Comparison grid ───────────────────────────────────────────────────────────
echo "==> Building comparison grid..."
python3 runpod/compare_outputs.py \
  --original_dir "$OUTPUTS/sdv14" \
  --erased_dir "$OUTPUTS/diffusers-VanGogh-ESDx1-UNET" \
  --prompts_csv "$PROMPTS" \
  --out "$OUTPUTS/comparison_grid.png"

echo ""
echo "Done! Download: $OUTPUTS/comparison_grid.png"
