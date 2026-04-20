#!/usr/bin/env bash
# Full SPACE vs ESD comparison pipeline — RunPod
#
# Runs all 3 conditions (Original SD, ESD-x, SPACE) and all 4 eval metrics:
#   CLIP score, ResNet50 classification accuracy, LPIPS, FID (vs original SD outputs)
#
# Prerequisites (should already be set up from setup_and_run.sh):
#   - /workspace/erasing cloned and deps installed
#   - $HF_TOKEN set
#   - ESD weights at esd-models/art/diffusers-VanGogh-ESDx1-UNET.safetensors
#
# Usage:
#   cd /workspace/erasing
#   bash runpod/run_space_comparison.sh
#
# Outputs:
#   outputs/sdv14/                        Original SD images (reuse if present)
#   outputs/diffusers-VanGogh-ESDx1-UNET/ ESD images (reuse if present)
#   outputs/space-Van_Gogh/               SPACE images (generated here)
#   outputs/space-Van_Gogh.safetensors    SPACE checkpoint
#   results/                              All metric CSVs
#   outputs/comparison_grid_3way.png      3-condition visual comparison

set -euo pipefail

WORKDIR="/workspace/erasing"
BASE_MODEL="CompVis/stable-diffusion-v1-4"
PROMPTS="$WORKDIR/runpod/vangogh_prompts.csv"
OUTPUTS="$WORKDIR/outputs"
RESULTS="$WORKDIR/results"
ESD_WEIGHTS="$WORKDIR/esd-models/art/diffusers-VanGogh-ESDx1-UNET.safetensors"
SPACE_WEIGHTS="$WORKDIR/esd-models/space/space-Van_Gogh-esdxstrict.safetensors"
SPACE_OUTPUT_DIR="$OUTPUTS/space-Van_Gogh"
CONCEPT_TEXT="a painting in the style of Van Gogh"

cd "$WORKDIR"

# ── git pull to get latest code ───────────────────────────────────────────────
echo "==> Pulling latest code..."
git pull origin main

# ── install any new deps ──────────────────────────────────────────────────────
echo "==> Installing/verifying deps..."
pip install -q \
  "diffusers==0.30.3" "transformers==4.43.4" "accelerate==0.33.0" \
  "safetensors==0.4.3" pandas Pillow tqdm huggingface_hub lpips \
  "git+https://github.com/openai/CLIP.git" clean-fid torchvision

# ── CUDA check (functional — actually allocates on GPU) ──────────────────────
python3 - << 'PYEOF'
import torch
# torch.cuda.is_available() can return False in PyTorch 2.11 due to lazy-init
# quirks even when the GPU is present. Use device_count() and a real allocation.
n = torch.cuda.device_count()
if n == 0:
    print(f"ERROR: No CUDA GPU found. torch={torch.__version__}")
    raise SystemExit(1)
try:
    t = torch.zeros(1, device="cuda")
    _ = t + t
except Exception as e:
    print(f"ERROR: GPU allocation failed: {e}")
    raise SystemExit(1)
print(f"  OK: torch={torch.__version__}, GPU={torch.cuda.get_device_name(0)} ({n} device(s))")
PYEOF

mkdir -p "$RESULTS"

# ── Pass 1: Original SD v1.4 (skip if already generated) ─────────────────────
if [ -f "$OUTPUTS/sdv14/0_0.png" ]; then
  echo "==> Pass 1 (Original SD) already present — skipping generation."
else
  echo "==> Pass 1 — Original SD v1.4..."
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
  echo "==> Pass 2 (ESD) already present — skipping generation."
else
  echo "==> Pass 2 — ESD Van Gogh-erased..."
  python3 evalscripts/generate-images.py \
    --base_model "$BASE_MODEL" \
    --esd_path "$ESD_WEIGHTS" \
    --prompts_path "$PROMPTS" \
    --save_path "$OUTPUTS" \
    --num_samples 5 \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --device cuda:0
fi

# ── Train SPACE ───────────────────────────────────────────────────────────────
if [ -f "$SPACE_WEIGHTS" ]; then
  echo "==> SPACE checkpoint already present — skipping training."
else
  echo "==> Training SPACE (Van Gogh, 1000 steps, η=5.0)..."
  python3 space_sd.py \
    --erase_concept "Van Gogh" \
    --space_pairs_path data/space_pairs/vangogh.json \
    --eta 5.0 \
    --iterations 1000 \
    --lr 1e-5 \
    --save_path "esd-models/space" \
    --gradient_clip_norm 1.0 \
    --device cuda:0
  echo "  SPACE checkpoint saved: $SPACE_WEIGHTS"
fi

# ── Pass 3: SPACE Van Gogh-erased ────────────────────────────────────────────
if [ -f "$SPACE_OUTPUT_DIR/0_0.png" ]; then
  echo "==> Pass 3 (SPACE) already present — skipping generation."
else
  echo "==> Pass 3 — SPACE Van Gogh-erased..."
  python3 evalscripts/generate-images.py \
    --base_model "$BASE_MODEL" \
    --esd_path "$SPACE_WEIGHTS" \
    --prompts_path "$PROMPTS" \
    --save_path "$OUTPUTS" \
    --num_samples 5 \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --device cuda:0
fi

# ── Eval 1: CLIP score ────────────────────────────────────────────────────────
echo "==> Eval: CLIP score..."
python3 evalscripts/clip_score.py \
  --image_dirs "$OUTPUTS/sdv14" \
              "$OUTPUTS/diffusers-VanGogh-ESDx1-UNET" \
              "$SPACE_OUTPUT_DIR" \
  --concept_text "$CONCEPT_TEXT" \
  --output_csv "$RESULTS/clip_scores.csv" \
  --device cuda

# ── Eval 2: ResNet50 classification accuracy ──────────────────────────────────
echo "==> Eval: ResNet50 classification..."
for dir_name in sdv14 diffusers-VanGogh-ESDx1-UNET space-Van_Gogh; do
  python3 evalscripts/imageclassify.py \
    --folder_path "$OUTPUTS/$dir_name" \
    --prompts_path "$PROMPTS" \
    --save_path "$RESULTS/classify_${dir_name}.csv" \
    --device cuda:0
done

# ── Eval 3: LPIPS (erased vs original) ───────────────────────────────────────
echo "==> Eval: LPIPS..."
python3 evalscripts/lpips_eval.py \
  --original_path "$OUTPUTS/sdv14" \
  --edited_path "$OUTPUTS/diffusers-VanGogh-ESDx1-UNET" \
  --prompts_path "$PROMPTS" \
  --save_path "$RESULTS/lpips_esd.csv"

python3 evalscripts/lpips_eval.py \
  --original_path "$OUTPUTS/sdv14" \
  --edited_path "$SPACE_OUTPUT_DIR" \
  --prompts_path "$PROMPTS" \
  --save_path "$RESULTS/lpips_space.csv"

# ── Eval 4: FID (erased vs original SD outputs as reference) ─────────────────
echo "==> Eval: FID (vs original SD outputs)..."
python3 evalscripts/fid_eval.py \
  --gen_dirs "$OUTPUTS/diffusers-VanGogh-ESDx1-UNET" "$SPACE_OUTPUT_DIR" \
  --ref_dir "$OUTPUTS/sdv14" \
  --output_csv "$RESULTS/fid_scores.csv" \
  --device cuda

# ── 3-way comparison grid ─────────────────────────────────────────────────────
echo "==> Building 3-way comparison grid..."
python3 runpod/compare_outputs_3way.py \
  --dirs "$OUTPUTS/sdv14" \
         "$OUTPUTS/diffusers-VanGogh-ESDx1-UNET" \
         "$SPACE_OUTPUT_DIR" \
  --labels "Original SD v1.4" "ESD-x (paper)" "SPACE (ours)" \
  --prompts_csv "$PROMPTS" \
  --out "$OUTPUTS/comparison_grid_3way.png"

# ── Collect and print summary table ──────────────────────────────────────────
echo "==> Collecting results..."
python3 runpod/collect_results.py \
  --results_dir "$RESULTS" \
  --output_csv "$RESULTS/summary_table.csv"

echo ""
echo "======================================================================"
echo "  All done! Files to download:"
echo "    $OUTPUTS/comparison_grid_3way.png"
echo "    $RESULTS/summary_table.csv"
echo "======================================================================"
