# RunPod Instructions — ESD Van Gogh Replication

## What this does
Runs inference with the pre-trained ESD Van Gogh-erased model (ICCV 2023 paper
"Erasing Concepts from Diffusion Models") and produces a side-by-side comparison:
**Original SD v1.4** vs **ESD-erased (Van Gogh style removed)**.

---

## Step-by-step

### 1. Start a RunPod pod with these exact settings

| Setting | Value |
|---------|-------|
| GPU | RTX 4090 (24 GB) |
| Template | **RunPod PyTorch 2.4.1** (CUDA 12.4) |
| Container disk | **50 GB minimum** |
| Volume | not required |

> **Why 50 GB?** SD v1.4 = 5.5 GB, ESD weights = 3.2 GB, pip cache + outputs ≈ 10 GB.
> The generic PyTorch template ships a torch version incompatible with the available CUDA driver — use PyTorch 2.4.1 specifically.

### 2. Open the terminal

Click **Connect** → **JupyterLab** → open a **Terminal** tab.

### 3. Set your Hugging Face token

```bash
export HF_TOKEN=<your_huggingface_token>
```

Make sure you have [accepted the SD v1.4 license](https://huggingface.co/CompVis/stable-diffusion-v1-4) on Hugging Face first.

### 4. Run the setup script

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/Vedang-P/erasing/main/runpod/setup_and_run.sh)
```

The script will:
1. Clone this repo to `/workspace/erasing`
2. Install Python dependencies (skips torch/torchvision/torch_xla — uses pod's pre-installed versions)
3. Assert CUDA is available — fails immediately with a helpful message if not
4. Download the Van Gogh ESD weights (~3.2 GB)
5. Clean any previous outputs
6. Run **Pass 1**: 5 images per prompt with original SD v1.4
7. Run **Pass 2**: 5 images per prompt with ESD-erased model
8. Build `outputs/comparison_grid.png`

Expected runtime: **~20-30 minutes** on RTX 4090.

### 5. Download the result

In JupyterLab, navigate to:
```
/workspace/erasing/outputs/comparison_grid.png
```
Right-click → **Download**.

---

## Output layout

```
outputs/
  sdv14/                          # Original SD v1.4 images
    0_0.png  0_1.png ...          # case_number_sample_index
    1_0.png ...
  diffusers-VanGogh-ESDx1-UNET/  # ESD-erased images
    0_0.png  0_1.png ...
  comparison_grid.png             # Final side-by-side grid
```

## Prompts used (from the paper)

| Case | Prompt |
|------|--------|
| 0 | Art by Van Gogh |
| 1 | A design of Van Gogh |
| 2 | An image in the style of Van Gogh |
| 3 | A reproduction of the famous art of Van Gogh |
| 4 | Van Gogh painting of a landscape |

---

## Files added to this fork (no existing code changed)

| File | Purpose |
|------|---------|
| `runpod/setup_and_run.sh` | One-command RunPod bootstrap |
| `runpod/vangogh_prompts.csv` | Evaluation prompts |
| `runpod/compare_outputs.py` | Side-by-side image grid |
| `RUNPOD_INSTRUCTIONS.md` | This file |
