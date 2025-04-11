# Erasing Concepts from Diffusion Models
###  [Project Website](https://erasing.baulab.info) | [Arxiv Preprint](https://arxiv.org/pdf/2303.07345.pdf) | [Fine-tuned Weights](https://erasing.baulab.info/weights/esd_models/) | [Demo](https://huggingface.co/spaces/baulab/Erasing-Concepts-In-Diffusion) <br>

### Updated code 🚀 - Now support diffusers!! (Faster and cleaner)

<div align='center'>
<img src = 'images/applications.png'>
</div>

## Code Update 🚀
We are releasing a cleaner code for ESD with diffusers support. Compared to our old-code this version uses almost half the GPU memory and is 5-8 times faster. Because of this diffusers support - we believe it allows to generalise to latest models (FLUX ESD coming soon ... ) <br>

To use the older version please visit our [older commit](https://github.com/rohitgandikota/erasing/tree/a2189e9ae677aca22a00c361bde25d3d320d8a61)

## Installation Guide
We recently updated our codebase to be much more cleaner and faster. The setup is also simple
```bash
git clone https://github.com/rohitgandikota/erasing.git
cd erasing
pip install -r requirements.txt
```

## Training Guide

### SDv1.4
After installation, follow these instructions to train a custom ESD model for Stable Diffusion V1.4. Pick from following `'xattn'`,`'noxattn'`, `'selfattn'`, `'full'`:
```python
python esd_diffusers.py --erase_concept 'Van Gogh' --train_method 'xattn'
```

💡 New application: You can now erase an attribute from a concept!! Instead of erasing a whole concept you can just precisely remove some of its attributes. For example, you can erase hats from cowboys but keep the rest intact!
```python
python esd_diffusers.py --erase_concept 'cowboy hat' --erase_from 'cowboy' --train_method 'xattn'
```

### SDXL
After installation, follow these instructions to train a custom ESD model for Stable Diffusion V1.4. Pick from following `'esd-x'`,`'esd-u'`, `'esd-a'`, `'esd-x-strict'`:
```python
python esd_sdxl.py --erase_concept 'Van Gogh' --train_method 'esd-x-strict'
```
The optimization process for erasing undesired visual concepts from pre-trained diffusion model weights involves using a short text description of the concept as guidance. The ESD model is fine-tuned with the conditioned and unconditioned scores obtained from frozen SD model to guide the output away from the concept being erased. The model learns from it's own knowledge to steer the diffusion process away from the undesired concept.
<div align='center'>
<img src = 'images/ESD.png'>
</div>

## Generating Images

Generating images from custom ESD model is super easy. Please follow `notebook/esd_inference_sdxl.ipynb` notebook

For an automated script to generate a ton of images for your evaluations use our evalscripts
```python
python evalscripts/generate-images.py --base_model 'stabilityai/stable-diffusion-xl-base-1.0' --esd_path 'esd-models/sdxl/esd-kelly-from-kelly.safetensors' --num_images 1 --prompts_path 'data/kelly_prompts.csv' --num_inference_steps 20 --guidance_scale 7
```

### UPDATE (NudeNet)
If you want to recreate the results from our paper on NSFW task - please use this https://drive.google.com/file/d/1J_O-yZMabmS9gBA2qsFSrmFoCl7tbFgK/view?usp=sharing

* Untar this file and save it in the homedirectory '~/.NudeNet'
* This should enable the right results as we use this checkpoint for our analysis.

## Running Gradio Demo Locally

To run the gradio interactive demo locally, clone the files from [demo repository](https://huggingface.co/spaces/baulab/Erasing-Concepts-In-Diffusion/tree/main) <br>

* Create an environment using the packages included in the requirements.txt file
* Run `python app.py`
* Open the application in browser at `http://127.0.0.1:7860/`
* Train, evaluate, and save models using our method
  
## Citing our work
The preprint can be cited as follows
```
@inproceedings{gandikota2023erasing,
  title={Erasing Concepts from Diffusion Models},
  author={Rohit Gandikota and Joanna Materzy\'nska and Jaden Fiotto-Kaufman and David Bau},
  booktitle={Proceedings of the 2023 IEEE International Conference on Computer Vision},
  year={2023}
}
```
