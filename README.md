# iccv-esd
This work addresses the challenge of erasing artistic and other visual styles from stable diffusion models

## Installation Guide

* To get started clone the following repository of Original Stable Diffusion [Link](https://github.com/CompVis/stable-diffusion)
* Then download the files from our iccv-esd repository to `stable-diffusion` main directory of stable diffusion
* Download the weights from [here]([https://huggingface.co/CompVis/stable-diffusion-v-1-4-original](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4-full-ema.ckpt)) and move them to `stable-diffusion/models/ldm/` (This will be `ckpt_path` variable in `train-scripts/train-esd.py`)
* [Only for training] To convert your trained models to diffusers download the diffusers Unet config from [here](https://huggingface.co/CompVis/stable-diffusion-v1-4/blob/main/unet/config.json)  (This will be `diffusers_config_path` variable in `train-scripts/train-esd.py`)

## Training Guide

After installation, follow these instructions to train a custom ESD model:

* `cd stable-diffusion` to the main repository of stable-diffusion
* [IMPORTANT] Edit `train-script/train-esd.py` and change the default argparser values according to your convenience (especially the config paths)
* To choose train_method, pick from following `'xattn'`,`'noxattn'`, `'selfattn'`, `'full'` 
* `python train-scripts/train-esd.py --prompt 'your prompt' --train_method 'your choice of training' --devices '0,1'`

Note that the default argparser values must be changed!

## Generating Images

To generate images from one of the custom models use the following instructions:

* To use `eval-scripts/generate-images.py` you would need a csv file with columns `prompt`, `evaluation_seed` and `case_number`. (Sample data in `data/`)
* To generate multiple images per prompt use the argument `num_samples`. It is default to 10.
* The path to model can be customised in the script.
* It is to be noted that the current version requires the model to be in saved in `stable-diffusion/compvis-<based on hyperparameters>/diffusers-<based on hyperparameters>.pt`
* `python eval-scripts/generate-images.py --model_name='compvis-word_VanGogh-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05' --prompts_path 'stable-diffusion/art_prompts.csv' --save_path 'evaluation_folder'` 
