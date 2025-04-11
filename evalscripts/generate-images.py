from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DiffusionPipeline
import torch
from PIL import Image
import pandas as pd
import argparse
import os
from safetensors.torch import load_file
torch.enable_grad(False)

def generate_images(base_model, esd_path, prompts_path, save_path, device='cuda:0', torch_dtype=torch.bfloat16, guidance_scale = 7.5, num_inference_steps=100, num_samples=10, from_case=0):
    '''
    Function to generate images from diffusers code
    
    The program requires the prompts to be in a csv format with headers 
        1. 'case_number' (used for file naming of image)
        2. 'prompt' (the prompt used to generate image)
        3. 'seed' (the inital seed to generate gaussion noise for diffusion input)
    
    Parameters
    ----------
    base_model : str
        name of the model to load.
    esd_path : str
        path for the esd model to load. Leave as None if you want to test original model
    prompts_path : str
        path for the csv file with prompts and corresponding seeds.
    save_path : str
        save directory for images.
    device : str, optional
        device to be used to load the model. The default is 'cuda:0'.
    guidance_scale : float, optional
        guidance value for inference. The default is 7.5.
    num_inference_steps : int, optional
        number of denoising steps. The default is 100.
    num_samples : int, optional
        number of samples generated per prompt. The default is 10.
    from_case : int, optional
        The starting offset in csv to generate images. The default is 0.

    Returns
    -------
    None.

    '''
    if esd_path is not None:
        model_name = os.path.basename(esd_path).split('.')[0]
    else:
        if 'xl' in base_model:
            model_name = 'sdxl'
        elif 'Comp' in base_model:
            model_name = 'sdv14'
        else:
            model_name = 'custom'
    
    pipe = DiffusionPipeline.from_pretrained(base_model, torch_dtype=torch_dtype).to(device)
    if esd_path is not None:
        try:
            esd_weights = load_file(esd_path)
            pipe.unet.load_state_dict(esd_weights, strict=False)
        except:
            raise Exception('Please load the correct base model for your esd file')
            
    df = pd.read_csv(prompts_path)

    folder_path = f'{save_path}/{model_name}'
    os.makedirs(folder_path, exist_ok=True)

    for _, row in df.iterrows():
        prompt = [str(row.prompt)]*num_samples
        seed = row.evaluation_seed
        case_number = row.case_number
        if case_number<from_case:
            continue

        pil_images = pipe(prompt, 
                          generator=torch.Generator().manual_seed(seed),
                          num_inference_steps=num_inference_steps,
                          guidance_scale=guidance_scale).images
        for num, im in enumerate(pil_images):
            im.save(f"{folder_path}/{case_number}_{num}.png")

if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'generateImages',
                    description = 'Generate Images using Diffusers Code')
    parser.add_argument('--base_model', help='base model to load', type=str, required=False, default='stabilityai/stable-diffusion-xl-base-1.0')
    parser.add_argument('--esd_path', help='base model to load', type=str, required=False, default=None)
    parser.add_argument('--prompts_path', help='path to csv file with prompts', type=str, required=True)
    parser.add_argument('--save_path', help='folder where to save images', type=str, required=False, default='esd-images/')
    parser.add_argument('--device', help='cuda device to run on', type=str, required=False, default='cuda:0')
    parser.add_argument('--guidance_scale', help='guidance to run eval', type=float, required=False, default=7.5)
    parser.add_argument('--from_case', help='continue generating from case_number', type=int, required=False, default=0)
    parser.add_argument('--num_samples', help='number of samples per prompt', type=int, required=False, default=1)
    parser.add_argument('--num_inference_steps', help='ddim steps of inference used to train', type=int, required=False, default=20)
    args = parser.parse_args()
    
    base_model = args.base_model
    esd_path = args.esd_path
    prompts_path = args.prompts_path
    save_path = args.save_path
    device = args.device
    guidance_scale = args.guidance_scale
    num_inference_steps = args.num_inference_steps
    num_samples= args.num_samples
    from_case = args.from_case
    
    generate_images(base_model=base_model, esd_path=esd_path, prompts_path=prompts_path, save_path=save_path, device=device, guidance_scale = guidance_scale, num_inference_steps=num_inference_steps, num_samples=num_samples, from_case=from_case)
