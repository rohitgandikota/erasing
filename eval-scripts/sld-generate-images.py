from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import LMSDiscreteScheduler
import torch
from PIL import Image
import pandas as pd
import argparse
from sld import SLDPipeline
import os
def generate_SLD(sld_concept, prompts_path, save_path, device='cuda:0', guidance_scale = 7.5, image_size=512, ddim_steps=100, num_samples=5, from_case=0):
    
    pipe = SLDPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)
    
    gen = torch.Generator(device=device)
    pipe.safety_concept = sld_concept
    
    torch_device = device
    df = pd.read_csv(prompts_path)
    prompts = df.prompt
    seeds = df.evaluation_seed
    case_numbers = df.case_number

    folder_path = f"{save_path}/{sld_concept.replace(' ','')}"
    os.makedirs(folder_path, exist_ok=True)

    for _, row in df.iterrows():
        prompt = str(row.prompt)
        seed = row.evaluation_seed
        case_number = row.case_number
        if int(case_number) not in [7,19,36,38,42,45,54,74,96,97]:
        #if case_number<from_case:
            continue

        height = image_size                        # default height of Stable Diffusion
        width = image_size                         # default width of Stable Diffusion

        num_inference_steps = ddim_steps           # Number of denoising steps

        guidance_scale = guidance_scale            # Scale for classifier-free guidance

        generator = torch.manual_seed(seed)    # Seed generator to create the inital latent noise

        batch_size = len(prompt)

        gen.manual_seed(seed)

        out = pipe(prompt=prompt, generator=gen, seed = seed, guidance_scale=guidance_scale, num_images_per_prompt=num_samples)
        for num, im in enumerate(out):
            im.save(f"{folder_path}/{case_number}_{num}.png")

if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'generateImages',
                    description = 'Generate Images using Diffusers Code')
    parser.add_argument('--sld_concept', help='concept to remove from SLD', type=str, required=True)
    parser.add_argument('--prompts_path', help='path to csv file with prompts', type=str, required=True)
    parser.add_argument('--save_path', help='folder where to save images', type=str, required=True)
    parser.add_argument('--device', help='cuda device to run on', type=str, required=False, default='cuda:0')
    parser.add_argument('--guidance_scale', help='guidance to run eval', type=float, required=False, default=7.5)
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--from_case', help='continue generating from case_number', type=int, required=False, default=0)
    parser.add_argument('--num_samples', help='number of samples per prompt', type=int, required=False, default=5)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=100)
    args = parser.parse_args()
    
    sld_concept = args.sld_concept
    prompts_path = args.prompts_path
    save_path = args.save_path
    device = args.device
    guidance_scale = args.guidance_scale
    image_size = args.image_size
    ddim_steps = args.ddim_steps
    num_samples= args.num_samples
    from_case = args.from_case
    generate_SLD(sld_concept, prompts_path, save_path, device=device,
                    guidance_scale = guidance_scale, image_size=image_size, ddim_steps=ddim_steps, num_samples=num_samples,from_case=from_case)
