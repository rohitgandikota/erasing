from PIL import Image
from matplotlib import pyplot as plt
import textwrap
import argparse
import torch
import copy
import os
import re
import numpy as np
from diffusers import AutoencoderKL, UNet2DConditionModel
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPFeatureExtractor
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_lms_discrete import LMSDiscreteScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from utils.utils import *

def train(erase_concept, erase_from, train_method, iterations, negative_guidance, lr, save_path):
  
    nsteps = 50

    diffuser = StableDiffuser(scheduler='DDIM').to('cuda')
    diffuser.train()

    finetuner = FineTunedModel(diffuser, train_method=train_method)

    optimizer = torch.optim.Adam(finetuner.parameters(), lr=lr)
    criteria = torch.nn.MSELoss()

    pbar = tqdm(range(iterations))
    erase_concept = erase_concept.split(',')
    erase_concept = [a.strip() for a in erase_concept]
    
    erase_from = erase_from.split(',')
    erase_from = [a.strip() for a in erase_from]
    
    
    if len(erase_from)!=len(erase_concept):
        if len(erase_from) == 1:
            c = erase_from[0]
            erase_from = [c for _ in erase_concept]
        else:
            print(erase_from, erase_concept)
            raise Exception("Erase from concepts length need to match erase concepts length")
            
    erase_concept_ = []
    for e, f in zip(erase_concept, erase_from):
        erase_concept_.append([e,f])
    
    
    
    erase_concept = erase_concept_
    
    
    
    print(erase_concept)

    torch.cuda.empty_cache()

    for i in pbar:
        with torch.no_grad():
            index = np.random.choice(len(erase_concept), 1, replace=False)[0]
            erase_concept_sampled = erase_concept[index]
            
            
            neutral_text_embeddings = diffuser.get_text_embeddings([''],n_imgs=1)
            positive_text_embeddings = diffuser.get_text_embeddings([erase_concept_sampled[0]],n_imgs=1)
            target_text_embeddings = diffuser.get_text_embeddings([erase_concept_sampled[1]],n_imgs=1)
        

            diffuser.set_scheduler_timesteps(nsteps)

            optimizer.zero_grad()

            iteration = torch.randint(1, nsteps - 1, (1,)).item()

            latents = diffuser.get_initial_latents(1, 512, 1)

            with finetuner:

                latents_steps, _ = diffuser.diffusion(
                    latents,
                    positive_text_embeddings,
                    start_iteration=0,
                    end_iteration=iteration,
                    guidance_scale=3, 
                    show_progress=False
                )

            diffuser.set_scheduler_timesteps(1000)

            iteration = int(iteration / nsteps * 1000)
            
            positive_latents = diffuser.predict_noise(iteration, latents_steps[0], positive_text_embeddings, guidance_scale=1)
            neutral_latents = diffuser.predict_noise(iteration, latents_steps[0], neutral_text_embeddings, guidance_scale=1)
            target_latents = diffuser.predict_noise(iteration, latents_steps[0], target_text_embeddings, guidance_scale=1)
            if erase_concept_sampled[0] == erase_concept_sampled[1]:
                target_latents = neutral_latents.clone().detach()
        with finetuner:
            negative_latents = diffuser.predict_noise(iteration, latents_steps[0], target_text_embeddings, guidance_scale=1)

        positive_latents.requires_grad = False
        neutral_latents.requires_grad = False
        

        loss = criteria(negative_latents, target_latents - (negative_guidance*(positive_latents - neutral_latents))) 
        
        loss.backward()
        optimizer.step()

    torch.save(finetuner.state_dict(), save_path)

    del diffuser, loss, optimizer, finetuner, negative_latents, neutral_latents, positive_latents, latents_steps, latents

    torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'TrainESD',
                    description = 'Finetuning stable diffusion to erase the concepts')
    parser.add_argument('--erase_concept', help='concept to erase', type=str, required=True)
    parser.add_argument('--erase_from', help='target concept to erase from', type=str, required=False, default = None)
    parser.add_argument('--train_method', help='Type of method (xattn, noxattn, full, xattn-strict', type=str, required=True)
    parser.add_argument('--iterations', help='Number of iterations', type=int, default=200)
    parser.add_argument('--lr', help='Learning rate', type=float, default=2e-5)
    parser.add_argument('--negative_guidance', help='Negative guidance value', type=float, required=False, default=1)
    parser.add_argument('--save_path', help='Path to save model', type=str, default='models/')
    parser.add_argument('--device', help='cuda device to train on', type=str, required=False, default='cuda:0')

    args = parser.parse_args()
    
    prompt = args.erase_concept #'car'
    erase_concept = args.erase_concept
    erase_from = args.erase_from
    if erase_from is None:
        erase_from = erase_concept
    train_method = args.train_method #'noxattn'
    iterations = args.iterations #200
    negative_guidance = args.negative_guidance #1
    lr = args.lr #1e-5
    name = f"esd-{erase_concept.lower().replace(' ','').replace(',','')}_from_{erase_from.lower().replace(' ','').replace(',','')}-{train_method}_{negative_guidance}-epochs_{iterations}"
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok = True)
    save_path = f'{args.save_path}/{name}.pt'
    train(erase_concept=erase_concept, erase_from=erase_from, train_method=train_method, iterations=iterations, negative_guidance=negative_guidance, lr=lr, save_path=save_path)