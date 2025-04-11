import os 
import torch
import sys
import random
from tqdm.auto import tqdm
from safetensors.torch import save_file
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
import argparse

sys.path.append('.')
from utils.sdxl_utils import esd_sdxl_call
StableDiffusionXLPipeline.__call__ = esd_sdxl_call

def load_sdxl_models(basemodel_id="stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16, device='cuda:0'):
    
    base_unet = UNet2DConditionModel.from_pretrained(basemodel_id, subfolder="unet").to(device, torch_dtype)
    base_unet.requires_grad_(False)
    
    esd_unet = UNet2DConditionModel.from_pretrained(basemodel_id, subfolder="unet").to(device, torch_dtype)
    pipe = StableDiffusionXLPipeline.from_pretrained(basemodel_id, unet=base_unet, torch_dtype=torch_dtype, use_safetensors=True).to(device)
    
    return pipe, base_unet, esd_unet

def get_esd_trainable_parameters(esd_unet, train_method='esd-x'):
    params = []
    param_names = []
    for name, param in esd_unet.named_parameters():
        if train_method == 'esd-x' and 'attn2' in name:
            params.append(param)
            param_names.append(name)
        if train_method == 'esd-u' and 'attn2' not in name:
            params.append(param)
            param_names.append(name)
        if train_method == 'esd-all' and 'block' in name:
            params.append(param)
            param_names.append(name)
        if train_method == 'esd-x-strict' and ('attn2.to_k' in name or 'attn2.to_v' in name):
            params.append(param)
            param_names.append(name)
    return params, param_names


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'TrainESD for SDXL',
                    description = 'Finetuning stable-diffusion-xl to erase the concepts')
    parser.add_argument('--erase_concept', help='concept to erase', type=str, required=True)
    parser.add_argument('--erase_from', help='target concept to erase from', type=str, required=False, default = None)
    parser.add_argument('--num_inference_steps', help='number of inference steps for diffusion model', type=int, required=False, default=20)
    parser.add_argument('--guidance_scale', help='guidance scale to run inference for diffusion model', type=float, required=False, default=7)
    
    parser.add_argument('--train_method', help='Type of method (esd-x, esd-u, esd-a, esd-x-strict)', type=str, required=True)
    parser.add_argument('--iterations', help='Number of iterations', type=int, default=200)
    parser.add_argument('--lr', help='Learning rate', type=float, default=2e-4)
    parser.add_argument('--negative_guidance', help='Negative guidance value', type=float, required=False, default=1)
    parser.add_argument('--save_path', help='Path to save model', type=str, default='esd-models/sdxl/')
    parser.add_argument('--device', help='cuda device to train on', type=str, required=False, default='cuda:0')

    args = parser.parse_args()

    erase_concept = args.erase_concept
    erase_concept_from = args.erase_from

    num_inference_steps = args.num_inference_steps
    
    guidance_scale = args.guidance_scale
    negative_guidance = args.negative_guidance
    train_method=args.train_method
    iterations = args.iterations
    batchsize = 1
    height=width=1024
    lr = args.lr
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    device = args.device
    torch_dtype = torch.bfloat16
    
    criteria = torch.nn.MSELoss()

    pipe, base_unet, esd_unet = load_sdxl_models(basemodel_id="stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch_dtype, device=device)
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler.set_timesteps(num_inference_steps)

    esd_params, esd_param_names = get_esd_trainable_parameters(esd_unet, train_method=train_method)
    optimizer = torch.optim.Adam(esd_params, lr=lr)

    # get prompt embeds
    with torch.no_grad():
        # for concept prompt and null prompt
        erase_embeds, null_embeds, erase_pooled_embeds, null_pooled_embeds = pipe.encode_prompt(prompt=erase_concept,
                                                                                                device=device,
                                                                                                num_images_per_prompt=batchsize,
                                                                                                do_classifier_free_guidance=True,
                                                                                                negative_prompt="",
                                                                                                )
        erase_embeds = erase_embeds.to(device)
        null_embeds = null_embeds.to(device)
        add_erase_embeds = erase_pooled_embeds.to(device)
        add_null_embeds = null_pooled_embeds.to(device)
        if pipe.text_encoder_2 is None:
            text_encoder_projection_dim = int(erase_pooled_embeds.shape[-1])
        else:
            text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim
        add_time_ids = pipe._get_add_time_ids(
            (height,width),
            (0,0),
            (height,width),
            dtype=erase_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        add_time_ids = add_time_ids.to(device).repeat(batchsize, 1)
        timestep_cond = None
        if pipe.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(batchsize)
            timestep_cond = pipe.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=pipe.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=xt.dtype)
        
        
        
        
        
        if erase_concept_from is not None:
            erase_from_embeds, _, erase_from_pooled_embeds, _ = pipe.encode_prompt(prompt=erase_concept_from,
                                                                                    device=device,
                                                                                    num_images_per_prompt=batchsize,
                                                                                    do_classifier_free_guidance=False,
                                                                                    negative_prompt="",
                                                                                    )
            add_erase_from_embeds = erase_from_pooled_embeds.to(device)
            erase_from_embeds = erase_from_embeds.to(device)
    

    # Start Training
    pbar = tqdm(range(iterations), desc='Training ESD')
    losses = []
    for iteration in pbar:
        optimizer.zero_grad()
        # get the noise predictions for erase concept
        pipe.unet = base_unet
        run_till_timestep = random.randint(0, num_inference_steps-1)
        run_till_timestep_scheduler = pipe.scheduler.timesteps[run_till_timestep]
        seed = random.randint(0, 2**15)
        with torch.no_grad():
            xt = pipe(erase_concept if erase_concept_from is None else erase_concept_from,
                  num_images_per_prompt=batchsize,
                  num_inference_steps=num_inference_steps,
                  guidance_scale=guidance_scale,
                  run_till_timestep = run_till_timestep,
                  generator=torch.Generator().manual_seed(seed),
                  output_type='latent',
                  height=height,
                  width=width,
                 ).images
            
            added_cond_kwargs = {"text_embeds": add_erase_embeds, "time_ids": add_time_ids}
            noise_pred_erase = pipe.unet(
                xt,
                run_till_timestep_scheduler,
                encoder_hidden_states=erase_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=None,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
            
            # get the noise predictions for null embeds
            added_cond_kwargs = {"text_embeds": add_null_embeds, "time_ids": add_time_ids}
            noise_pred_null = pipe.unet(
                xt,
                run_till_timestep_scheduler,
                encoder_hidden_states=null_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=None,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
            
            # get the noise predictions for erase concept from embeds
            if erase_concept_from is not None:
                added_cond_kwargs = {"text_embeds": add_erase_from_embeds, "time_ids": add_time_ids}
                noise_pred_erase_from = pipe.unet(
                    xt,
                    run_till_timestep_scheduler,
                    encoder_hidden_states=erase_from_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=None,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
            else:
                noise_pred_erase_from = noise_pred_erase

        # get noise prediction from esd model for the concept being erased
        pipe.unet = esd_unet
        added_cond_kwargs = {"text_embeds": add_erase_embeds if erase_concept_from is None else add_erase_from_embeds, "time_ids": add_time_ids}
        noise_pred_esd_model = pipe.unet(
            xt,
            run_till_timestep_scheduler,
            encoder_hidden_states=erase_embeds if erase_concept_from is None else erase_from_embeds,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=None,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
        
        
        loss = criteria(noise_pred_esd_model, noise_pred_erase_from - (negative_guidance*(noise_pred_erase - noise_pred_null))) 
        loss.backward()
        losses.append(loss.item())
        pbar.set_postfix(esd_loss=loss.item(),
                         timestep=run_till_timestep,)
        optimizer.step()
    
    param_dict = {}
    for name, param in zip(esd_param_names, esd_params):
        param_dict[name] = param

    if erase_concept_from is None:
        erase_concept_from = erase_concept
    save_file(param_dict, f"{save_path}/esd-{erase_concept.replace(' ', '_')}-from-{erase_concept_from.replace(' ', '_')}.safetensors")