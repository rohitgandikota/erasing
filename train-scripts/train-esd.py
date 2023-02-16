from omegaconf import OmegaConf
import torch
from PIL import Image
from torchvision import transforms
import os
from tqdm import tqdm
from einops import rearrange
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
import random
import glob
import re
import shutil
import pdb
import argparse
from convertModels import savemodelDiffusers
# Util Functions
def load_model_from_config(config, ckpt, device="cpu", verbose=False):
    """Loads a model from config and a ckpt
    if config is a path will use omegaconf to load
    """
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    pl_sd = torch.load(ckpt, map_location="cpu")
    global_step = pl_sd["global_step"]
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    return model

@torch.no_grad()
def sample_model(model, sampler, c, h, w, ddim_steps, scale, ddim_eta, start_code=None, n_samples=1,t_start=-1,log_every_t=None,till_T=None,verbose=True):
    """Sample the model"""
    uc = None
    if scale != 1.0:
        uc = model.get_learned_conditioning(n_samples * [""])
    log_t = 100
    if log_every_t is not None:
        log_t = log_every_t
    shape = [4, h // 8, w // 8]
    samples_ddim, inters = sampler.sample(S=ddim_steps,
                                     conditioning=c,
                                     batch_size=n_samples,
                                     shape=shape,
                                     verbose=False,
                                     x_T=start_code,
                                     unconditional_guidance_scale=scale,
                                     unconditional_conditioning=uc,
                                     eta=ddim_eta,
                                     verbose_iter = verbose,
                                     t_start=t_start,
                                     log_every_t = log_t,
                                     till_T = till_T
                                    )
    if log_every_t is not None:
        return samples_ddim, inters
    return samples_ddim

def load_img(path, target_size=512):
    """Load an image, resize and output -1..1"""
    image = Image.open(path).convert("RGB")


    tform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ])
    image = tform(image)
    return 2.*image - 1.

def decode_to_im(samples, n_samples=1, nrow=1):
    """Decode a latent and return PIL image"""
    samples = model.decode_first_stage(samples)
    ims = torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0)
    x_sample = 255. * rearrange(ims.cpu().numpy(), '(n1 n2) c h w -> (n1 h) (n2 w) c', n1=n_samples//nrow, n2=nrow)
    return Image.fromarray(x_sample.astype(np.uint8))


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_loss(losses, path,word, n=100):
    v = moving_average(losses, n)
    plt.plot(v, label=f'{word}_loss')
    plt.legend(loc="upper left")
    plt.title('Average loss in trainings', fontsize=20)
    plt.xlabel('Data point', fontsize=16)
    plt.ylabel('Loss value', fontsize=16)
    plt.savefig(path)

##################### ESD Functions
def get_models(config_path, ckpt_path, devices):
    model_orig = load_model_from_config(config_path, ckpt_path, devices[1])
    sampler_orig = DDIMSampler(model_orig)

    model = load_model_from_config(config_path, ckpt_path, devices[0])
    sampler = DDIMSampler(model)

    return model_orig, sampler_orig, model, sampler

def train_esd(prompt, train_method, start_guidance, negative_guidance, iterations, lr, config_path, ckpt_path, diffusers_config_path, devices, seperator=None, image_size=512, ddim_steps=50, ddim_eta=0):
    # PROMPT CLEANING
    word_print = prompt.replace(' ','')
    if prompt == 'allartist':
        prompt = "Kelly Mckernan, Thomas Kinkade, Ajin Demi Human, Alena Aenami, Tyler Edlin, Kilian Eng"
    if prompt == 'i2p':
        prompt = "hate, harassment, violence, suffering, humiliation, harm, suicide, sexual, nudity, bodily fluids, blood"
    if prompt == "badimages":
        prompt = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy"

    if seperator is not None:
        words = prompt.split(seperator)
        words = [word.strip() for word in words]
    else:
        words = [prompt]
    print(words)

    # MODEL TRAINING SETUP

    model_orig, sampler_orig, model, sampler = get_models(config_path, ckpt_path, devices)

    # choose parameters to train based on train_method
    parameters = []
    for name, param in model.model.diffusion_model.named_parameters():
        # train all layers except x-attns and time_embed layers
        if train_method == 'noxattn':
            if name.startswith('out.') or 'attn2' in name or 'time_embed' in name:
                pass
            else:
                print(name)
                parameters.append(param)
        # train only self attention layers
        if train_method == 'selfattn':
            if 'attn1' in name:
                print(name)
                parameters.append(param)
        # train only x attention layers
        if train_method == 'xattn':
            if 'attn2' in name:
                print(name)
                parameters.append(param)
        # train all layers
        if train_method == 'full':
            print(name)
            parameters.append(param)
        # train all layers except time embed layers
        if train_method == 'notime':
            if not (name.startswith('out.') or 'time_embed' in name):
                print(name)
                parameters.append(param)
    # set model to train
    model.train()
    # create a lambda function for cleaner use of sampling code (only denoising till time step t)
    quick_sample_till_t = lambda x, s, code, t: sample_model(model, sampler,
                                                                 x, image_size, image_size, ddim_steps, s, ddim_eta,
                                                                 start_code=code, till_T=t, verbose=False)

    losses = []
    opt = torch.optim.Adam(parameters, lr=lr)
    criteria = torch.nn.MSELoss()
    history = []

    name = f'compvis-word_{word_print}-method_{train_method}-sg_{start_guidance}-ng_{negative_guidance}-iter_{iterations}-lr_{lr}'
    # TRAINING CODE
    pbar = tqdm(range(iterations))
    for i in pbar:
        word = random.sample(words,1)[0]
        emb_0 = model.get_learned_conditioning([''])
        emb_p = model.get_learned_conditioning([word])
        emb_n = model.get_learned_conditioning([f'{word}'])

        opt.zero_grad()

        t_enc = torch.randint(ddim_steps, (1,), device=devices[0])
        # time step from 1000 to 0 (0 being good)
        og_num = round((int(t_enc)/ddim_steps)*1000)
        og_num_lim = round((int(t_enc+1)/ddim_steps)*1000)

        t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])

        start_code = torch.randn((1, 4, 64, 64)).to(devices[0])

        with torch.no_grad():
            z = quick_sample_till_t(emb_p.to(devices[0]), start_guidance, start_code, int(t_enc)) # emb_p seems to work better instead of emb_0
            e_0 = model_orig.apply_model(z.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_0.to(devices[1]))
            e_p = model_orig.apply_model(z.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_p.to(devices[1]))
        # breakpoint()
        e_n = model.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_n.to(devices[0]))
        e_0.requires_grad = False
        e_p.requires_grad = False

        loss = criteria(e_n.to(devices[0]), e_0.to(devices[0]) - (negative_guidance*(e_p.to(devices[0]) - e_0.to(devices[0])))) #loss = criteria(e_n, e_0) works the best try 5000 epochs
        loss.backward()
        losses.append(loss.item())
        pbar.set_postfix({"loss": loss.item()})
        history.append(loss.item())
        opt.step()

        if (i+1) % 500 == 0 and i+1 != iterations and i+1>= 500:
            save_model(model, name, i-1, save_compvis=True, save_diffusers=False)

        if i % 100 == 0:
            save_history(losses, name, word_print)

    model.eval()

    save_model(model, name, None, save_compvis=True, save_diffusers=True, compvis_config_file=config_path, diffusers_config_file=diffusers_config_path)
    save_history(losses, name, word_print)

    return model

def save_model(model, name, num, compvis_config_file=None, diffusers_config_file=None, device='cpu', save_compvis=True, save_diffusers=True):
    # SAVE MODEL

#     PATH = f'{FOLDER}/{model_type}-word_{word_print}-method_{train_method}-sg_{start_guidance}-ng_{neg_guidance}-iter_{i+1}-lr_{lr}-startmodel_{start_model}-numacc_{numacc}.pt'

    folder_path = f'models/{name}'
    os.makedirs(folder_path, exist_ok=True)
    if num is not None:
        path = f'{folder_path}/{name}-epoch_{num}.pt'
    else:
        path = f'{folder_path}/{name}.pt'
    if save_compvis:
        torch.save(model.state_dict(), path)

    if save_diffusers:
        print('Saving Model in Diffusers Format')
        savemodelDiffusers(name, compvis_config_file, diffusers_config_file, device=device )

def save_history(losses, name, word_print):
    folder_path = f'models/{name}'
    os.makedirs(folder_path, exist_ok=True)
    with open(f'{folder_path}/loss.txt', 'w') as f:
        f.writelines([str(i) for i in losses])
    plot_loss(losses,f'{folder_path}/loss.png' , word_print, n=3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'TrainESD',
                    description = 'Finetuning stable diffusion model to erase concepts using ESD method')
    parser.add_argument('--prompt', help='prompt corresponding to concept to erase', type=str, required=True)
    parser.add_argument('--train_method', help='method of training', type=str, required=True)
    parser.add_argument('--start_guidance', help='guidance of start image used to train', type=float, required=False, default=3)
    parser.add_argument('--negative_guidance', help='guidance of negative training used to train', type=float, required=False, default=1)
    parser.add_argument('--iterations', help='iterations used to train', type=int, required=False, default=1000)
    parser.add_argument('--lr', help='learning rate used to train', type=int, required=False, default=1e-5)
    parser.add_argument('--config_path', help='config path for stable diffusion v1-4 inference', type=str, required=False, default='/share/u/rohit/stable-diffusion/configs/stable-diffusion/v1-inference.yaml')
    parser.add_argument('--ckpt_path', help='ckpt path for stable diffusion v1-4', type=str, required=False, default='/share/u/rohit/stable-diffusion/models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt')
    parser.add_argument('--diffusers_config_path', help='diffusers unet config json path', type=str, required=False, default='/share/u/rohit/stable-diffusion/diffusers_unet_config.json')
    parser.add_argument('--devices', help='cuda devices to train on', type=str, required=False, default='0,0')
    parser.add_argument('--seperator', help='separator if you want to train bunch of words separately', type=str, required=False, default=None)
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=50)
    args = parser.parse_args()
    
    prompt = args.prompt
    train_method = args.train_method
    start_guidance = args.start_guidance
    negative_guidance = args.negative_guidance
    iterations = args.iterations
    lr = args.lr
    config_path = args.config_path
    ckpt_path = args.ckpt_path
    diffusers_config_path = args.diffusers_config_path
    devices = [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]
    seperator = args.seperator
    image_size = args.image_size
    ddim_steps = args.ddim_steps

    train_esd(prompt=prompt, train_method=train_method, start_guidance=start_guidance, negative_guidance=negative_guidance, iterations=iterations, lr=lr, config_path=config_path, ckpt_path=ckpt_path, diffusers_config_path=diffusers_config_path, devices=devices, seperator=seperator, image_size=image_size, ddim_steps=ddim_steps, ddim_eta=0)
