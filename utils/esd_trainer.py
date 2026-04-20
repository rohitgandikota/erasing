from __future__ import annotations

import gc
import json
import os
import random
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from tqdm.auto import tqdm

from utils.esd_checkpoint import save_esd_checkpoint
from utils.sd_utils import esd_sd_call
from utils.sdxl_utils import esd_sdxl_call

# FLUX imports are lazy (inside adapter methods) — older diffusers builds used
# for SD/SDXL/SPACE don't have FluxIPAdapterMixin and would fail at import time.


TARGET_MODULE_TYPES = {
    "Linear",
    "Conv2d",
    "LoRACompatibleLinear",
    "LoRACompatibleConv",
}


def flux_latent_patch_grid_hw(
    height_px: int, width_px: int, vae_scale_factor: int
) -> tuple[int, int]:
    """Height/width of the Flux patch grid for packed latents.

    Matches ``FluxPipeline.prepare_latents`` and ``_prepare_latent_image_ids(..., h//2, w//2)``.
    Same geometry as HuggingFace ``examples/dreambooth/train_dreambooth_flux.py``, which uses
    ``model_input.shape[2] // 2`` and ``model_input.shape[3] // 2`` on unpacked VAE latents.
    """
    latent_h = 2 * (int(height_px) // (vae_scale_factor * 2))
    latent_w = 2 * (int(width_px) // (vae_scale_factor * 2))
    return latent_h // 2, latent_w // 2


@dataclass
class ESDConfig:
    family: str
    base_model_id: str
    erase_concept: str
    erase_from: Optional[str]
    train_method: str
    iterations: int
    lr: Optional[float]
    negative_guidance: float
    num_inference_steps: int
    guidance_scale: float
    batch_size: int
    resolution: Optional[int]
    save_path: str
    device: str = "cuda:0"
    torch_dtype: torch.dtype = torch.bfloat16
    inference_guidance_scale: Optional[float] = None
    max_sequence_length: int = 77
    gradient_checkpointing: bool = False
    allow_tf32: bool = False
    gradient_clip_norm: Optional[float] = None
    protect_concept: Optional[str] = None
    protect_concepts_k: Optional[List[str]] = None  # K>1 protection concepts for GS
    space_pairs_path: Optional[str] = None
    pres_lambda: float = 0.0   # weight for preservation loss (0 disables it; set >0 only with stylistically distant protection concepts)
    n_latents_avg: int = 4     # latents averaged for stable d_style estimate

    @property
    def erase_from_effective(self) -> str:
        return self.erase_from if self.erase_from is not None else self.erase_concept


@dataclass
class StepResult:
    model_pred: torch.Tensor
    target: torch.Tensor
    timestep_index: int
    metrics: Dict[str, Any] = field(default_factory=dict)


class PreparedComponent:
    def __init__(
        self,
        component: torch.nn.Module,
        student_params: "OrderedDict[str, torch.nn.Parameter]",
        base_params: "OrderedDict[str, torch.nn.Parameter]",
    ) -> None:
        self.component = component
        self.student_params = student_params
        self.base_params = base_params

    def use_base(self) -> None:
        for name, param in self.base_params.items():
            set_module(self.component, name, param)

    def use_student(self) -> None:
        for name, param in self.student_params.items():
            set_module(self.component, name, param)

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        return self.student_params.values()

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            name: param.detach().cpu().contiguous()
            for name, param in self.student_params.items()
        }


def set_module(module: torch.nn.Module, module_name, new_module) -> None:
    if isinstance(module_name, str):
        module_name = module_name.split(".")

    if len(module_name) == 1:
        setattr(module, module_name[0], new_module)
        return

    child_module = getattr(module, module_name[0])
    set_module(child_module, module_name[1:], new_module)


def resolve_default_resolution(pipe, fallback_component: Optional[str] = None) -> int:
    default_sample_size = getattr(pipe, "default_sample_size", None)
    if default_sample_size is None and fallback_component is not None:
        component = getattr(pipe, fallback_component)
        default_sample_size = component.config.sample_size

    if isinstance(default_sample_size, (tuple, list)):
        default_sample_size = default_sample_size[0]

    return int(default_sample_size) * pipe.vae_scale_factor


def select_parameter_names(
    component: torch.nn.Module,
    module_selector,
) -> list[str]:
    selected_names = []
    seen = set()
    for module_name, module in component.named_modules():
        if module.__class__.__name__ not in TARGET_MODULE_TYPES:
            continue
        if not module_selector(module_name):
            continue

        for param_name, _ in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            if full_name in seen:
                continue
            seen.add(full_name)
            selected_names.append(full_name)

    return selected_names


def prepare_component(
    component: torch.nn.Module,
    parameter_names: list[str],
    trainable_dtype: Optional[torch.dtype] = None,
) -> PreparedComponent:
    if not parameter_names:
        raise ValueError("No trainable parameters were selected for this configuration.")

    named_params = dict(component.named_parameters())
    component.requires_grad_(False)

    student_params: "OrderedDict[str, torch.nn.Parameter]" = OrderedDict()
    base_params: "OrderedDict[str, torch.nn.Parameter]" = OrderedDict()
    for parameter_name in parameter_names:
        if parameter_name not in named_params:
            raise KeyError(f"Parameter '{parameter_name}' was not found on the target component.")

        param = named_params[parameter_name]
        if trainable_dtype is not None and param.dtype != trainable_dtype:
            student_param = torch.nn.Parameter(
                param.detach().to(dtype=trainable_dtype).clone(),
                requires_grad=True,
            )
        else:
            param.requires_grad_(True)
            student_param = param

        student_params[parameter_name] = student_param
        base_params[parameter_name] = torch.nn.Parameter(param.detach().clone(), requires_grad=False)

    return PreparedComponent(component, student_params, base_params)


def sanitize_checkpoint_name(text: str) -> str:
    return text.replace(" ", "_")


def clear_device_cache(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def _suppress_transformers_pipeline_load_noise() -> Iterator[None]:
    """Hide benign Transformers 5.x chatter during diffusers `from_pretrained` (SD/SDXL/…).

    The "… LOAD REPORT" tables are emitted at WARNING on the **`transformers.modeling_utils`**
    logger (see `log_state_dict_report(..., logger=logger)` in Transformers), not on
    `transformers.utils.loading_report`. Lowering the whole library verbosity for this
    block reliably suppresses those messages plus legacy CLIP config warnings.
    """
    try:
        from transformers import logging as transformers_logging
    except ImportError:
        yield
        return
    previous = transformers_logging.get_verbosity()
    transformers_logging.set_verbosity_error()
    try:
        yield
    finally:
        transformers_logging.set_verbosity(previous)


def offload_modules_to_cpu(device: str, *modules: Optional[torch.nn.Module]) -> None:
    for module in modules:
        if module is not None:
            module.to("cpu")
    clear_device_cache(device)
    gc.collect()


def make_sampling_generator(device: str, seed: int) -> torch.Generator:
    target_device = torch.device(device)
    if target_device.type == "cuda" and torch.cuda.is_available():
        return torch.Generator(device=target_device).manual_seed(seed)
    return torch.Generator().manual_seed(seed)


class BaseESDAdapter:
    family = ""
    component_attr = ""
    default_base_model_id = ""
    default_save_path = ""

    def normalize_train_method(self, train_method: str) -> str:
        raise NotImplementedError

    def default_lr_for_method(self, train_method: str) -> float:
        raise NotImplementedError

    def load_pipeline(self, config: ESDConfig):
        raise NotImplementedError

    def trainable_param_dtype(self, config: ESDConfig) -> Optional[torch.dtype]:
        return None

    def select_parameter_names(self, component: torch.nn.Module, train_method: str) -> list[str]:
        raise NotImplementedError

    def prepare_context(self, pipe, config: ESDConfig) -> Dict[str, Any]:
        raise NotImplementedError

    def training_step(
        self,
        pipe,
        prepared: PreparedComponent,
        context: Dict[str, Any],
        config: ESDConfig,
    ) -> StepResult:
        raise NotImplementedError

    def resolve_resolution(self, pipe, config: ESDConfig) -> int:
        if config.resolution is not None:
            return config.resolution
        return resolve_default_resolution(pipe, fallback_component=self.component_attr)

    def resolve_learning_rate(self, config: ESDConfig) -> float:
        if config.lr is not None:
            return config.lr
        return self.default_lr_for_method(config.train_method)

    def build_metadata(self, config: ESDConfig) -> Dict[str, str]:
        metadata = {
            "family": self.family,
            "component": self.component_attr,
            "base_model_id": config.base_model_id,
            "train_method": config.train_method,
            "erase_concept": config.erase_concept,
            "erase_from": config.erase_from or "",
            "num_inference_steps": str(config.num_inference_steps),
            "guidance_scale": str(config.guidance_scale),
            "negative_guidance": str(config.negative_guidance),
            "batch_size": str(config.batch_size),
        }
        if config.resolution is not None:
            metadata["resolution"] = str(config.resolution)
        return metadata

    def build_checkpoint_path(self, config: ESDConfig) -> str:
        method_suffix = config.train_method.replace("-", "")
        filename = (
            f"esd-{sanitize_checkpoint_name(config.erase_concept)}"
            f"-from-{sanitize_checkpoint_name(config.erase_from_effective)}"
            f"-{method_suffix}.safetensors"
        )
        return os.path.join(config.save_path, filename)

    def create_prepared_component(self, pipe, train_method: str, config: ESDConfig) -> PreparedComponent:
        component = getattr(pipe, self.component_attr)
        parameter_names = self.select_parameter_names(component, train_method)
        return prepare_component(component, parameter_names, trainable_dtype=self.trainable_param_dtype(config))


class StableDiffusionESDAdapter(BaseESDAdapter):
    family = "sd"
    component_attr = "unet"
    default_base_model_id = "CompVis/stable-diffusion-v1-4"
    default_save_path = "esd-models/sd/"

    def normalize_train_method(self, train_method: str) -> str:
        aliases = {
            "xattn": "esd-x",
            "noxattn": "esd-u",
            "full": "esd-all",
            "xattn-strict": "esd-x-strict",
            "selfattn": "selfattn",
            "esd-x": "esd-x",
            "esd-u": "esd-u",
            "esd-all": "esd-all",
            "esd-x-strict": "esd-x-strict",
        }
        normalized = aliases.get(train_method)
        if normalized is None:
            raise ValueError(f"Unsupported SD train method: {train_method}")
        return normalized

    def default_lr_for_method(self, train_method: str) -> float:
        return 5e-5

    def load_pipeline(self, config: ESDConfig):
        pipe = StableDiffusionPipeline.from_pretrained(
            config.base_model_id,
            torch_dtype=config.torch_dtype,
            use_safetensors=True,
        ).to(config.device)
        pipe.vae.requires_grad_(False)
        pipe.text_encoder.requires_grad_(False)
        if pipe.safety_checker is not None:
            pipe.safety_checker.requires_grad_(False)
        return pipe

    def select_parameter_names(self, component: torch.nn.Module, train_method: str) -> list[str]:
        def selector(module_name: str) -> bool:
            if train_method == "esd-x":
                return "attn2" in module_name
            if train_method == "esd-u":
                return "attn2" not in module_name
            if train_method == "esd-all":
                return True
            if train_method == "esd-x-strict":
                return "attn2.to_k" in module_name or "attn2.to_v" in module_name
            if train_method == "selfattn":
                return "attn1" in module_name
            return False

        return select_parameter_names(component, selector)

    def prepare_context(self, pipe, config: ESDConfig) -> Dict[str, Any]:
        resolution = self.resolve_resolution(pipe, config)
        with torch.no_grad():
            erase_embeds, null_embeds = pipe.encode_prompt(
                prompt=config.erase_concept,
                device=config.device,
                num_images_per_prompt=config.batch_size,
                do_classifier_free_guidance=True,
                negative_prompt="",
            )
            erase_embeds = erase_embeds.to(config.device)
            null_embeds = null_embeds.to(config.device)

            erase_from_embeds = None
            if config.erase_from is not None:
                erase_from_embeds, _ = pipe.encode_prompt(
                    prompt=config.erase_from,
                    device=config.device,
                    num_images_per_prompt=config.batch_size,
                    do_classifier_free_guidance=False,
                    negative_prompt="",
                )
                erase_from_embeds = erase_from_embeds.to(config.device)

            timestep_cond = None
            if pipe.unet.config.time_cond_proj_dim is not None:
                guidance_scale_tensor = torch.tensor(config.guidance_scale - 1).repeat(config.batch_size)
                timestep_cond = pipe.get_guidance_scale_embedding(
                    guidance_scale_tensor,
                    embedding_dim=pipe.unet.config.time_cond_proj_dim,
                ).to(device=config.device, dtype=config.torch_dtype)

        offload_modules_to_cpu(config.device, pipe.vae, pipe.text_encoder, pipe.safety_checker)

        return {
            "resolution": resolution,
            "erase_embeds": erase_embeds,
            "null_embeds": null_embeds,
            "erase_from_embeds": erase_from_embeds,
            "sample_prompt_embeds": erase_embeds if erase_from_embeds is None else erase_from_embeds,
            "sample_negative_prompt_embeds": null_embeds,
            "student_prompt_embeds": erase_embeds if erase_from_embeds is None else erase_from_embeds,
            "timestep_cond": timestep_cond,
        }

    def training_step(self, pipe, prepared: PreparedComponent, context: Dict[str, Any], config: ESDConfig) -> StepResult:
        run_till_timestep = random.randint(0, config.num_inference_steps - 1)
        seed = random.randint(0, 2**15)

        prepared.use_base()
        prepared.component.eval()
        with torch.no_grad():
            xt = esd_sd_call(
                pipe,
                prompt_embeds=context["sample_prompt_embeds"],
                negative_prompt_embeds=context["sample_negative_prompt_embeds"],
                num_images_per_prompt=1,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                run_till_timestep=run_till_timestep,
                generator=make_sampling_generator(config.device, seed),
                output_type="latent",
                height=context["resolution"],
                width=context["resolution"],
            ).images

            timestep = pipe.scheduler.timesteps[run_till_timestep]
            noise_pred_erase = prepared.component(
                xt,
                timestep,
                encoder_hidden_states=context["erase_embeds"],
                timestep_cond=context["timestep_cond"],
                cross_attention_kwargs=None,
                added_cond_kwargs=None,
                return_dict=False,
            )[0]
            noise_pred_null = prepared.component(
                xt,
                timestep,
                encoder_hidden_states=context["null_embeds"],
                timestep_cond=context["timestep_cond"],
                cross_attention_kwargs=None,
                added_cond_kwargs=None,
                return_dict=False,
            )[0]

            if context["erase_from_embeds"] is not None:
                noise_pred_erase_from = prepared.component(
                    xt,
                    timestep,
                    encoder_hidden_states=context["erase_from_embeds"],
                    timestep_cond=context["timestep_cond"],
                    cross_attention_kwargs=None,
                    added_cond_kwargs=None,
                    return_dict=False,
                )[0]
            else:
                noise_pred_erase_from = noise_pred_erase

        prepared.use_student()
        prepared.component.train()
        model_pred = prepared.component(
            xt,
            timestep,
            encoder_hidden_states=context["student_prompt_embeds"],
            timestep_cond=context["timestep_cond"],
            cross_attention_kwargs=None,
            added_cond_kwargs=None,
            return_dict=False,
        )[0]

        target = noise_pred_erase_from - config.negative_guidance * (noise_pred_erase - noise_pred_null)
        return StepResult(model_pred=model_pred, target=target, timestep_index=run_till_timestep)


class StableDiffusionXLESDAdapter(BaseESDAdapter):
    family = "sdxl"
    component_attr = "unet"
    default_base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    default_save_path = "esd-models/sdxl/"

    def normalize_train_method(self, train_method: str) -> str:
        aliases = {
            "xattn": "esd-x",
            "noxattn": "esd-u",
            "full": "esd-all",
            "xattn-strict": "esd-x-strict",
            "esd-x": "esd-x",
            "esd-u": "esd-u",
            "esd-all": "esd-all",
            "esd-x-strict": "esd-x-strict",
        }
        normalized = aliases.get(train_method)
        if normalized is None:
            raise ValueError(f"Unsupported SDXL train method: {train_method}")
        if normalized in {"esd-u", "esd-all"}:
            warnings.warn(
                "SDXL `esd-u` and `esd-all` update a very large fraction of the UNet and can strongly degrade "
                "overall model quality. Prefer `esd-x` or `esd-x-strict` unless you intentionally want global erasure.",
                stacklevel=2,
            )
        return normalized

    def default_lr_for_method(self, train_method: str) -> float:
        if train_method.startswith("esd-x"):
            return 2e-4
        return 1e-5

    def load_pipeline(self, config: ESDConfig):
        pipe = StableDiffusionXLPipeline.from_pretrained(
            config.base_model_id,
            torch_dtype=config.torch_dtype,
            use_safetensors=True,
        ).to(config.device)
        pipe.vae.requires_grad_(False)
        if pipe.text_encoder is not None:
            pipe.text_encoder.requires_grad_(False)
        if pipe.text_encoder_2 is not None:
            pipe.text_encoder_2.requires_grad_(False)
        return pipe

    def select_parameter_names(self, component: torch.nn.Module, train_method: str) -> list[str]:
        def selector(module_name: str) -> bool:
            if train_method == "esd-x":
                return "attn2" in module_name
            if train_method == "esd-u":
                return "attn2" not in module_name and "emb" not in module_name and "block" in module_name
            if train_method == "esd-all":
                return "emb" not in module_name and "block" in module_name
            if train_method == "esd-x-strict":
                return "attn2.to_k" in module_name or "attn2.to_v" in module_name
            return False

        return select_parameter_names(component, selector)

    def prepare_context(self, pipe, config: ESDConfig) -> Dict[str, Any]:
        resolution = self.resolve_resolution(pipe, config)
        with torch.no_grad():
            erase_embeds, null_embeds, erase_pooled_embeds, null_pooled_embeds = pipe.encode_prompt(
                prompt=config.erase_concept,
                device=config.device,
                num_images_per_prompt=config.batch_size,
                do_classifier_free_guidance=True,
                negative_prompt="",
            )
            erase_embeds = erase_embeds.to(config.device)
            null_embeds = null_embeds.to(config.device)
            erase_pooled_embeds = erase_pooled_embeds.to(config.device)
            null_pooled_embeds = null_pooled_embeds.to(config.device)

            if pipe.text_encoder_2 is None:
                text_encoder_projection_dim = int(erase_pooled_embeds.shape[-1])
            else:
                text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim

            add_time_ids = pipe._get_add_time_ids(
                (resolution, resolution),
                (0, 0),
                (resolution, resolution),
                dtype=erase_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            ).to(config.device)
            add_time_ids = add_time_ids.repeat(config.batch_size, 1)

            erase_from_embeds = None
            erase_from_pooled_embeds = None
            if config.erase_from is not None:
                erase_from_embeds, _, erase_from_pooled_embeds, _ = pipe.encode_prompt(
                    prompt=config.erase_from,
                    device=config.device,
                    num_images_per_prompt=config.batch_size,
                    do_classifier_free_guidance=False,
                    negative_prompt="",
                )
                erase_from_embeds = erase_from_embeds.to(config.device)
                erase_from_pooled_embeds = erase_from_pooled_embeds.to(config.device)

            timestep_cond = None
            if pipe.unet.config.time_cond_proj_dim is not None:
                guidance_scale_tensor = torch.tensor(config.guidance_scale - 1).repeat(config.batch_size)
                timestep_cond = pipe.get_guidance_scale_embedding(
                    guidance_scale_tensor,
                    embedding_dim=pipe.unet.config.time_cond_proj_dim,
                ).to(device=config.device, dtype=config.torch_dtype)

        offload_modules_to_cpu(config.device, pipe.vae, pipe.text_encoder, pipe.text_encoder_2)

        return {
            "resolution": resolution,
            "erase_embeds": erase_embeds,
            "null_embeds": null_embeds,
            "erase_pooled_embeds": erase_pooled_embeds,
            "null_pooled_embeds": null_pooled_embeds,
            "erase_from_embeds": erase_from_embeds,
            "erase_from_pooled_embeds": erase_from_pooled_embeds,
            "sample_prompt_embeds": erase_embeds if erase_from_embeds is None else erase_from_embeds,
            "sample_negative_prompt_embeds": null_embeds,
            "sample_pooled_prompt_embeds": erase_pooled_embeds if erase_from_pooled_embeds is None else erase_from_pooled_embeds,
            "sample_negative_pooled_prompt_embeds": null_pooled_embeds,
            "student_prompt_embeds": erase_embeds if erase_from_embeds is None else erase_from_embeds,
            "student_pooled_prompt_embeds": erase_pooled_embeds if erase_from_pooled_embeds is None else erase_from_pooled_embeds,
            "add_time_ids": add_time_ids,
            "timestep_cond": timestep_cond,
        }

    def training_step(self, pipe, prepared: PreparedComponent, context: Dict[str, Any], config: ESDConfig) -> StepResult:
        run_till_timestep = random.randint(0, config.num_inference_steps - 1)
        seed = random.randint(0, 2**15)

        prepared.use_base()
        prepared.component.eval()
        with torch.no_grad():
            xt = esd_sdxl_call(
                pipe,
                prompt_embeds=context["sample_prompt_embeds"],
                negative_prompt_embeds=context["sample_negative_prompt_embeds"],
                pooled_prompt_embeds=context["sample_pooled_prompt_embeds"],
                negative_pooled_prompt_embeds=context["sample_negative_pooled_prompt_embeds"],
                num_images_per_prompt=1,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                run_till_timestep=run_till_timestep,
                generator=make_sampling_generator(config.device, seed),
                output_type="latent",
                height=context["resolution"],
                width=context["resolution"],
            ).images

            timestep = pipe.scheduler.timesteps[run_till_timestep]
            erase_kwargs = {"text_embeds": context["erase_pooled_embeds"], "time_ids": context["add_time_ids"]}
            noise_pred_erase = prepared.component(
                xt,
                timestep,
                encoder_hidden_states=context["erase_embeds"],
                timestep_cond=context["timestep_cond"],
                cross_attention_kwargs=None,
                added_cond_kwargs=erase_kwargs,
                return_dict=False,
            )[0]

            null_kwargs = {"text_embeds": context["null_pooled_embeds"], "time_ids": context["add_time_ids"]}
            noise_pred_null = prepared.component(
                xt,
                timestep,
                encoder_hidden_states=context["null_embeds"],
                timestep_cond=context["timestep_cond"],
                cross_attention_kwargs=None,
                added_cond_kwargs=null_kwargs,
                return_dict=False,
            )[0]

            if context["erase_from_embeds"] is not None:
                erase_from_kwargs = {
                    "text_embeds": context["erase_from_pooled_embeds"],
                    "time_ids": context["add_time_ids"],
                }
                noise_pred_erase_from = prepared.component(
                    xt,
                    timestep,
                    encoder_hidden_states=context["erase_from_embeds"],
                    timestep_cond=context["timestep_cond"],
                    cross_attention_kwargs=None,
                    added_cond_kwargs=erase_from_kwargs,
                    return_dict=False,
                )[0]
            else:
                noise_pred_erase_from = noise_pred_erase

        prepared.use_student()
        prepared.component.train()
        student_kwargs = {
            "text_embeds": context["student_pooled_prompt_embeds"],
            "time_ids": context["add_time_ids"],
        }
        model_pred = prepared.component(
            xt,
            timestep,
            encoder_hidden_states=context["student_prompt_embeds"],
            timestep_cond=context["timestep_cond"],
            cross_attention_kwargs=None,
            added_cond_kwargs=student_kwargs,
            return_dict=False,
        )[0]

        target = noise_pred_erase_from - config.negative_guidance * (noise_pred_erase - noise_pred_null)
        return StepResult(model_pred=model_pred, target=target, timestep_index=run_till_timestep)


class FluxESDAdapter(BaseESDAdapter):
    family = "flux"
    component_attr = "transformer"
    default_base_model_id = "black-forest-labs/FLUX.1-dev"
    default_save_path = "esd-models/flux/"

    def normalize_train_method(self, train_method: str) -> str:
        aliases = {
            "xattn": "esd-x",
            "xattn-strict": "esd-x-strict",
            "esd-x": "esd-x",
            "esd-x-strict": "esd-x-strict",
        }
        normalized = aliases.get(train_method)
        if normalized is None:
            raise ValueError(f"Unsupported FLUX train method: {train_method}")
        return normalized

    def default_lr_for_method(self, train_method: str) -> float:
        return 1e-4

    def load_pipeline(self, config: ESDConfig):
        from diffusers import FluxPipeline  # lazy: not available in older diffusers builds
        # No VAE for ESD (unlike DreamBooth): training latents come only from the transformer
        # sampling loop (`esd_flux_call` with `output_type="latent"`), never from pixel encode/decode.
        # `vae=None` skips loading the VAE; `FluxPipeline` still sets `vae_scale_factor` (default 8) for geometry.
        pipe = FluxPipeline.from_pretrained(
            config.base_model_id,
            vae=None,
            torch_dtype=config.torch_dtype,
            use_safetensors=True,
        ).to(config.device)
        if pipe.text_encoder is not None:
            pipe.text_encoder.requires_grad_(False)
        if getattr(pipe, "text_encoder_2", None) is not None:
            pipe.text_encoder_2.requires_grad_(False)
        return pipe

    def build_metadata(self, config: ESDConfig) -> Dict[str, str]:
        metadata = super().build_metadata(config)
        metadata["max_sequence_length"] = str(config.max_sequence_length)
        if config.inference_guidance_scale is not None:
            metadata["inference_guidance_scale"] = str(config.inference_guidance_scale)
        return metadata

    def select_parameter_names(self, component: torch.nn.Module, train_method: str) -> list[str]:
        def selector(module_name: str) -> bool:
            if train_method == "esd-x":
                return "attn" in module_name
            if train_method == "esd-x-strict":
                return "attn" in module_name and ("to_k" in module_name or "to_v" in module_name)
            return False

        return select_parameter_names(component, selector)

    def prepare_context(self, pipe, config: ESDConfig) -> Dict[str, Any]:
        resolution = self.resolve_resolution(pipe, config)
        prompts = [config.erase_concept]
        if config.erase_from is not None:
            prompts.append(config.erase_from)
        prompts.append("")
        with torch.no_grad():
            prompt_embeds_all, pooled_prompt_embeds_all, text_ids = pipe.encode_prompt(
                prompts,
                prompt_2=prompts,
                num_images_per_prompt=config.batch_size,
                max_sequence_length=config.max_sequence_length,
            )

            if config.erase_from is None:
                erase_prompt_embeds, null_prompt_embeds = prompt_embeds_all.chunk(2)
                erase_pooled_prompt_embeds, null_pooled_prompt_embeds = pooled_prompt_embeds_all.chunk(2)
                erase_from_prompt_embeds = None
                erase_from_pooled_prompt_embeds = None
            else:
                erase_prompt_embeds, erase_from_prompt_embeds, null_prompt_embeds = prompt_embeds_all.chunk(3)
                erase_pooled_prompt_embeds, erase_from_pooled_prompt_embeds, null_pooled_prompt_embeds = (
                    pooled_prompt_embeds_all.chunk(3)
                )

        # VAE is not loaded (`vae=None`); only move text encoders off the train device after caching embeds.
        offload_modules_to_cpu(config.device, pipe.text_encoder, getattr(pipe, "text_encoder_2", None))

        erase_prompt_embeds = erase_prompt_embeds.to(config.device)
        null_prompt_embeds = null_prompt_embeds.to(config.device)
        erase_pooled_prompt_embeds = erase_pooled_prompt_embeds.to(config.device)
        null_pooled_prompt_embeds = null_pooled_prompt_embeds.to(config.device)
        if erase_from_prompt_embeds is not None:
            erase_from_prompt_embeds = erase_from_prompt_embeds.to(config.device)
        if erase_from_pooled_prompt_embeds is not None:
            erase_from_pooled_prompt_embeds = erase_from_pooled_prompt_embeds.to(config.device)

        return {
            "resolution": resolution,
            "erase_prompt_embeds": erase_prompt_embeds,
            "erase_from_prompt_embeds": erase_from_prompt_embeds,
            "null_prompt_embeds": null_prompt_embeds,
            "erase_pooled_prompt_embeds": erase_pooled_prompt_embeds,
            "erase_from_pooled_prompt_embeds": erase_from_pooled_prompt_embeds,
            "null_pooled_prompt_embeds": null_pooled_prompt_embeds,
            # Flux `text_ids` is (seq_len, 3), same for all batch rows — do not slice dim 0 by batch_size.
            "text_ids": text_ids.to(config.device),
            "sample_prompt_embeds": erase_prompt_embeds if config.erase_from is None else erase_from_prompt_embeds,
            "sample_pooled_prompt_embeds": erase_pooled_prompt_embeds if config.erase_from is None else erase_from_pooled_prompt_embeds,
            "student_prompt_embeds": erase_prompt_embeds if config.erase_from is None else erase_from_prompt_embeds,
            "student_pooled_prompt_embeds": erase_pooled_prompt_embeds if config.erase_from is None else erase_from_pooled_prompt_embeds,
        }

    def get_training_timesteps(self, pipe, num_inference_steps: int, image_seq_len: int, device: str):
        from diffusers.pipelines.flux.pipeline_flux import calculate_shift
        from diffusers.pipelines.flux.pipeline_flux import retrieve_timesteps as retrieve_flux_timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        if getattr(pipe.scheduler.config, "use_flow_sigmas", False):
            sigmas = None

        mu = calculate_shift(
            image_seq_len,
            pipe.scheduler.config.get("base_image_seq_len", 256),
            pipe.scheduler.config.get("max_image_seq_len", 4096),
            pipe.scheduler.config.get("base_shift", 0.5),
            pipe.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, _ = retrieve_flux_timesteps(
            pipe.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        return timesteps

    def training_step(self, pipe, prepared: PreparedComponent, context: Dict[str, Any], config: ESDConfig) -> StepResult:
        from diffusers import FluxPipeline  # lazy
        from utils.flux_utils import esd_flux_call  # lazy
        run_till_timestep = random.randint(0, config.num_inference_steps - 1)
        seed = random.randint(0, 2**15)

        prepared.use_base()
        prepared.component.eval()
        with torch.no_grad():
            xt = esd_flux_call(
                pipe,
                prompt_embeds=context["sample_prompt_embeds"],
                pooled_prompt_embeds=context["sample_pooled_prompt_embeds"],
                num_images_per_prompt=1,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.inference_guidance_scale or config.guidance_scale,
                max_sequence_length=config.max_sequence_length,
                run_till_timestep=run_till_timestep,
                generator=make_sampling_generator(config.device, seed),
                output_type="latent",
                height=context["resolution"],
                width=context["resolution"],
            ).images

            timesteps = self.get_training_timesteps(
                pipe,
                num_inference_steps=config.num_inference_steps,
                image_seq_len=xt.shape[1],
                device=config.device,
            )
            timestep = timesteps[run_till_timestep].unsqueeze(0).to(config.device)
            guidance = None
            if pipe.transformer.config.guidance_embeds:
                guidance = torch.full(
                    [xt.shape[0]],
                    config.guidance_scale,
                    device=config.device,
                    dtype=torch.float32,
                )

            # Patch grid must match packed latents from `prepare_latents` / DreamBooth-style training.
            h_px = w_px = int(context["resolution"])
            patch_h, patch_w = flux_latent_patch_grid_hw(h_px, w_px, pipe.vae_scale_factor)
            expected_seq = patch_h * patch_w
            if xt.shape[1] != expected_seq:
                raise ValueError(
                    f"FLUX packed latent length {xt.shape[1]} != patch grid {patch_h}×{patch_w}={expected_seq}. "
                    "Sampling and transformer `img_ids` must use the same geometry as FluxPipeline.prepare_latents."
                )
            latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                xt.shape[0],
                patch_h,
                patch_w,
                config.device,
                config.torch_dtype,
            )

            noise_pred_null = prepared.component(
                hidden_states=xt,
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=context["null_pooled_prompt_embeds"],
                encoder_hidden_states=context["null_prompt_embeds"],
                txt_ids=context["text_ids"],
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]
            noise_pred_erase = prepared.component(
                hidden_states=xt,
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=context["erase_pooled_prompt_embeds"],
                encoder_hidden_states=context["erase_prompt_embeds"],
                txt_ids=context["text_ids"],
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]
            if context["erase_from_prompt_embeds"] is not None:
                noise_pred_from = prepared.component(
                    hidden_states=xt,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=context["erase_from_pooled_prompt_embeds"],
                    encoder_hidden_states=context["erase_from_prompt_embeds"],
                    txt_ids=context["text_ids"],
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
            else:
                noise_pred_from = noise_pred_erase

        prepared.use_student()
        prepared.component.train()
        model_pred = prepared.component(
            hidden_states=xt,
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=context["student_pooled_prompt_embeds"],
            encoder_hidden_states=context["student_prompt_embeds"],
            txt_ids=context["text_ids"],
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]

        target = noise_pred_from - config.negative_guidance * (noise_pred_erase - noise_pred_null)
        return StepResult(model_pred=model_pred, target=target, timestep_index=run_till_timestep)


class Flux2KleinESDAdapter(BaseESDAdapter):
    family = "flux2_klein"
    component_attr = "transformer"
    default_base_model_id = "black-forest-labs/FLUX.2-klein-base-4B"
    default_save_path = "esd-models/flux2-klein/"

    def normalize_train_method(self, train_method: str) -> str:
        aliases = {
            "xattn": "esd-x",
            "xattn-strict": "esd-x-strict",
            "esd-x": "esd-x",
            "esd-x-strict": "esd-x-strict",
        }
        normalized = aliases.get(train_method)
        if normalized is None:
            raise ValueError(f"Unsupported FLUX.2 Klein train method: {train_method}")
        return normalized

    def default_lr_for_method(self, train_method: str) -> float:
        return 1e-4

    def load_pipeline(self, config: ESDConfig):
        try:
            from diffusers import Flux2KleinPipeline
        except ImportError as exc:
            raise ImportError(
                "FLUX.2 Klein support requires a newer diffusers/transformers install with `Flux2KleinPipeline`."
            ) from exc

        # Same as FLUX.1 ESD: no VAE — latents are produced in the Klein sampling path, not from pixels.
        pipe = Flux2KleinPipeline.from_pretrained(
            config.base_model_id,
            vae=None,
            torch_dtype=config.torch_dtype,
            use_safetensors=True,
        ).to(config.device)
        pipe.text_encoder.requires_grad_(False)
        return pipe

    def build_metadata(self, config: ESDConfig) -> Dict[str, str]:
        metadata = super().build_metadata(config)
        metadata["max_sequence_length"] = str(config.max_sequence_length)
        if config.inference_guidance_scale is not None:
            metadata["inference_guidance_scale"] = str(config.inference_guidance_scale)
        return metadata

    def select_parameter_names(self, component: torch.nn.Module, train_method: str) -> list[str]:
        attention_suffixes = (
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
        )
        strict_suffixes = ("to_k", "to_v", "add_k_proj", "add_v_proj")

        def selector(module_name: str) -> bool:
            if train_method == "esd-x":
                return module_name.endswith(attention_suffixes)
            if train_method == "esd-x-strict":
                return module_name.endswith(strict_suffixes)
            return False

        return select_parameter_names(component, selector)

    def prepare_context(self, pipe, config: ESDConfig) -> Dict[str, Any]:
        resolution = self.resolve_resolution(pipe, config)
        with torch.no_grad():
            erase_prompt_embeds, erase_text_ids = pipe.encode_prompt(
                prompt=config.erase_concept,
                device=config.device,
                num_images_per_prompt=config.batch_size,
                max_sequence_length=config.max_sequence_length,
            )
            erase_prompt_embeds = erase_prompt_embeds.to(config.device)
            erase_text_ids = erase_text_ids.to(config.device)

            null_prompt_embeds, null_text_ids = pipe.encode_prompt(
                prompt="",
                device=config.device,
                num_images_per_prompt=config.batch_size,
                max_sequence_length=config.max_sequence_length,
            )
            null_prompt_embeds = null_prompt_embeds.to(config.device)
            null_text_ids = null_text_ids.to(config.device)

            erase_from_prompt_embeds = None
            erase_from_text_ids = None
            if config.erase_from is not None:
                erase_from_prompt_embeds, erase_from_text_ids = pipe.encode_prompt(
                    prompt=config.erase_from,
                    device=config.device,
                    num_images_per_prompt=config.batch_size,
                    max_sequence_length=config.max_sequence_length,
                )
                erase_from_prompt_embeds = erase_from_prompt_embeds.to(config.device)
                erase_from_text_ids = erase_from_text_ids.to(config.device)

        offload_modules_to_cpu(config.device, pipe.text_encoder)

        return {
            "resolution": resolution,
            "erase_prompt_embeds": erase_prompt_embeds,
            "erase_from_prompt_embeds": erase_from_prompt_embeds,
            "null_prompt_embeds": null_prompt_embeds,
            "erase_text_ids": erase_text_ids,
            "erase_from_text_ids": erase_from_text_ids,
            "null_text_ids": null_text_ids,
            "sample_prompt_embeds": erase_prompt_embeds if erase_from_prompt_embeds is None else erase_from_prompt_embeds,
            "sample_text_ids": erase_text_ids if erase_from_text_ids is None else erase_from_text_ids,
            "student_prompt_embeds": erase_prompt_embeds if erase_from_prompt_embeds is None else erase_from_prompt_embeds,
            "student_text_ids": erase_text_ids if erase_from_text_ids is None else erase_from_text_ids,
        }

    def get_training_timesteps(self, pipe, num_inference_steps: int, image_seq_len: int, device: str):
        from utils.flux2_klein_utils import compute_empirical_mu, retrieve_flux2_klein_timesteps  # lazy
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        if getattr(pipe.scheduler.config, "use_flow_sigmas", False):
            sigmas = None

        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
        timesteps, _ = retrieve_flux2_klein_timesteps(
            pipe.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        return timesteps

    def training_step(self, pipe, prepared: PreparedComponent, context: Dict[str, Any], config: ESDConfig) -> StepResult:
        from utils.flux2_klein_utils import esd_flux2_klein_call  # lazy
        run_till_timestep = random.randint(0, config.num_inference_steps - 1)
        seed = random.randint(0, 2**15)

        prepared.use_base()
        prepared.component.eval()
        with torch.no_grad():
            sample_result = esd_flux2_klein_call(
                pipe,
                prompt_embeds=context["sample_prompt_embeds"],
                negative_prompt_embeds=context["null_prompt_embeds"],
                text_ids=context["sample_text_ids"],
                negative_text_ids=context["null_text_ids"],
                num_images_per_prompt=1,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.inference_guidance_scale or config.guidance_scale,
                max_sequence_length=config.max_sequence_length,
                run_till_timestep=run_till_timestep,
                generator=make_sampling_generator(config.device, seed),
                output_type="latent",
                height=context["resolution"],
                width=context["resolution"],
            )
            xt = sample_result.images
            latent_ids = sample_result.latent_ids

            timesteps = self.get_training_timesteps(
                pipe,
                num_inference_steps=config.num_inference_steps,
                image_seq_len=xt.shape[1],
                device=config.device,
            )
            timestep = timesteps[run_till_timestep].unsqueeze(0).to(config.device)
            guidance = None

            noise_pred_null = prepared.component(
                hidden_states=xt,
                timestep=timestep / 1000,
                guidance=guidance,
                encoder_hidden_states=context["null_prompt_embeds"],
                txt_ids=context["null_text_ids"],
                img_ids=latent_ids,
                return_dict=False,
            )[0]
            noise_pred_erase = prepared.component(
                hidden_states=xt,
                timestep=timestep / 1000,
                guidance=guidance,
                encoder_hidden_states=context["erase_prompt_embeds"],
                txt_ids=context["erase_text_ids"],
                img_ids=latent_ids,
                return_dict=False,
            )[0]
            if context["erase_from_prompt_embeds"] is not None:
                noise_pred_from = prepared.component(
                    hidden_states=xt,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states=context["erase_from_prompt_embeds"],
                    txt_ids=context["erase_from_text_ids"],
                    img_ids=latent_ids,
                    return_dict=False,
                )[0]
            else:
                noise_pred_from = noise_pred_erase

        prepared.use_student()
        prepared.component.train()
        model_pred = prepared.component(
            hidden_states=xt,
            timestep=timestep / 1000,
            guidance=guidance,
            encoder_hidden_states=context["student_prompt_embeds"],
            txt_ids=context["student_text_ids"],
            img_ids=latent_ids,
            return_dict=False,
        )[0]

        target = noise_pred_from - config.negative_guidance * (noise_pred_erase - noise_pred_null)
        return StepResult(model_pred=model_pred, target=target, timestep_index=run_till_timestep)


class SpaceSDAdapter(BaseESDAdapter):
    """SPACE: Semantically Precise Attribute Concept Erasure for SD v1.x.

    Key differences from ESD-x:
      - Uses 20 concept/anchor prompt pairs (anchor = semantic match minus the attribute).
      - Style direction d_style = eps(concept) - eps(anchor) isolates the attribute only.
      - K-GS projects d_style orthogonal to protection directions (use stylistically distant concepts, e.g. "a photograph").
      - Training latent sampled directly from N(0,I) via the forward diffusion equation
        (no partial denoising trajectory needed).
      - eta=5.0 is safe because d_style is a clean, low-noise direction.
    """

    family = "sd"
    component_attr = "unet"
    default_base_model_id = "CompVis/stable-diffusion-v1-4"
    default_save_path = "esd-models/space/"

    def normalize_train_method(self, train_method: str) -> str:
        # SPACE always uses esd-x-strict (only attn2.to_k and attn2.to_v)
        aliases = {
            "esd-x-strict": "esd-x-strict",
            "xattn-strict": "esd-x-strict",
        }
        normalized = aliases.get(train_method, "esd-x-strict")
        return normalized

    def default_lr_for_method(self, train_method: str) -> float:
        return 1e-5

    def load_pipeline(self, config: ESDConfig):
        pipe = StableDiffusionPipeline.from_pretrained(
            config.base_model_id,
            torch_dtype=config.torch_dtype,
            use_safetensors=True,
        ).to(config.device)
        pipe.vae.requires_grad_(False)
        pipe.text_encoder.requires_grad_(False)
        if pipe.safety_checker is not None:
            pipe.safety_checker.requires_grad_(False)
        return pipe

    def select_parameter_names(self, component: torch.nn.Module, train_method: str) -> list[str]:
        def selector(module_name: str) -> bool:
            return "attn2.to_k" in module_name or "attn2.to_v" in module_name
        return select_parameter_names(component, selector)

    def _load_pairs(self, config: ESDConfig) -> Dict[str, Any]:
        if config.space_pairs_path:
            pairs_path = Path(config.space_pairs_path)
        else:
            concept_slug = config.erase_concept.lower().replace(" ", "_")
            pairs_path = Path("data/space_pairs") / f"{concept_slug}.json"
        if not pairs_path.exists():
            raise FileNotFoundError(
                f"SPACE pairs file not found: {pairs_path}. "
                "Create it or pass --space_pairs_path explicitly."
            )
        with open(pairs_path) as f:
            return json.load(f)

    def prepare_context(self, pipe, config: ESDConfig) -> Dict[str, Any]:
        resolution = self.resolve_resolution(pipe, config)
        pairs_data = self._load_pairs(config)
        pairs = pairs_data["pairs"]
        protect_prompt = config.protect_concept or pairs_data.get("protect", "")
        if not protect_prompt:
            raise ValueError(
                "SPACE requires a protection concept. Add a 'protect' key to the pairs JSON "
                "or pass --protect_concept explicitly."
            )

        concept_embeds: List[torch.Tensor] = []
        anchor_embeds: List[torch.Tensor] = []

        with torch.no_grad():
            for pair in pairs:
                c_emb, _ = pipe.encode_prompt(
                    prompt=pair["concept"],
                    device=config.device,
                    num_images_per_prompt=config.batch_size,
                    do_classifier_free_guidance=False,
                    negative_prompt="",
                )
                concept_embeds.append(c_emb.to(config.device))

                a_emb, _ = pipe.encode_prompt(
                    prompt=pair["anchor"],
                    device=config.device,
                    num_images_per_prompt=config.batch_size,
                    do_classifier_free_guidance=False,
                    negative_prompt="",
                )
                anchor_embeds.append(a_emb.to(config.device))

            # K protection concepts: CLI override > JSON "protect_k" > fallback to single "protect"
            protect_prompts_k = (
                config.protect_concepts_k
                or pairs_data.get("protect_k", [protect_prompt])
            )
            protect_embeds_k = []
            for pp in protect_prompts_k:
                p_emb, _ = pipe.encode_prompt(
                    prompt=pp,
                    device=config.device,
                    num_images_per_prompt=config.batch_size,
                    do_classifier_free_guidance=False,
                    negative_prompt="",
                )
                protect_embeds_k.append(p_emb.to(config.device))

            # Null embed ("") as fixed anchor for protection directions —
            # ensures pres_dir_k = ε(prot_k) - ε(null) is consistent across steps.
            null_emb, _ = pipe.encode_prompt(
                prompt="",
                device=config.device,
                num_images_per_prompt=config.batch_size,
                do_classifier_free_guidance=False,
                negative_prompt="",
            )
            null_embed = null_emb.to(config.device)

        alphas_cumprod = pipe.scheduler.alphas_cumprod.to(config.device)

        timestep_cond = None
        if pipe.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(config.guidance_scale - 1).repeat(config.batch_size)
            timestep_cond = pipe.get_guidance_scale_embedding(
                guidance_scale_tensor,
                embedding_dim=pipe.unet.config.time_cond_proj_dim,
            ).to(device=config.device, dtype=config.torch_dtype)

        offload_modules_to_cpu(config.device, pipe.vae, pipe.text_encoder, pipe.safety_checker)

        return {
            "resolution": resolution,
            "concept_embeds": concept_embeds,
            "anchor_embeds": anchor_embeds,
            "protect_embed": protect_embeds_k[0],   # backward compat
            "protect_embeds_k": protect_embeds_k,   # full list for K-GS + pres loss
            "null_embed": null_embed,               # fixed anchor for protection directions
            "alphas_cumprod": alphas_cumprod,
            "timestep_cond": timestep_cond,
        }

    def training_step(self, pipe, prepared: PreparedComponent, context: Dict[str, Any], config: ESDConfig) -> StepResult:
        n_pairs = len(context["concept_embeds"])
        i = random.randint(0, n_pairs - 1)

        # Bias toward style-informative timestep range [100, 700].
        # Uniform [0,999] wastes ~30% of steps on near-clean (t<100) and pure-noise
        # (t>700) regions where style gradients are minimal.
        t = random.randint(100, 700)

        res = context["resolution"] // 8
        b   = config.batch_size
        N   = config.n_latents_avg   # number of latents to average for stable d_style
        tc  = context["timestep_cond"]

        alpha_bar = context["alphas_cumprod"][t].to(dtype=config.torch_dtype)
        timestep  = torch.tensor([t], device=config.device, dtype=torch.long)

        concept_emb      = context["concept_embeds"][i]    # (b, 77, 768)
        anchor_emb       = context["anchor_embeds"][i]
        null_emb         = context["null_embed"]            # (b, 77, 768) — fixed ""
        protect_embeds_k = context["protect_embeds_k"]     # list of K tensors
        K = len(protect_embeds_k)

        # ── Direction-estimation latents (N independent samples) ──────────────
        x0_est  = torch.randn(N, 4, res, res, device=config.device, dtype=config.torch_dtype)
        eps_est = torch.randn_like(x0_est)
        xt_est  = alpha_bar.sqrt() * x0_est + (1.0 - alpha_bar).sqrt() * eps_est

        # xt used for student/preservation is the first estimation latent
        xt = xt_est[:b]

        # ── Frozen forward A: concept × N and anchor × N in one batched call ──
        # Expanding text embeddings across N latents (works for b=1 which SPACE uses)
        concept_exp = concept_emb.expand(N, -1, -1)   # (N, 77, 768)
        anchor_exp  = anchor_emb.expand(N, -1, -1)    # (N, 77, 768)
        batch_ca    = torch.cat([concept_exp, anchor_exp], dim=0)  # (2N, 77, 768)
        xt_ca = torch.cat([xt_est, xt_est], dim=0)    # (2N, 4, res, res)
        tc_ca = tc.expand(2 * N, -1) if tc is not None else None

        prepared.use_base()
        prepared.component.eval()
        with torch.no_grad():
            preds_ca = prepared.component(
                xt_ca,
                timestep.expand(2 * N),
                encoder_hidden_states=batch_ca,
                timestep_cond=tc_ca,
                cross_attention_kwargs=None,
                added_cond_kwargs=None,
                return_dict=False,
            )[0]

        eps_concepts = preds_ca[:N]     # (N, 4, res, res)
        eps_anchors  = preds_ca[N:2*N]  # (N, 4, res, res)

        # Average style direction over N latents — reduces per-step noise significantly
        avg_d_style    = (eps_concepts - eps_anchors).mean(0, keepdim=True)  # (1, 4, res, res)
        eps_anchor_xt  = eps_anchors[:b]  # prediction for xt specifically (used in target)

        # ── Frozen forward B: null + K protection embeds with xt (single latent) ─
        null_prot_batch = torch.cat([null_emb] + protect_embeds_k, dim=0)  # ((K+1)*b, 77, 768)
        xt_np  = xt.expand(K + 1, -1, -1, -1)
        tc_np  = tc.expand(K + 1, -1) if tc is not None else None

        with torch.no_grad():
            preds_np = prepared.component(
                xt_np,
                timestep.expand(K + 1),
                encoder_hidden_states=null_prot_batch,
                timestep_cond=tc_np,
                cross_attention_kwargs=None,
                added_cond_kwargs=None,
                return_dict=False,
            )[0]

        eps_null_xt = preds_np[:b]  # (b, 4, res, res)
        # Protection directions relative to null — consistent across all steps
        raw_pres_dirs = [preds_np[(j + 1) * b:(j + 2) * b] - eps_null_xt for j in range(K)]

        # ── Modified Gram-Schmidt: orthogonalize protection dirs against each other ──
        # Standard sequential GS on d_style is only correct if the protection directions
        # are mutually orthogonal. MGS on the protection directions first then project.
        ortho_pres_dirs = []
        for raw_dir in raw_pres_dirs:
            q = raw_dir.clone()
            for u in ortho_pres_dirs:
                q_flat = q.reshape(-1).float()
                u_flat = u.reshape(-1).float()
                q = q - (torch.dot(q_flat, u_flat) / (torch.dot(u_flat, u_flat) + 1e-8)) * u
            if q.norm() > 1e-8:
                ortho_pres_dirs.append(q)

        # Project avg_d_style onto orthogonal complement of span{protection directions}
        d_proj = avg_d_style
        for u in ortho_pres_dirs:
            d_flat = d_proj.reshape(-1).float()
            u_flat = u.reshape(-1).float()
            d_proj = d_proj - (torch.dot(d_flat, u_flat) / (torch.dot(u_flat, u_flat) + 1e-8)) * u

        # Normalize to unit norm — rescaling back to avg_d_style magnitude amplifies noise
        # when similar protection styles (e.g. Monet/Cézanne/Renoir) eat most of d_style.
        # Unit norm keeps η as a clean, consistent step size independent of projection residual.
        d_proj_norm = d_proj / (d_proj.norm() + 1e-8)

        # Target: steer concept to anchor behaviour, then push η steps in style direction
        target = eps_anchor_xt - config.negative_guidance * d_proj_norm

        # ── Student forward ───────────────────────────────────────────────────
        prepared.use_student()
        prepared.component.train()
        model_pred = prepared.component(
            xt,
            timestep,
            encoder_hidden_states=concept_emb,
            timestep_cond=tc,
            cross_attention_kwargs=None,
            added_cond_kwargs=None,
            return_dict=False,
        )[0]

        # ── Preservation loss (reuse already-computed frozen preds_np) ────────
        pres_loss = None
        if config.pres_lambda > 0.0:
            k_idx = random.randint(0, K - 1)
            # preds_np layout: [null, prot_0, prot_1, ..., prot_{K-1}] each of size b
            eps_pres_frozen = preds_np[(k_idx + 1) * b:(k_idx + 2) * b]
            pres_emb = protect_embeds_k[k_idx]

            eps_pres_student = prepared.component(
                xt,
                timestep,
                encoder_hidden_states=pres_emb.to(dtype=config.torch_dtype),
                timestep_cond=tc,
                cross_attention_kwargs=None,
                added_cond_kwargs=None,
                return_dict=False,
            )[0]
            pres_loss = F.mse_loss(eps_pres_student.float(), eps_pres_frozen.float())

        metrics = {} if pres_loss is None else {"pres_loss": pres_loss}
        return StepResult(model_pred=model_pred, target=target, timestep_index=t, metrics=metrics)

    def build_checkpoint_path(self, config: ESDConfig) -> str:
        filename = f"space-{sanitize_checkpoint_name(config.erase_concept)}-esdxstrict.safetensors"
        return os.path.join(config.save_path, filename)

    def build_metadata(self, config: ESDConfig) -> Dict[str, str]:
        metadata = super().build_metadata(config)
        metadata["method"] = "space"
        metadata["eta"] = str(config.negative_guidance)
        metadata["protect_concept"] = config.protect_concept or ""
        return metadata


ADAPTERS = {
    "sd": StableDiffusionESDAdapter(),
    "sdxl": StableDiffusionXLESDAdapter(),
    "flux": FluxESDAdapter(),
    "flux2_klein": Flux2KleinESDAdapter(),
    "space-sd": SpaceSDAdapter(),
}


def get_adapter(family: str) -> BaseESDAdapter:
    try:
        return ADAPTERS[family]
    except KeyError as exc:
        raise ValueError(f"Unsupported ESD family: {family}") from exc


def run_esd_training(config: ESDConfig) -> str:
    adapter = get_adapter(config.family)
    config.train_method = adapter.normalize_train_method(config.train_method)
    if config.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    with _suppress_transformers_pipeline_load_noise():
        pipe = adapter.load_pipeline(config)
    pipe.set_progress_bar_config(disable=True)
    component = getattr(pipe, adapter.component_attr)
    if config.gradient_checkpointing and hasattr(component, "enable_gradient_checkpointing"):
        component.enable_gradient_checkpointing()

    prepared = adapter.create_prepared_component(pipe, config.train_method, config)
    prepared.use_student()

    learning_rate = adapter.resolve_learning_rate(config)
    optimizer = torch.optim.Adam(prepared.parameters(), lr=learning_rate)
    context = adapter.prepare_context(pipe, config)

    method = "SPACE" if config.family.startswith("space") else "ESD"
    pbar = tqdm(range(config.iterations), desc=f"Training {method} ({adapter.family})")
    for _ in pbar:
        optimizer.zero_grad(set_to_none=True)
        step_result = adapter.training_step(pipe, prepared, context, config)
        loss = F.mse_loss(step_result.model_pred.float(), step_result.target.float())
        if step_result.metrics.get("pres_loss") is not None:
            loss = loss + config.pres_lambda * step_result.metrics["pres_loss"]
        loss.backward()
        if config.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(list(prepared.parameters()), config.gradient_clip_norm)
        optimizer.step()

        postfix = {"esd_loss": f"{loss.item():.4f}", "timestep": step_result.timestep_index}
        postfix.update({key: str(value) for key, value in step_result.metrics.items()})
        pbar.set_postfix(postfix)

    prepared.use_student()
    checkpoint_path = adapter.build_checkpoint_path(config)
    save_esd_checkpoint(prepared.state_dict(), checkpoint_path, metadata=adapter.build_metadata(config))
    return checkpoint_path
