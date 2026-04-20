from __future__ import annotations

import gc
import json
import os
import random
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from tqdm.auto import tqdm

from utils.esd_checkpoint import save_esd_checkpoint
from utils.sd_utils import esd_sd_call

# This repo intentionally keeps only the SD v1.x ESD baseline and SPACE
# experiment path. SDXL/FLUX branches were removed to keep debugging focused.


TARGET_MODULE_TYPES = {
    "Linear",
    "Conv2d",
    "LoRACompatibleLinear",
    "LoRACompatibleConv",
}


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
    n_latents_avg: int = 1     # legacy; SPACE now uses on-trajectory latents (ESD-x style)

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
    """Hide benign Transformers 5.x chatter during diffusers `from_pretrained`.

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


class SpaceSDAdapter(BaseESDAdapter):
    """SPACE: Semantically Precise Attribute Concept Erasure for SD v1.x.

    SPACE keeps the ESD-x training distribution: sample an on-trajectory latent
    with the frozen base model, then train the student prediction toward an
    anchor-prompt target instead of the null-prompt ESD target.
    """

    family = "sd"
    component_attr = "unet"
    default_base_model_id = "CompVis/stable-diffusion-v1-4"
    default_save_path = "esd-models/space/"

    def normalize_train_method(self, train_method: str) -> str:
        aliases = {
            "xattn": "esd-x",
            "esd-x": "esd-x",
            "esd-x-strict": "esd-x-strict",
            "xattn-strict": "esd-x-strict",
        }
        normalized = aliases.get(train_method)
        if normalized is None:
            raise ValueError(f"Unsupported SPACE train method: {train_method}")
        return normalized

    def default_lr_for_method(self, train_method: str) -> float:
        return 2e-5 if train_method == "esd-x" else 1e-5

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
            if train_method == "esd-x-strict":
                return "attn2.to_k" in module_name or "attn2.to_v" in module_name
            return False
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

            # Null embed — used as CFG negative prompt in esd_sd_call (same role as ESD-x)
            null_emb, _ = pipe.encode_prompt(
                prompt="",
                device=config.device,
                num_images_per_prompt=config.batch_size,
                do_classifier_free_guidance=False,
                negative_prompt="",
            )
            null_embed = null_emb.to(config.device)

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
            "protect_embeds_k": protect_embeds_k,
            "null_embed": null_embed,
            "timestep_cond": timestep_cond,
        }

    def training_step(self, pipe, prepared: PreparedComponent, context: Dict[str, Any], config: ESDConfig) -> StepResult:
        n_pairs = len(context["concept_embeds"])
        i = random.randint(0, n_pairs - 1)

        run_till_timestep = random.randint(0, config.num_inference_steps - 1)
        seed = random.randint(0, 2**15)

        b   = config.batch_size
        tc  = context["timestep_cond"]

        concept_emb      = context["concept_embeds"][i]    # (b, 77, 768)
        anchor_emb       = context["anchor_embeds"][i]
        null_emb         = context["null_embed"]            # (b, 77, 768) — fixed ""
        protect_embeds_k = context["protect_embeds_k"]     # list of K tensors
        K = len(protect_embeds_k)

        prepared.use_base()
        prepared.component.eval()
        with torch.no_grad():
            xt = esd_sd_call(
                pipe,
                prompt_embeds=concept_emb,
                negative_prompt_embeds=null_emb,
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

            frozen_embeds = torch.cat([concept_emb, anchor_emb] + protect_embeds_k, dim=0)
            xt_batch = xt.expand(K + 2, -1, -1, -1)
            tc_batch = tc.expand(K + 2, -1) if tc is not None else None
            frozen_preds = prepared.component(
                xt_batch,
                timestep.expand(K + 2),
                encoder_hidden_states=frozen_embeds,
                timestep_cond=tc_batch,
                cross_attention_kwargs=None,
                added_cond_kwargs=None,
                return_dict=False,
            )[0]

        eps_concept = frozen_preds[0:b]
        eps_anchor = frozen_preds[b:2 * b]
        eps_protect = [frozen_preds[(j + 2) * b:(j + 3) * b] for j in range(K)]

        d_style = eps_concept - eps_anchor
        raw_pres_dirs = [eps_p - eps_anchor for eps_p in eps_protect]
        ortho_pres_dirs = self._orthogonalize(raw_pres_dirs)

        d_proj = d_style
        for u in ortho_pres_dirs:
            d_flat = d_proj.reshape(-1).float()
            u_flat = u.reshape(-1).float()
            d_proj = d_proj - (torch.dot(d_flat, u_flat) / (torch.dot(u_flat, u_flat) + 1e-8)) * u

        style_norm = d_style.norm().detach()
        proj_norm = d_proj.norm().detach()
        scale = torch.clamp(style_norm / (proj_norm + 1e-8), max=2.0)
        d_proj = d_proj * scale

        target = eps_anchor - config.negative_guidance * d_proj

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
            eps_pres_frozen = eps_protect[k_idx]
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

        metrics = {
            "style_norm": round(float(style_norm), 4),
            "proj_ratio": round(float(proj_norm / (style_norm + 1e-8)), 4),
        }
        if pres_loss is not None:
            metrics["pres_loss"] = pres_loss
        return StepResult(model_pred=model_pred, target=target, timestep_index=run_till_timestep, metrics=metrics)

    @staticmethod
    def _orthogonalize(directions: List[torch.Tensor]) -> List[torch.Tensor]:
        ortho_dirs = []
        for raw_dir in directions:
            q = raw_dir.clone()
            for u in ortho_dirs:
                q_flat = q.reshape(-1).float()
                u_flat = u.reshape(-1).float()
                q = q - (torch.dot(q_flat, u_flat) / (torch.dot(u_flat, u_flat) + 1e-8)) * u
            if q.norm() > 1e-8:
                ortho_dirs.append(q)
        return ortho_dirs

    def build_checkpoint_path(self, config: ESDConfig) -> str:
        method_suffix = config.train_method.replace("-", "")
        filename = f"space-{sanitize_checkpoint_name(config.erase_concept)}-{method_suffix}.safetensors"
        return os.path.join(config.save_path, filename)

    def build_metadata(self, config: ESDConfig) -> Dict[str, str]:
        metadata = super().build_metadata(config)
        metadata["method"] = "space"
        metadata["eta"] = str(config.negative_guidance)
        metadata["protect_concept"] = config.protect_concept or ""
        return metadata


ADAPTERS = {
    "sd": StableDiffusionESDAdapter(),
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
