import inspect
import re
from typing import Any, Dict, List, Optional, Union

import numpy as np
import PIL
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import LongCatImageTransformer2DModel
from diffusers.pipelines.longcat_image.pipeline_output import LongCatImagePipelineOutput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_torch_xla_available, logging
from diffusers.utils.torch_utils import randn_tensor
from utils.task_specific import (
    CONDITION_MODALITY_IDS,
    CONDITION_ORDER,
    PROMPT_TEMPLATE_SUFFIX,
    get_default_prompt,
    get_prompt_template,
    get_task_conditions,
    get_task_id,
    normalize_task,
    required_conditions,
    resolve_negative_task,
)

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)


def split_quotation(prompt, quote_pairs=None):
    word_internal_quote_pattern = re.compile(r"[a-zA-Z]+'[a-zA-Z]+")
    matches_word_internal_quote_pattern = word_internal_quote_pattern.findall(prompt)
    mapping_word_internal_quote = []

    for index, word_src in enumerate(set(matches_word_internal_quote_pattern)):
        word_tgt = "longcat_$##$_longcat" * (index + 1)
        prompt = prompt.replace(word_src, word_tgt)
        mapping_word_internal_quote.append([word_src, word_tgt])

    if quote_pairs is None:
        quote_pairs = [("'", "'"), ('"', '"'), ("‘", "’"), ("“", "”")]
    pattern = "|".join(
        [re.escape(left) + r"[^" + re.escape(left + right) + r"]*?" + re.escape(right) for left, right in quote_pairs]
    )
    parts = re.split(f"({pattern})", prompt)

    result = []
    for part in parts:
        for word_src, word_tgt in mapping_word_internal_quote:
            part = part.replace(word_tgt, word_src)
        if re.match(pattern, part):
            if part:
                result.append((part, True))
        elif part:
            result.append((part, False))
    return result


def prepare_pos_ids(modality_id=0, type="text", start=(0, 0), num_token=None, height=None, width=None):
    if type == "text":
        if num_token is None:
            raise ValueError("num_token is required for text positions.")
        pos_ids = torch.zeros(num_token, 3)
        pos_ids[..., 0] = modality_id
        pos_ids[..., 1] = torch.arange(num_token) + start[0]
        pos_ids[..., 2] = torch.arange(num_token) + start[1]
        return pos_ids
    if type == "image":
        if height is None or width is None:
            raise ValueError("height and width are required for image positions.")
        pos_ids = torch.zeros(height, width, 3)
        pos_ids[..., 0] = modality_id
        pos_ids[..., 1] = pos_ids[..., 1] + torch.arange(height)[:, None] + start[0]
        pos_ids[..., 2] = pos_ids[..., 2] + torch.arange(width)[None, :] + start[1]
        return pos_ids.reshape(height * width, 3)
    raise KeyError(f"Unknown position type: {type}")


def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15):
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    return image_seq_len * slope + base_shift - slope * base_seq_len


def retrieve_timesteps(scheduler, num_inference_steps=None, device=None, timesteps=None, sigmas=None, **kwargs):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of timesteps or sigmas can be provided.")
    if timesteps is not None:
        if "timesteps" not in inspect.signature(scheduler.set_timesteps).parameters:
            raise ValueError("The scheduler does not support custom timesteps.")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
    elif sigmas is not None:
        if "sigmas" not in inspect.signature(scheduler.set_timesteps).parameters:
            raise ValueError("The scheduler does not support custom sigmas.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    return scheduler.timesteps, len(scheduler.timesteps)


def retrieve_latents(encoder_output, generator=None, sample_mode="sample"):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    if hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access VAE latents.")


class LongCatImageOrthoPipeline(DiffusionPipeline, FromSingleFileMixin):
    model_cpu_offload_seq = "text_encoder->image_encoder->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: Qwen2_5_VLForConditionalGeneration,
        tokenizer: Qwen2Tokenizer,
        text_processor: Qwen2VLProcessor,
        transformer: LongCatImageTransformer2DModel,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
            text_processor=text_processor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.image_processor_vl = text_processor.image_processor
        self.image_token = "<|image_pad|>"
        self.tokenizer_max_length = 512

    def _encode_prompt(self, prompt, img_list, task):
        raw_vl_input = self.image_processor_vl(images=img_list, return_tensors="pt")
        pixel_values = raw_vl_input["pixel_values"]
        image_grid_thw = raw_vl_input["image_grid_thw"]

        all_tokens = []
        for clean_prompt_sub, matched in split_quotation(prompt[0]):
            if matched:
                for sub_word in clean_prompt_sub:
                    all_tokens.extend(self.tokenizer(sub_word, add_special_tokens=False)["input_ids"])
            else:
                all_tokens.extend(self.tokenizer(clean_prompt_sub, add_special_tokens=False)["input_ids"])

        if len(all_tokens) > self.tokenizer_max_length:
            logger.warning("Prompt was truncated to %s tokens.", self.tokenizer_max_length)
            all_tokens = all_tokens[: self.tokenizer_max_length]

        text_tokens_and_mask = self.tokenizer.pad(
            {"input_ids": [all_tokens]},
            max_length=self.tokenizer_max_length,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        text = get_prompt_template(task)
        merge_length = self.image_processor_vl.merge_size ** 2
        for index in range(len(img_list)):
            current_thw = image_grid_thw[index]
            num_image_tokens = current_thw.prod() // merge_length
            if "<|image_pad|>" in text:
                text = text.replace("<|image_pad|>", "<|placeholder|>" * num_image_tokens, 1)
        text = text.replace("<|placeholder|>", self.image_token)

        prefix_tokens = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        suffix_tokens = self.tokenizer(PROMPT_TEMPLATE_SUFFIX, add_special_tokens=False)["input_ids"]
        vision_start_token_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        prefix_len = prefix_tokens.index(vision_start_token_id)
        suffix_len = len(suffix_tokens)

        prefix_tokens_mask = torch.tensor([1] * len(prefix_tokens), dtype=text_tokens_and_mask.attention_mask[0].dtype)
        suffix_tokens_mask = torch.tensor([1] * len(suffix_tokens), dtype=text_tokens_and_mask.attention_mask[0].dtype)
        prefix_tokens = torch.tensor(prefix_tokens, dtype=text_tokens_and_mask.input_ids.dtype)
        suffix_tokens = torch.tensor(suffix_tokens, dtype=text_tokens_and_mask.input_ids.dtype)

        input_ids = torch.cat((prefix_tokens, text_tokens_and_mask.input_ids[0], suffix_tokens), dim=-1)
        attention_mask = torch.cat(
            (prefix_tokens_mask, text_tokens_and_mask.attention_mask[0], suffix_tokens_mask), dim=-1
        )
        input_ids = input_ids.unsqueeze(0).to(self.device)
        attention_mask = attention_mask.unsqueeze(0).to(self.device)
        pixel_values = pixel_values.to(self.device)
        image_grid_thw = image_grid_thw.to(self.device)
        if image_grid_thw.ndim == 3:
            image_grid_thw = image_grid_thw.flatten(0, 1)

        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
        )
        return text_output.hidden_states[-1].detach()[:, prefix_len:-suffix_len, :]

    def encode_prompt(self, prompt=None, image=None, num_images_per_prompt=1, prompt_embeds=None, task="vton"):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt_embeds is None:
            if prompt is None or len(prompt) != 1:
                raise ValueError("Only one prompt is supported.")
            prompt_embeds = self._encode_prompt(prompt, image, task)

        batch_size, sequence_length, hidden_size = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, sequence_length, hidden_size)
        text_ids = prepare_pos_ids(
            modality_id=0,
            type="text",
            start=(0, 0),
            num_token=prompt_embeds.shape[1],
        ).to(self.device)
        return prompt_embeds, text_ids

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        return latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, _, channels = latents.shape
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))
        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        return latents.reshape(batch_size, channels // 4, height, width)

    def _encode_vae_image(self, image, generator):
        if isinstance(generator, list):
            latents = [
                retrieve_latents(self.vae.encode(image[index : index + 1]), generator=generator[index], sample_mode="argmax")
                for index in range(image.shape[0])
            ]
            image_latents = torch.cat(latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator, sample_mode="argmax")
        return (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    def check_inputs(self, prompt, height, width, negative_prompt=None, prompt_embeds=None, negative_prompt_embeds=None):
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning("height and width will be resized to valid VAE dimensions.")
        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Provide prompt or prompt_embeds, not both.")
        if prompt is None and prompt_embeds is None:
            raise ValueError("Provide prompt or prompt_embeds.")
        if prompt is not None and not (isinstance(prompt, str) or isinstance(prompt, list) and len(prompt) == 1):
            raise ValueError("prompt must be a string or a list of length one.")
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError("Provide negative_prompt or negative_prompt_embeds, not both.")

    def _set_task(self, task):
        task_id = get_task_id(task)
        for module in self.transformer.modules():
            setter = getattr(module, "set_task", None)
            if callable(setter):
                setter(task_id)

    def _prepare_images(self, conditions, height, width):
        prompt_images = {}
        latent_images = {}
        for name in CONDITION_ORDER:
            image = conditions.get(name)
            if image is None:
                continue
            image = self.image_processor.resize(image, height, width)
            prompt_images[name] = self.image_processor.resize(image, height // 2, width // 2)
            latent_images[name] = self.image_processor.preprocess(image, height, width)
        return prompt_images, latent_images

    def prepare_latents(
        self,
        conditions,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        prompt_embeds_length,
        device,
        generator,
        latents=None,
    ):
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        condition_latents = {}
        condition_ids = {}

        for name in CONDITION_ORDER:
            image = conditions.get(name)
            if image is None:
                continue
            image = image.to(device=self.device, dtype=dtype)
            if image.shape[1] != self.vae.config.latent_channels:
                encoded = self._encode_vae_image(image, generator)
            else:
                encoded = image
            if batch_size > encoded.shape[0] and batch_size % encoded.shape[0] == 0:
                encoded = torch.cat([encoded] * (batch_size // encoded.shape[0]), dim=0)
            elif batch_size > encoded.shape[0]:
                raise ValueError(f"Cannot duplicate {name} batch {encoded.shape[0]} to {batch_size}.")
            condition_latents[name] = self._pack_latents(encoded, batch_size, num_channels_latents, height, width)
            condition_ids[name] = prepare_pos_ids(
                modality_id=CONDITION_MODALITY_IDS[name],
                type="image",
                start=(prompt_embeds_length, prompt_embeds_length),
                height=height // 2,
                width=width // 2,
            ).to(device, dtype=torch.float64)

        latents_ids = prepare_pos_ids(
            modality_id=1,
            type="image",
            start=(prompt_embeds_length, prompt_embeds_length),
            height=height // 2,
            width=width // 2,
        ).to(device)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f"Expected {batch_size} generators, got {len(generator)}.")
        if latents is None:
            latents = randn_tensor(
                (batch_size, num_channels_latents, height, width),
                generator=generator,
                device=device,
                dtype=dtype,
            )
            latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        else:
            latents = latents.to(device=device, dtype=dtype)
        return latents, condition_latents, latents_ids, condition_ids

    @staticmethod
    def _concat_tokens(latents, condition_latents, task):
        return torch.cat([latents, *(condition_latents[name] for name in get_task_conditions(task))], dim=1)

    @staticmethod
    def _concat_ids(latents_ids, condition_ids, task):
        return torch.cat([latents_ids, *(condition_ids[name] for name in get_task_conditions(task))], dim=0)

    @torch.no_grad()
    def __call__(
        self,
        image: Optional[PIL.Image.Image] = None,
        cloth_image: Optional[PIL.Image.Image] = None,
        pose_image: Optional[PIL.Image.Image] = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        task: str = "vton",
        sampler: str = "cfg",
        negative_task: Optional[str] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 2.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        ref_image: Optional[PIL.Image.Image] = None,
    ):
        task = normalize_task(task)
        if ref_image is not None:
            if image is not None:
                raise ValueError("Provide image or ref_image, not both.")
            image = ref_image
        fng_task = resolve_negative_task(task, sampler, negative_task)
        negative_branch_task = fng_task or task
        conditions = {"ref_image": image, "cloth_image": cloth_image, "pose_image": pose_image}
        needed = required_conditions(task, fng_task)
        missing = [name for name in needed if conditions[name] is None]
        if missing:
            raise ValueError(f"Missing conditions for {task}/{negative_branch_task}: {', '.join(missing)}.")

        height, width = 512, 384
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False
        batch_size = 1 if isinstance(prompt, str) else len(prompt) if prompt is not None else prompt_embeds.shape[0]
        device = self._execution_device

        selected = {name: conditions[name] for name in needed}
        prompt_images, latent_images = self._prepare_images(selected, height, width)
        prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            image=[prompt_images[name] for name in get_task_conditions(task)],
            prompt_embeds=prompt_embeds,
            num_images_per_prompt=num_images_per_prompt,
            task=task,
        )

        if self.do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = get_default_prompt(negative_branch_task) if fng_task is not None else ""
            negative_prompt_embeds, negative_text_ids = self.encode_prompt(
                prompt=negative_prompt,
                image=[prompt_images[name] for name in get_task_conditions(negative_branch_task)],
                prompt_embeds=negative_prompt_embeds,
                num_images_per_prompt=num_images_per_prompt,
                task=negative_branch_task,
            )

        latents, condition_latents, latents_ids, condition_ids = self.prepare_latents(
            conditions=latent_images,
            batch_size=batch_size * num_images_per_prompt,
            num_channels_latents=16,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            prompt_embeds_length=prompt_embeds.shape[1],
            device=device,
            generator=generator,
            latents=latents,
        )

        sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        self._joint_attention_kwargs = self._joint_attention_kwargs or {}
        positive_ids = self._concat_ids(latents_ids, condition_ids, task)
        negative_ids = self._concat_ids(latents_ids, condition_ids, negative_branch_task)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for step, timestep_value in enumerate(timesteps):
                if self.interrupt:
                    continue
                self._current_timestep = timestep_value
                positive_input = self._concat_tokens(latents, condition_latents, task)
                timestep = timestep_value.expand(positive_input.shape[0]).to(latents.dtype)

                self._set_task(task)
                with self.transformer.cache_context("cond"):
                    noise_pred_text = self.transformer(
                        hidden_states=positive_input,
                        timestep=timestep / 1000,
                        guidance=None,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=positive_ids,
                        return_dict=False,
                    )[0][:, :image_seq_len]

                if self.do_classifier_free_guidance:
                    negative_input = self._concat_tokens(latents, condition_latents, negative_branch_task)
                    self._set_task(negative_branch_task)
                    with self.transformer.cache_context("uncond"):
                        noise_pred_negative = self.transformer(
                            hidden_states=negative_input,
                            timestep=timestep / 1000,
                            encoder_hidden_states=negative_prompt_embeds,
                            txt_ids=negative_text_ids,
                            img_ids=negative_ids,
                            return_dict=False,
                        )[0][:, :image_seq_len]
                    self._set_task(task)
                    noise_pred = noise_pred_negative + self.guidance_scale * (noise_pred_text - noise_pred_negative)
                else:
                    noise_pred = noise_pred_text

                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, timestep_value, latents, return_dict=False)[0]
                if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)
                if step == len(timesteps) - 1 or (step + 1 > num_warmup_steps and (step + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
            if latents.dtype != self.vae.dtype:
                latents = latents.to(dtype=self.vae.dtype)
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return LongCatImagePipelineOutput(images=image)
