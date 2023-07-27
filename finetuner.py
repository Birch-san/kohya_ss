import os
import argparse
import socket
import time
import torch
import torchvision
import transformers
import diffusers
import random
import tqdm
import resource
import math
import psutil
import pynvml
import sys
import wandb
import gc
import accelerate
import pickle
import itertools
import copy
import numpy as np
import io
from connector.store import AspectBucket, AspectDataset, AspectBucketSampler, ImageStore
from PIL import Image
from collections.abc import Iterable

try:
    pynvml.nvmlInit()
except pynvml.nvml.NVMLError_LibraryNotFound:
    pynvml = None

from typing import Iterable, Optional
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.optimization import get_scheduler
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer

# Latent Scale Factor - https://github.com/huggingface/diffusers/issues/437
L_SCALE_FACTOR = 0.13025

# defaults should be good for everyone
bool_t = lambda x: (str(x).lower() in ["true", "1", "t", "y", "yes"])
parser = argparse.ArgumentParser(description="Stable Diffusion Finetuner")
parser.add_argument(
    "--model",
    type=str,
    default=None,
    required=True,
    help="The name of the model to use for finetuning. Could be HuggingFace ID or a directory",
)
parser.add_argument(
    "--run_name",
    type=str,
    default=None,
    required=True,
    help="Name of the finetune run.",
)
parser.add_argument(
    "--dataset",
    type=str,
    default=None,
    required=False,
    help="The path to the pickled data file to use for finetuning.",
)
parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
parser.add_argument(
    "--epochs", type=int, default=10, help="Number of epochs to train for"
)
parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
parser.add_argument(
    "--use_ema", type=bool_t, default="False", help="Use EMA for finetuning"
)
parser.add_argument(
    "--ucg",
    type=float,
    default=0.1,
    help="Percentage chance of dropping out the text condition per batch. Ranges from 0.0 to 1.0 where 1.0 means 100% text condition dropout.",
)  # 10% dropout probability
parser.add_argument(
    "--partial_dropout",
    type=bool,
    default=True,
    help="Enable randomly dropping part of the conditioning"
)
parser.add_argument(
    "--gradient_checkpointing",
    dest="gradient_checkpointing",
    type=bool_t,
    default="False",
    help="Enable gradient checkpointing",
)
parser.add_argument(
    "--use_8bit_adam",
    dest="use_8bit_adam",
    type=bool_t,
    default="False",
    help="Use 8-bit Adam optimizer",
)
parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
parser.add_argument(
    "--adam_beta2", type=float, default=0.999, help="Adam beta2"
)
parser.add_argument(
    "--adam_weight_decay", type=float, default=1e-2, help="Adam weight decay"
)
parser.add_argument(
    "--adam_epsilon", type=float, default=1e-08, help="Adam epsilon"
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Seed for random number generator, this is to be used for reproducibility purposes.",
)
parser.add_argument(
    "--output_path",
    type=str,
    default="./output",
    help="Root path for all outputs.",
)
parser.add_argument(
    "--save_steps",
    type=int,
    default=1000,
    help="Number of steps to save checkpoints at.",
)
parser.add_argument(
    "--resume_steps",
    type=int,
    default=-1,
    help="Step at which to resume.",
)
parser.add_argument(
    "--shuffle",
    dest="shuffle",
    type=bool_t,
    default="True",
    help="Shuffle dataset",
)
parser.add_argument(
    "--hf_token",
    type=str,
    default="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",
    required=False,
    help="A HuggingFace token is needed to download private models for training.",
)
parser.add_argument(
    "--project_id",
    type=str,
    default="diffusers",
    help="Project ID for reporting to WandB",
)
parser.add_argument(
    "--fp16",
    dest="fp16",
    type=bool_t,
    default="False",
    help="Train in mixed precision",
)
parser.add_argument(
    "--image_log_steps",
    type=int,
    default=200,
    help="Number of steps to log images at.",
)
parser.add_argument(
    "--image_log_amount",
    type=int,
    default=5,
    help="Number of images to log every image_log_steps",
)
parser.add_argument(
    "--vae",
    type=str,
    default="vae",
    required=False,
    help="A path to a vae to use.",
)
parser.add_argument(
    "--unet",
    type=str,
    default=None,
    required=False,
    help="A path to a unet to use.",
)
parser.add_argument(
    "--lr_scheduler",
    type=str,
    default="constant",
    help=(
        'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
        ' "constant", "constant_with_warmup"]'
    ),
)
parser.add_argument(
    "--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
)
parser.add_argument(
    "--lr_num_cycles", type=int, default=1, help="Number of cycles for cosine_with_restarts lr scheduler."
)
parser.add_argument(
    "--lr_min_scale", type=float, default=0.0, help="Minimum scaling factor for cosine_with_restarts lr scheduler."
)
parser.add_argument(
    "--lr_max_scale", type=float, default=1.0, help="Maximum scaling factor for cosine_with_restarts lr scheduler."
)
parser.add_argument('--use_xformers', type=bool_t, default='False', help='Use memory efficient attention')
parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")
parser.add_argument('--extended_mode_chunks', type=int, default=0, help='Enables extended mode for tokenization with given amount of maximum chunks. Values < 2 disable.')
parser.add_argument('--clip_penultimate', action="store_true", default=False, help='Use penultimate CLIP layer for text embedding')
parser.add_argument(
    "--caption_log_steps", type=int, default=0, help="Number of steps to log captions, 0 disables"
)
parser.add_argument("--text_encoder_learning_rate", type=float, default=7e-9, help="Learning rate for the text encoder")
args = parser.parse_args()


def get_gpu_ram() -> str:
    """
    Returns memory usage statistics for the CPU, GPU, and Torch.

    :return:
    """
    gpu_str = ""
    torch_str = ""
    try:
        cuda_dev = torch.cuda.current_device()
        nvml_device = pynvml.nvmlDeviceGetHandleByIndex(cuda_dev)
        gpu_info = pynvml.nvmlDeviceGetMemoryInfo(nvml_device)
        gpu_total = int(gpu_info.total / 1e6)
        gpu_free = int(gpu_info.free / 1e6)
        gpu_used = int(gpu_info.used / 1e6)
        gpu_str = (
            f"GPU: (U: {gpu_used:,}mb F: {gpu_free:,}mb "
            f"T: {gpu_total:,}mb) "
        )
        torch_reserved_gpu = int(torch.cuda.memory.memory_reserved() / 1e6)
        torch_reserved_max = int(torch.cuda.memory.max_memory_reserved() / 1e6)
        torch_used_gpu = int(torch.cuda.memory_allocated() / 1e6)
        torch_max_used_gpu = int(torch.cuda.max_memory_allocated() / 1e6)
        torch_str = (
            f"TORCH: (R: {torch_reserved_gpu:,}mb/"
            f"{torch_reserved_max:,}mb, "
            f"A: {torch_used_gpu:,}mb/{torch_max_used_gpu:,}mb)"
        )
    except AssertionError:
        pass
    cpu_max_rss = int(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3
        + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e3
    )
    cpu_vmem = psutil.virtual_memory()
    cpu_free = int(cpu_vmem.free / 1e6)
    return (
        f"CPU: (max_rss: {cpu_max_rss:,}mb F: {cpu_free:,}mb) "
        f"{gpu_str}"
        f"{torch_str}"
    )

# Adapted from torch-ema https://github.com/fadel/pytorch_ema/blob/master/torch_ema/ema.py#L14
class EMAModel:
    """
    Exponential Moving Average of models weights
    """

    def __init__(self, parameters: Iterable[torch.nn.Parameter], decay=0.9999):
        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]

        self.decay = decay
        self.optimization_step = 0

    def get_decay(self, optimization_step: int) -> float:
        """
        Compute the decay factor for the exponential moving average.
        """
        value = (1 + optimization_step) / (10 + optimization_step)
        return 1 - min(self.decay, value)

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        parameters = list(parameters)

        self.optimization_step += 1
        self.decay = self.get_decay(self.optimization_step)

        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                tmp = self.decay * (s_param - param)
                s_param.sub_(tmp)
            else:
                s_param.copy_(param)

        torch.cuda.empty_cache()

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Copy current averaged parameters into given collection of parameters.
        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)

    def store(
        self,
        parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ) -> None:
        """
        Save the current parameters for restoring later.
        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                temporarily stored. If `None`, the parameters of with which this
                `ExponentialMovingAverage` was initialized will be used.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(
        self,
        parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ) -> None:
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored parameters. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

        del self.collected_params
        gc.collect()

    def to(self, device=None, dtype=None) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.
        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        # .to() on the tensors handles None correctly
        self.shadow_params = [
            p.to(device=device, dtype=dtype)
            if p.is_floating_point()
            else p.to(device=device)
            for p in self.shadow_params
        ]


class StableDiffusionTrainer:
    def __init__(
        self,
        accelerator: accelerate.Accelerator,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder2: CLIPTextModel,
        tokenizer2: CLIPTokenizer,
        ema: EMAModel,
        train_dataloader: torch.utils.data.DataLoader,
        noise_scheduler: DDPMScheduler,
        lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
        optimizer: torch.optim.Optimizer,
        weight_dtype: torch.dtype,
        args: argparse.Namespace,
    ):
        self.accelerator = accelerator
        self.vae = vae
        self.unet = unet
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.text_encoder2 = text_encoder2
        self.tokenizer2 = tokenizer2
        self.ema = ema
        self.train_dataloader = train_dataloader
        self.noise_scheduler = noise_scheduler
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.weight_dtype = weight_dtype
        self.args = args

        self.image_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.5], [0.5]),
            ]
        )

        if accelerator.is_main_process:
            self.progress_bar = tqdm.tqdm(
                range(args.epochs * len(self.train_dataloader)),
                desc="Total Steps",
                leave=False,
            )
        self.run = wandb.init(
            project=args.project_id,
            name=f"{args.run_name}-p{accelerator.local_process_index}",
            config={
                k: v for k, v in vars(args).items() if k not in ["hf_token"]
            },
            dir=args.output_path + "/wandb",
            group=f"{args.run_name}-group"
        )
        self.global_step = 0

    def save_checkpoint(self):
        unet = self.accelerator.unwrap_model(self.unet)
        if args.use_ema:
            self.ema.store(unet.parameters())
            self.ema.copy_to(unet.parameters())
        print(f'Saving model (step: {self.global_step})...')
        unet.save_pretrained(os.path.join(args.output_path, 'step_' + str(self.global_step)), safe_serialization=True)
        if args.use_ema:
            self.ema.restore(unet.parameters())
        del unet
        gc.collect()
    
    def get_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.accelerator.unwrap_model(self.unet).config.addition_time_embed_dim * len(add_time_ids) + self.accelerator.unwrap_model(self.text_encoder2).config.projection_dim
        )
        expected_add_embed_dim = self.accelerator.unwrap_model(self.unet).add_embedding.linear_1.in_features

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids
    
    def pooled_inputs(self, text):
        encs = [(self.tokenizer, self.accelerator.unwrap_model(self.text_encoder)), (self.tokenizer2, self.accelerator.unwrap_model(self.text_encoder2))]
        prompt_embs = []
        for enc in encs:
            text_inputs = enc[0](
                text,
                padding="max_length",
                max_length=enc[0].model_max_length,
                truncation=True,
                return_tensors='pt'
            ).input_ids

            prompt_embeds = enc[1](
                text_inputs.to(self.accelerator.device),
                output_hidden_states=True,
            )

            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]

            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, 1, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * 1, seq_len, -1)

            prompt_embs.append(prompt_embeds)
        
        prompt_embs = torch.concat(prompt_embs, dim=-1)

        return prompt_embs, pooled_prompt_embeds

    def sub_step(self, batch: dict, epoch: int) -> torch.Tensor:
        # Load our network-streamed images using PIL
        latents = list(map(lambda x: self.image_transforms(Image.open(io.BytesIO(x)).convert('RGB')), batch["latents"]))
        # Make sure we have latents of all the same size
        # (this should be true unless there is a db or preprocessing error)
        latent_sizes = {}
        for idx, l in enumerate(latents):
            if l.size() in latent_sizes:
                latent_sizes[l.size()] = (idx, latent_sizes[l.size()][1] + 1)
            else:
                latent_sizes[l.size()] = (idx, 1)
        largest_latent = max(list(latent_sizes.items()), key=lambda x:x[1][1])[1][0]

        for idx, l in enumerate(latents):
            if l.size() != latents[largest_latent].size():
                print(
                    f'ERROR: Uneven latent size found at step {self.global_step} ({l.size()} -> {latents[largest_latent].size()})! Replacing...'
                )
                self.run.alert(
                    title="Uneven Latent Size",
                    text=f"Step: {self.global_step} ({l.size()} -> {latents[largest_latent].size()})",
                    level=wandb.AlertLevel.WARN
                )

                latents[idx] = latents[largest_latent].clone()
                batch["captions"][idx] = batch["captions"][largest_latent]

        # Finally stack our latents of the same guaranteed size
        latents = torch.stack(latents).to(
            self.accelerator.device, dtype=self.vae.dtype)

        latents = self.vae.encode(latents).latent_dist.sample()
        latents = latents * L_SCALE_FACTOR

        if torch.isnan(latents.any()):
            raise Exception('Latents are nan')

        # Sample noise
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.num_train_timesteps,
            (bsz,),
            device=latents.device,
        )
        timesteps = timesteps.long()


        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(
            latents, noise, timesteps
        )

        # Encode captions with respect to extended mode and penultimate options
        prompt_embeds, pooled_prompt_embeds = self.pooled_inputs(batch["captions"])

        # Prepare added time IDs & embeddings
        target = tuple(torch.tensor(list(latent_sizes.keys())[0][1:]).tolist())

        add_text_embeds = pooled_prompt_embeds.view(1, -1)
        add_time_ids = self.get_time_ids(
            target,
            (0, 0),
            target,
            dtype=prompt_embeds.dtype
        )

        prompt_embeds = prompt_embeds.to(self.accelerator.device)
        add_text_embeds = add_text_embeds.to(self.accelerator.device)
        add_time_ids = add_time_ids.to(self.accelerator.device).repeat(args.batch_size, 1)

        added_cond_kwargs = {
            "text_embeds": add_text_embeds,
            "time_ids": add_time_ids
        }
        
        # Predict the noise residual and compute loss
        #noisy_latents = self.noise_scheduler.scale_model_input(noisy_latents, timesteps)
        if torch.isnan(noisy_latents.any()):
            raise Exception('noisy_latents is nan')
        if torch.isnan(timesteps.any()):
            raise Exception('timesteps is nan')
        if torch.isnan(prompt_embeds.any()):
            raise Exception('prompt_embeds is nan')

        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs
        ).sample

        # Pew pew
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(
                latents, noise, timesteps
            )
        else:
            raise ValueError(
                f"Invalid prediction type: {self.noise_scheduler.config.prediction_type}"
            )

        loss = torch.nn.functional.mse_loss(
            noise_pred.float(), target.float(), reduction="mean"
        )

        # Backprop
        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            if args.train_text_encoder:
                # 1 / sqrt(2) because we are clipping two different gradients
                self.accelerator.clip_grad_norm_(self.text_encoder.parameters(), 2 ** -0.5)
                self.accelerator.clip_grad_norm_(self.unet.parameters(), 2 ** -0.5)
            else:
                self.accelerator.clip_grad_norm_(self.unet.parameters(), 1.0) 
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        self.run.log({
            "rank/loss": loss.detach().item()
        }, step=self.global_step)

        del latents
        return self.accelerator.gather_for_metrics(loss).mean()

    def step(self, batch: dict, epoch: int) -> dict:
        with self.accelerator.accumulate(self.unet):
            loss = self.sub_step(batch, epoch)
        if self.accelerator.sync_gradients:
            # Update EMA
            if args.use_ema:
                self.ema.step(self.unet.parameters())

        return {
            "train/loss": loss.detach().item(),
            "train/lr": self.lr_scheduler.get_last_lr()[0],
        }

    def train(self) -> None:
        for epoch in range(args.epochs):
            self.unet.train()
            if args.train_text_encoder:
                self.text_encoder.train()
            for _, batch in enumerate(self.train_dataloader):
                step_start = time.perf_counter()
                if self.global_step < args.resume_steps:
                    if self.accelerator.is_main_process:
                        self.progress_bar.update(1)
                    self.global_step += 1
                    self.lr_scheduler.step()
                    del batch["captions"]
                    del batch["latents"]
                    del batch
                    continue

                logs = self.step(batch, epoch)

                self.global_step += 1

                if args.caption_log_steps > 0 and self.global_step % args.caption_log_steps == 0:
                    self.captions_table_labels = ['step']
                    for key in batch:
                        if key != 'latents' and key != 'captions':
                            self.captions_table_labels.append(key)
                    self.captions_table_labels.append('captions')
                    self.captions_table = wandb.Table(columns=self.captions_table_labels)
                    for i in range(len(batch["captions"])):
                        caption_data = [self.global_step]
                        for key in self.captions_table_labels[1:]:
                            caption_data.append(batch[key][i])
                        self.captions_table.add_data(*caption_data)
                    self.run.log({"rank/captions": self.captions_table}, step=self.global_step)

                if self.accelerator.is_main_process:
                    rank_samples_per_second = args.batch_size * (
                        1 / (time.perf_counter() - step_start)
                    )
                    world_samples_per_second = (
                        rank_samples_per_second * self.accelerator.num_processes
                    )
                    logs.update(
                        {
                            "perf/rank_samples_per_second": rank_samples_per_second,
                            "perf/world_samples_per_second": world_samples_per_second,
                            "train/epoch": epoch,
                            "train/step": self.global_step,
                            "train/samples_seen": self.global_step * self.accelerator.num_processes * args.batch_size,
                        }
                    )

                    # Output GPU RAM to flush tqdm
                    if not hasattr(self, 'report_idx'):
                        self.report_idx = 1
                    else:
                        self.report_idx += 1
                    if self.report_idx % 100 == 0:
                        print(f"\nLOSS: {logs['train/loss']} {get_gpu_ram()}", file=sys.stderr)
                        sys.stderr.flush()

                    self.progress_bar.update(1)
                    self.progress_bar.set_postfix(**logs)

                    self.run.log(logs, step=self.global_step)

                    if self.global_step % args.save_steps == 0:
                        self.save_checkpoint()

                del batch["captions"]
                del batch["latents"]
                del batch

        self.accelerator.wait_for_everyone()
        self.save_checkpoint()

def get_cosine_with_hard_restarts_schedule_with_warmup_and_scaling(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 1, 
    last_epoch: int = -1,
    max_scale: float = 1.0,
    min_scale: float = 0.0
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (max_scale + min_scale + (max_scale - min_scale) * math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def main() -> None:
    if args.hf_token is None:
        try:
            args.hf_token = os.environ["HF_API_TOKEN"]
        except KeyError:
            print(
                "Please set HF_API_TOKEN environment variable or pass --hf_token"
            )
            exit(1)
    else:
        print(
            "WARNING: Using HF_API_TOKEN from command line. This is insecure. Use environment variables instead."
        )

    # get device
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="fp16" if args.fp16 else "no",
        even_batches=False
    )

    # Set seed
    accelerate.utils.set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_path, exist_ok=True)
        # Inform the user of host, and various versions -- useful for debugging issues.
        print("RUN_NAME:", args.run_name)
        print("HOST:", socket.gethostname())
        print("CUDA:", torch.version.cuda)
        print("TORCH:", torch.__version__)
        print("TRANSFORMERS:", transformers.__version__)
        print("DIFFUSERS:", diffusers.__version__)
        print("MODEL:", args.model)
        print("FP16:", args.fp16)
        print("RANDOM SEED:", args.seed)

    tokenizer = CLIPTokenizer.from_pretrained(
        args.model, subfolder="tokenizer", use_auth_token=args.hf_token
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.model, subfolder="text_encoder", use_auth_token=args.hf_token
    )
    tokenizer2 = CLIPTokenizer.from_pretrained(
        args.model, subfolder="tokenizer_2", use_auth_token=args.hf_token
    )
    text_encoder2 = CLIPTextModelWithProjection.from_pretrained(
        args.model, subfolder="text_encoder_2", use_auth_token=args.hf_token
    )
    if args.vae is None:
        vae = AutoencoderKL.from_pretrained(
            args.model, subfolder="vae", use_auth_token=args.hf_token
        )
    else:
        vae = AutoencoderKL.from_pretrained(
            args.vae, use_auth_token=args.hf_token
        )
        if accelerator.is_main_process:
            print('VAE:', args.vae)

    unet_path = args.model
    unet_subfolder="unet"
    if args.unet is not None:
        unet_path = args.unet
        unet_subfolder=None

    unet = UNet2DConditionModel.from_pretrained(
        unet_path, subfolder=unet_subfolder, use_auth_token=args.hf_token
    )

    # Freeze vae and (maybe) text_encoder
    vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)
        text_encoder2.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    if args.use_xformers:
        unet.set_use_memory_efficient_attention_xformers(True)

    if (
        args.use_8bit_adam
    ):  # Bits and bytes is only supported on certain CUDA setups, so default to regular adam if it fails.
        try:
            import bitsandbytes as bnb

            optimizer_cls = bnb.optim.AdamW8bit
        except:
            print("bitsandbytes not supported, using regular Adam optimizer")
            optimizer_cls = torch.optim.AdamW
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters() if not args.train_text_encoder else [
            {"params": unet.parameters()},
            {"params": text_encoder.parameters(), "lr": args.text_encoder_learning_rate}
        ],
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        weight_decay=args.adam_weight_decay,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.model, subfolder="scheduler", use_auth_token=args.hf_token
    )

    def drop_random(data):
        if not isinstance(data, Iterable):
            return ''
        if random.random() > args.ucg:
            if args.partial_dropout:
                # the equation https://www.desmos.com/calculator/yrfoynzfcp is used
                # to keep a random percent of the data, where the random number is the x-axis
                x = random.randint(0, 100)
                if x >= 50:
                    return ', '.join(data)
                else:
                    return ', '.join(data[:len(data) * x * 2 // 100])
            return ', '.join(data)
        else:
            # drop for unconditional guidance
            return ''

    def is_valid_image(image_bytes):
        try:
            Image.open(io.BytesIO(image_bytes))
            return True
        except:
            return False

    def collate_fn(examples):
        valid_examples = []
        for i in range(len(examples)):
            if examples[i][0] is not None or is_valid_image(examples[i][0]):
                valid_examples.append(examples[i])
            else:
                if valid_examples:
                    examples[i] = random.choice(valid_examples)
                else:
                    raise ValueError("No valid images in batch")
        return_dict = {
            "latents": [example[0] for example in examples],
            "captions": [drop_random(example[1]) for example in examples],
            "source_name": [example[2] for example in examples],
            "source_id": [example[3] for example in examples]
        }

        return return_dict

    #with open(args.dataset, 'rb') as f:
    #    bucket: AspectBucket = pickle.load(f)

    bucket = AspectBucket(ImageStore(), args.batch_size)
    dataset = AspectDataset(bucket)
    sampler = AspectBucketSampler(bucket=bucket, dataset=dataset)

    if accelerator.is_main_process:
        print(f"Loaded {len(dataset)} images from bucket.")
        print(f"Total of {len(sampler)} batches found.")
        args.batch_size = bucket.batch_size
        print(f"BATCH SIZE: {args.batch_size}")
        if args.resume_steps > 0:
            print(f"RESUME: {args.resume_steps}")

    # prefetch_factor is 2 by default ->
    # 2 * num_workers batches will prefetch
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=6,
        collate_fn=collate_fn
    )

    if args.lr_scheduler == 'cosine_with_restarts':
        lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup_and_scaling(
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=args.epochs * len(train_dataloader),
            num_cycles=args.lr_num_cycles,
            max_scale=args.lr_max_scale,
            min_scale=args.lr_min_scale
        )
    else:
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=args.epochs * len(train_dataloader)
        )

    if not args.train_text_encoder:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )

    weight_dtype = torch.float16 if args.fp16 else torch.float32

    # move models to device
    vae = vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder = text_encoder.to(
            accelerator.device, dtype=weight_dtype
        )
        text_encoder2 = text_encoder2.to(
            accelerator.device, dtype=weight_dtype
        )
    else:
         text_encoder = text_encoder.to(accelerator.device)

    #unet = unet.to(accelerator.device, dtype=weight_dtype)

    # create ema
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters())

    print(get_gpu_ram())

    trainer = StableDiffusionTrainer(
        accelerator,
        vae,
        unet,
        text_encoder,
        tokenizer,
        text_encoder2,
        tokenizer2,
        ema_unet if args.use_ema else None,
        train_dataloader,
        noise_scheduler,
        lr_scheduler,
        optimizer,
        weight_dtype,
        args,
    )
    trainer.train()

    if accelerator.is_main_process:
        print(get_gpu_ram())
        print("Done!")

if __name__ == "__main__":
    main()
