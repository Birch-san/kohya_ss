# training with captions

from PIL import Image
import io
from collections.abc import Iterable
import random
import argparse
import gc
import math
import torchvision
import os
from multiprocessing import Value

from tqdm import tqdm
import torch
from accelerate.utils import set_seed
from diffusers import DDPMScheduler
from library import sdxl_model_util

import library.train_util as train_util
import library.config_util as config_util
import library.sdxl_train_util as sdxl_train_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
import library.custom_train_functions as custom_train_functions
from library.custom_train_functions import (
    apply_snr_weight,
    prepare_scheduler_for_custom_training,
    pyramid_noise_like,
    apply_noise_offset,
    scale_v_prediction_loss_like_noise_prediction,
)
from library.sdxl_original_unet import SdxlUNet2DConditionModel

from store import AspectBucket, ImageStore, AspectDataset, AspectBucketSampler

Image.MAX_IMAGE_PIXELS = None

image_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5], [0.5]),
    ]
)

def train(args):
    train_util.verify_training_args(args)
    train_util.prepare_dataset_args(args, True)
    sdxl_train_util.verify_sdxl_training_args(args)

    assert not args.weighted_captions, "weighted_captions is not supported currently / weighted_captionsは現在サポートされていません"
    assert (
        not args.train_text_encoder or not args.cache_text_encoder_outputs
    ), "cache_text_encoder_outputs is not supported when training text encoder / text encoderを学習するときはcache_text_encoder_outputsはサポートされていません"

    cache_latents = args.cache_latents
    use_dreambooth_method = args.in_json is None

    if args.seed is not None:
        set_seed(args.seed)  # 乱数系列を初期化する

    tokenizer1, tokenizer2 = sdxl_train_util.load_tokenizers(args)

    tokenizer2("fuck", padding="max_length", truncation=True, max_length=77, return_tensors='pt')

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)

    # acceleratorを準備する
    print("prepare accelerator")
    accelerator = train_util.prepare_accelerator(args)

    # mixed precisionに対応した型を用意しておき適宜castする
    weight_dtype, save_dtype = train_util.prepare_dtype(args)
    vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

    # モデルを読み込む
    (
        load_stable_diffusion_format,
        text_encoder1,
        text_encoder2,
        vae,
        unet,
        logit_scale,
        ckpt_info,
    ) = sdxl_train_util.load_target_model(args, accelerator, "sdxl", weight_dtype)
    # logit_scale = logit_scale.to(accelerator.device, dtype=weight_dtype)

    # verify load/save model formats
    if load_stable_diffusion_format:
        src_stable_diffusion_ckpt = args.pretrained_model_name_or_path
        src_diffusers_model_path = None
    else:
        src_stable_diffusion_ckpt = None
        src_diffusers_model_path = args.pretrained_model_name_or_path

    if args.save_model_as is None:
        save_stable_diffusion_format = load_stable_diffusion_format
        use_safetensors = args.use_safetensors
    else:
        save_stable_diffusion_format = args.save_model_as.lower() == "ckpt" or args.save_model_as.lower() == "safetensors"
        use_safetensors = args.use_safetensors or ("safetensors" in args.save_model_as.lower())
        assert save_stable_diffusion_format, "save_model_as must be ckpt or safetensors / save_model_asはckptかsafetensorsである必要があります"

    # Diffusers版のxformers使用フラグを設定する関数
    def set_diffusers_xformers_flag(model, valid):
        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(valid)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        fn_recursive_set_mem_eff(model)

    # モデルに xformers とか memory efficient attention を組み込む
    if args.diffusers_xformers:
        # もうU-Netを独自にしたので動かないけどVAEのxformersは動くはず
        accelerator.print("Use xformers by Diffusers")
        # set_diffusers_xformers_flag(unet, True)
        set_diffusers_xformers_flag(vae, True)
    else:
        # Windows版のxformersはfloatで学習できなかったりxformersを使わない設定も可能にしておく必要がある
        accelerator.print("Disable Diffusers' xformers")
        train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
        vae.set_use_memory_efficient_attention_xformers(args.xformers)

    # 学習を準備する：モデルを適切な状態にする
    training_models = []
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    training_models.append(unet)

    if args.train_text_encoder:
        # TODO each option for two text encoders?
        accelerator.print("enable text encoder training")
        if args.gradient_checkpointing:
            text_encoder1.gradient_checkpointing_enable()
            text_encoder2.gradient_checkpointing_enable()
        training_models.append(text_encoder1)
        training_models.append(text_encoder2)
    else:
        text_encoder1.requires_grad_(False)
        text_encoder2.requires_grad_(False)
        text_encoder1.eval()
        text_encoder2.eval()

    if not cache_latents:
        vae.requires_grad_(False)
        vae.eval()
        vae.to(accelerator.device, dtype=vae_dtype)

    for m in training_models:
        m.requires_grad_(True)
    params = []
    for m in training_models:
        params.extend(m.parameters())
    params_to_optimize = params

    # calculate number of trainable parameters
    n_params = 0
    for p in params:
        n_params += p.numel()
    accelerator.print(f"number of models: {len(training_models)}")
    accelerator.print(f"number of trainable parameters: {n_params}")

    # 学習に必要なクラスを準備する
    accelerator.print("prepare optimizer, data loader etc.")
    _, _, optimizer = train_util.get_optimizer(args, trainable_params=params_to_optimize)

    # dataloaderを準備する
    # DataLoaderのプロセス数：0はメインプロセスになる
    n_workers = min(args.max_data_loader_n_workers, os.cpu_count() - 1)  # cpu_count-1 ただし最大で指定された数まで
    bucket = AspectBucket(ImageStore(), args.train_batch_size)
    dataset = AspectDataset(bucket)
    sampler = AspectBucketSampler(bucket=bucket, dataset=dataset)

    def drop_random(data):
        if not isinstance(data, Iterable):
            return ''
        if random.random() > 0.1:
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
        except Exception:
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
                    return {"error": "YOU FUCKED UP!!!!!!! KILL YOURSELF NOW! YOU SERVE ZERO PURPOSE! YOUR LIFE IS AS VALUABLE AS A SUMMER ANT. IM GONNA KEEP STOMPIN YOU AND YOU ARE GONNA KEEP COMIN BACK AND IM GONNA SEAL UP ALL MY CRACKS AND YOU ARE GONNA KEEP COMIN BACK. YOU WANNA KNOW WHY? CUZ YOU KEEP SMELLIN THE SYRUP"}
        return_dict = {
            "latents": [example[0] for example in examples],
            "input_ids": [tokenizer1(drop_random(example[1]), padding="max_length", truncation=True, max_length=77, return_tensors='pt').input_ids for example in examples],
            "input_ids2": [tokenizer2(drop_random(example[1]), padding="max_length", truncation=True, max_length=77, return_tensors='pt').input_ids for example in examples],
            "original_sizes_hw": torch.stack([torch.LongTensor(example[4]) for example in examples]),
            "crop_top_lefts": torch.stack([torch.LongTensor(example[6]) for example in examples]),
            "target_sizes_hw": torch.stack([torch.LongTensor(example[5]) for example in examples]),
            "source_name": [example[2] for example in examples],
            "source_id": [example[3] for example in examples]
        }
        return return_dict

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=args.max_data_loader_n_workers,
    )

    args.max_train_steps = args.max_train_epochs * math.ceil(
        len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
    )
    accelerator.print(f"override steps. steps for {args.max_train_epochs} epochs is / 指定エポックまでのステップ数: {args.max_train_steps}")

    # lr schedulerを用意する
    lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

    # 実験的機能：勾配も含めたfp16学習を行う　モデル全体をfp16にする
    if args.full_fp16:
        assert (
            args.mixed_precision == "fp16"
        ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
        accelerator.print("enable full fp16 training.")
        unet.to(weight_dtype)
        text_encoder1.to(weight_dtype)
        text_encoder2.to(weight_dtype)

    # acceleratorがなんかよろしくやってくれるらしい
    if args.train_text_encoder:
        unet, text_encoder1, text_encoder2, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder1, text_encoder2, optimizer, train_dataloader, lr_scheduler
        )

        # transform DDP after prepare
        text_encoder1, text_encoder2, unet = train_util.transform_models_if_DDP([text_encoder1, text_encoder2, unet])
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, train_dataloader, lr_scheduler)
        (unet,) = train_util.transform_models_if_DDP([unet])
        text_encoder1.to(weight_dtype)
        text_encoder2.to(weight_dtype)
        text_encoder1.eval()
        text_encoder2.eval()

    # TextEncoderの出力をキャッシュする
    if args.cache_text_encoder_outputs:
        text_encoder1_cache, text_encoder2_cache = sdxl_train_util.cache_text_encoder_outputs(
            args, accelerator, (tokenizer1, tokenizer2), (text_encoder1, text_encoder2), train_dataloader, None
        )
        accelerator.wait_for_everyone()
        # Text Encoder doesn't work on CPU with fp16
        text_encoder1.to("cpu", dtype=torch.float32)
        text_encoder2.to("cpu", dtype=torch.float32)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        text_encoder1_cache = None
        text_encoder2_cache = None
        text_encoder1.to(accelerator.device)
        text_encoder2.to(accelerator.device)

    # 実験的機能：勾配も含めたfp16学習を行う　PyTorchにパッチを当ててfp16でのgrad scaleを有効にする
    if args.full_fp16:
        train_util.patch_accelerator_for_fp16_training(accelerator)

    # resumeする
    train_util.resume_from_local_or_hf_if_specified(accelerator, args)

    # epoch数を計算する
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
        args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

    # 学習する
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    accelerator.print("running training / 学習開始")
    accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
    accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
    accelerator.print(f"  batch size per device / バッチサイズ: {args.train_batch_size}")
    accelerator.print(
        f"  total train batch size (with parallel & distributed & accumulation) / 総バッチサイズ（並列学習、勾配合計含む）: {total_batch_size}"
    )
    accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
    accelerator.print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
    global_step = 0

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
    )
    prepare_scheduler_for_custom_training(noise_scheduler, accelerator.device)

    if accelerator.is_main_process:
        accelerator.init_trackers("finetuning" if args.log_tracker_name is None else args.log_tracker_name)

    for epoch in range(num_train_epochs):

        accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
        current_epoch.value = epoch + 1

        for m in training_models:
            m.train()

        loss_total = 0
        for step, batch in enumerate(train_dataloader):
            if (batch.get("error")):
                print(f"Step {step} is completely fucked up yo yo yo!")
                continue

            current_step.value = global_step
            # with accelerator.accumulate(training_models[0]):  # 複数モデルに対応していない模様だがとりあえずこうしておく
            if True:
                with torch.no_grad():
                    # latentに変換
                    latents = list(map(lambda x: image_transforms(Image.open(io.BytesIO(x)).convert('RGB')), batch["latents"]))
                    latents = torch.stack(latents).to(device=accelerator.device, dtype=vae_dtype)
                    latents = vae.encode(latents).latent_dist.sample().to(weight_dtype)

                    # NaNが含まれていれば警告を表示し0に置き換える
                    if torch.any(torch.isnan(latents)):
                        accelerator.print("NaN found in latents, replacing with zeros")
                        latents = torch.where(torch.isnan(latents), torch.zeros_like(latents), latents)
                latents = latents * sdxl_model_util.VAE_SCALE_FACTOR
                b_size = latents.shape[0]

                input_ids1 = torch.stack(batch["input_ids"])
                input_ids2 = torch.stack(batch["input_ids2"])
                if not args.cache_text_encoder_outputs:
                    with torch.set_grad_enabled(args.train_text_encoder):
                        # Get the text embedding for conditioning
                        # TODO support weighted captions
                        # if args.weighted_captions:
                        #     encoder_hidden_states = get_weighted_text_embeddings(
                        #         tokenizer,
                        #         text_encoder,
                        #         batch["captions"],
                        #         accelerator.device,
                        #         args.max_token_length // 75 if args.max_token_length else 1,
                        #         clip_skip=args.clip_skip,
                        #     )
                        # else:
                        input_ids1 = input_ids1.to(accelerator.device)
                        input_ids2 = input_ids2.to(accelerator.device)
                        encoder_hidden_states1, encoder_hidden_states2, pool2 = sdxl_train_util.get_hidden_states(
                            args,
                            input_ids1,
                            input_ids2,
                            tokenizer1,
                            tokenizer2,
                            text_encoder1,
                            text_encoder2,
                            None if not args.full_fp16 else weight_dtype,
                        )
                else:
                    encoder_hidden_states1 = []
                    encoder_hidden_states2 = []
                    pool2 = []
                    for input_id1, input_id2 in zip(input_ids1, input_ids2):
                        input_id1_cache_key = tuple(input_id1.squeeze(0).flatten().tolist())
                        input_id2_cache_key = tuple(input_id2.squeeze(0).flatten().tolist())
                        encoder_hidden_states1.append(text_encoder1_cache[input_id1_cache_key])
                        hidden_states2, p2 = text_encoder2_cache[input_id2_cache_key]
                        encoder_hidden_states2.append(hidden_states2)
                        pool2.append(p2)
                    encoder_hidden_states1 = torch.stack(encoder_hidden_states1).to(accelerator.device).to(weight_dtype)
                    encoder_hidden_states2 = torch.stack(encoder_hidden_states2).to(accelerator.device).to(weight_dtype)
                    pool2 = torch.stack(pool2).to(accelerator.device).to(weight_dtype)

                # get size embeddings
                orig_size = batch["original_sizes_hw"]
                crop_size = batch["crop_top_lefts"]
                target_size = batch["target_sizes_hw"]
                embs = sdxl_train_util.get_size_embeddings(orig_size, crop_size, target_size, accelerator.device).to(weight_dtype)

                # concat embeddings
                vector_embedding = torch.cat([pool2, embs], dim=1).to(weight_dtype)
                text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(weight_dtype)

                # Sample noise, sample a random timestep for each image, and add noise to the latents,
                # with noise offset and/or multires noise if specified
                noise, noisy_latents, timesteps = train_util.get_noise_noisy_latents_and_timesteps(args, noise_scheduler, latents)

                noisy_latents = noisy_latents.to(weight_dtype)  # TODO check why noisy_latents is not weight_dtype

                # Predict the noise residual
                with accelerator.autocast():
                    noise_pred = unet(noisy_latents, timesteps, text_embedding, vector_embedding)

                target = noise

                if args.min_snr_gamma:
                    # do not mean over batch dimension for snr weight or scale v-pred loss
                    loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
                    loss = loss.mean([1, 2, 3])

                    if args.min_snr_gamma:
                        loss = apply_snr_weight(loss, timesteps, noise_scheduler, args.min_snr_gamma)

                    loss = loss.mean()  # mean over batch dimension
                else:
                    loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                    params_to_clip = []
                    for m in training_models:
                        params_to_clip.extend(m.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                sdxl_train_util.sample_images(
                    accelerator,
                    args,
                    None,
                    global_step,
                    accelerator.device,
                    vae,
                    [tokenizer1, tokenizer2],
                    [text_encoder1, text_encoder2],
                    unet,
                )

                # 指定ステップごとにモデルを保存
                if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
                        sdxl_train_util.save_sd_model_on_epoch_end_or_stepwise(
                            args,
                            False,
                            accelerator,
                            src_path,
                            save_stable_diffusion_format,
                            use_safetensors,
                            save_dtype,
                            epoch,
                            num_train_epochs,
                            global_step,
                            accelerator.unwrap_model(text_encoder1),
                            accelerator.unwrap_model(text_encoder2),
                            accelerator.unwrap_model(unet),
                            vae,
                            logit_scale,
                            ckpt_info,
                        )

            current_loss = loss.detach().item()  # 平均なのでbatch sizeは関係ないはず
            if args.logging_dir is not None:
                logs = {"loss": current_loss, "lr": float(lr_scheduler.get_last_lr()[0])}
                if (
                    args.optimizer_type.lower().startswith("DAdapt".lower()) or args.optimizer_type.lower() == "Prodigy"
                ):  # tracking d*lr value
                    logs["lr/d*lr"] = (
                        lr_scheduler.optimizers[0].param_groups[0]["d"] * lr_scheduler.optimizers[0].param_groups[0]["lr"]
                    )
                accelerator.log(logs, step=global_step)

            # TODO moving averageにする
            loss_total += current_loss
            avr_loss = loss_total / (step + 1)
            logs = {"loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if args.logging_dir is not None:
            logs = {"loss/epoch": loss_total / len(train_dataloader)}
            accelerator.log(logs, step=epoch + 1)

        accelerator.wait_for_everyone()

        if args.save_every_n_epochs is not None:
            if accelerator.is_main_process:
                src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
                sdxl_train_util.save_sd_model_on_epoch_end_or_stepwise(
                    args,
                    True,
                    accelerator,
                    src_path,
                    save_stable_diffusion_format,
                    use_safetensors,
                    save_dtype,
                    epoch,
                    num_train_epochs,
                    global_step,
                    accelerator.unwrap_model(text_encoder1),
                    accelerator.unwrap_model(text_encoder2),
                    accelerator.unwrap_model(unet),
                    vae,
                    logit_scale,
                    ckpt_info,
                )

        sdxl_train_util.sample_images(
            accelerator,
            args,
            epoch + 1,
            global_step,
            accelerator.device,
            vae,
            [tokenizer1, tokenizer2],
            [text_encoder1, text_encoder2],
            unet,
        )

    is_main_process = accelerator.is_main_process
    # if is_main_process:
    unet = accelerator.unwrap_model(unet)
    text_encoder1 = accelerator.unwrap_model(text_encoder1)
    text_encoder2 = accelerator.unwrap_model(text_encoder2)

    accelerator.end_training()

    if args.save_state:  # and is_main_process:
        train_util.save_state_on_train_end(args, accelerator)

    del accelerator  # この後メモリを使うのでこれは消す

    if is_main_process:
        src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
        sdxl_train_util.save_sd_model_on_train_end(
            args,
            src_path,
            save_stable_diffusion_format,
            use_safetensors,
            save_dtype,
            epoch,
            global_step,
            text_encoder1,
            text_encoder2,
            unet,
            vae,
            logit_scale,
            ckpt_info,
        )
        print("model saved.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, False)
    train_util.add_sd_saving_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)
    sdxl_train_util.add_sdxl_training_arguments(parser)

    parser.add_argument("--diffusers_xformers", action="store_true", help="use xformers by diffusers / Diffusersでxformersを使用する")
    parser.add_argument("--train_text_encoder", action="store_true", help="train text encoder / text encoderも学習する")
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)

    train(args)
