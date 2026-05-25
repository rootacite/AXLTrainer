# train.py
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from diffusers import DDIMScheduler
from peft import LoraConfig, get_peft_model

from config import TrainConfig
from dataset import SDXLLoraDataset, make_collate_fn
from models import build_optimizer, build_scheduler, load_sdxl_pipeline, save_lora_checkpoint, enable_flash_attention  # 改这里
from text_processing import encode_prompt_batch
from utils import build_time_ids
from collections import defaultdict

# Set environment variables for MIGraphX (AMD GPU) optimization
cache_dir = os.path.join(os.getcwd(), "migraphx_cache")
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

os.environ["ORT_MIGRAPHX_MODEL_CACHE_PATH"] = cache_dir
os.environ["ORT_MIGRAPHX_CACHE_PATH"] = cache_dir

# Enable readline for path auto-completion (Linux/macOS)

torch.backends.cudnn.benchmark = False


def main() -> None:
    cfg = TrainConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.logging_dir, exist_ok=True)

    set_seed(cfg.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with="tensorboard",
        project_dir=cfg.logging_dir,
    )
    weight_dtype = torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float16
    device = accelerator.device

    # 可选：打开 PyTorch SDPA backend 开关
    # 这不会强制一定是 flash kernel，但会允许 PyTorch 选择优化实现
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    except Exception:
        pass

    pipe = load_sdxl_pipeline(cfg.pretrained_model_name_or_path, weight_dtype)
    vae = pipe.vae
    unet = pipe.unet
    tokenizer_1, tokenizer_2 = pipe.tokenizer, pipe.tokenizer_2
    text_encoder_1, text_encoder_2 = pipe.text_encoder, pipe.text_encoder_2
    noise_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    vae.requires_grad_(False)
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet.requires_grad_(False)

    # 关键修改：在 LoRA 包装前，先把 UNet attention 切到 SDPA/FlashAttention 路径
    enable_flash_attention(unet)

    lora_config = LoraConfig(
        r=cfg.network_dim,
        lora_alpha=cfg.network_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora_config)
    unet.enable_gradient_checkpointing()
    unet.print_trainable_parameters()

    vae.to(device=device, dtype=weight_dtype).eval()
    text_encoder_1.to(device=device, dtype=weight_dtype).eval()
    text_encoder_2.to(device=device, dtype=weight_dtype).eval()

    train_dataset = SDXLLoraDataset(cfg)
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.max_data_loader_n_workers,
        pin_memory=True,
        persistent_workers=cfg.persistent_workers,
        collate_fn=make_collate_fn(),
        drop_last=True,
    )

    optimizer = build_optimizer(cfg, unet.parameters())
    steps_per_epoch = max(1, len(dataloader) // cfg.gradient_accumulation_steps)
    total_steps = steps_per_epoch * cfg.epoch
    lr_scheduler = build_scheduler(optimizer, total_steps, cfg)

    if cfg.cache_latents and cfg.cache_latents_to_disk:
        print("Checking/Generating latents cache...")
        if accelerator.is_main_process:
            vae.to(device=device, dtype=weight_dtype).eval()  # 确保 VAE 在 GPU
            items_to_process = [
                i for i in range(len(train_dataset))
                if not train_dataset._cache_path(train_dataset.images[i], 1024, 1024).exists()
            ]

            if items_to_process:
                pbar = tqdm(total=len(train_dataset), desc="Encoding Latents")
                for i in range(len(train_dataset)):
                    data = train_dataset[i]
                    if data["img_type"] == "pixel":
                        pixel_values = data["img_data"].unsqueeze(0).to(device=device, dtype=weight_dtype)
                        with torch.no_grad():
                            latent = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                        torch.save(latent.squeeze(0).detach().cpu(), data["cache_path"])
                    pbar.update(1)
                pbar.close()
                print("Latents caching complete.")

    vae.to("cpu")
    torch.cuda.empty_cache()

    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=cfg.output_name,
            config=vars(cfg)  # 可选：把 config 里的超参数一并存入 TensorBoard
        )

    global_step = 0
    progress = tqdm(range(total_steps), disable=not accelerator.is_local_main_process)

    for epoch in range(cfg.epoch):
        train_dataset.set_epoch(epoch)
        unet.train()

        for batch in dataloader:
            with accelerator.accumulate(unet):
                # 先按 bucket 分组，保证同组里的 latent 尺寸一致
                groups = defaultdict(list)
                for i in range(len(batch["caption"])):
                    bw = int(batch["bucket_w"][i].item())
                    bh = int(batch["bucket_h"][i].item())
                    groups[(bw, bh)].append(i)

                batch_loss_sum = 0.0
                batch_item_count = 0

                for (bucket_w, bucket_h), indices in groups.items():
                    prompts = [batch["caption"][i] for i in indices]
                    latents_list, time_ids_list = [], []

                    for i in indices:
                        img_type = batch["img_type"][i]
                        src_w = int(batch["src_w"][i].item())
                        src_h = int(batch["src_h"][i].item())
                        cache_path = Path(batch["cache_path"][i])
                        img_data = batch["img_data"][i]

                        if img_type == "latent":
                            latent = img_data.to(device=device, dtype=weight_dtype)
                        else:
                            pixel_values = img_data.unsqueeze(0).to(device=device, dtype=weight_dtype)
                            with torch.no_grad():
                                latent = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                            latent = latent.squeeze(0)

                            if cfg.cache_latents and cfg.cache_latents_to_disk:
                                torch.save(latent.detach().cpu(), cache_path)

                        latents_list.append(latent)
                        time_ids_list.append(
                            build_time_ids(
                                original_size=(src_h, src_w),
                                crop_top_left=(0, 0),
                                target_size=(bucket_h, bucket_w),
                                device=device,
                                dtype=weight_dtype,
                            )
                        )

                    # 现在同组尺寸一致，stack 就不会炸了
                    latents = torch.stack(latents_list, dim=0).to(device=device, dtype=weight_dtype)
                    time_ids = torch.stack(time_ids_list, dim=0)

                    prompt_embeds, pooled_prompt_embeds = encode_prompt_batch(
                        prompts=prompts,
                        tokenizer_1=tokenizer_1,
                        tokenizer_2=tokenizer_2,
                        text_encoder_1=text_encoder_1,
                        text_encoder_2=text_encoder_2,
                        clip_skip=cfg.clip_skip,
                        max_token_length=cfg.max_token_length,
                        device=device,
                        dtype=weight_dtype,
                    )

                    noise = torch.randn_like(latents)
                    if cfg.noise_offset > 0:
                        offset = cfg.noise_offset * torch.randn(
                            latents.shape[0], latents.shape[1], 1, 1, device=device, dtype=weight_dtype
                        )
                        noise = noise + offset

                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (latents.shape[0],),
                        device=device,
                        dtype=torch.long,
                    )
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    model_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs={
                            "text_embeds": pooled_prompt_embeds,
                            "time_ids": time_ids,
                        },
                        return_dict=False,
                    )[0]

                    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                    # 按样本数加权，保证多个 bucket 的梯度贡献更稳定
                    scaled_loss = loss * (len(indices) / len(batch["caption"]))
                    accelerator.backward(scaled_loss)

                    batch_loss_sum += loss.item() * len(indices)
                    batch_item_count += len(indices)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), cfg.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                avg_loss = batch_loss_sum / max(1, batch_item_count)
                progress.set_description(f"epoch={epoch + 1}/{cfg.epoch} loss={avg_loss:.4f}")

                # ==================== 【新增：写入 TensorBoard】 ====================
                if accelerator.is_main_process:
                    # 获取 Scheduler 计算的基础学习率
                    current_base_lr = lr_scheduler.get_last_lr()[0]
                    prodigy_d = optimizer.param_groups[0].get("d", 1.0)
                    effective_lr = current_base_lr * prodigy_d

                    accelerator.log({
                        "Train/Loss": avg_loss,
                        "LR/Base_Scheduled": current_base_lr,
                        "LR/Prodigy_D_Factor": prodigy_d,
                        "LR/Effective_Actual_LR": effective_lr
                    }, step=global_step)
                # ====================================================================

                if cfg.save_every_n_steps > 0 and global_step % cfg.save_every_n_steps == 0:
                    save_lora_checkpoint(accelerator, unet, cfg, global_step)

    save_lora_checkpoint(accelerator, unet, cfg, global_step, final=True)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        accelerator.end_training()


if __name__ == "__main__":
    main()
