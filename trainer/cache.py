from __future__ import annotations

from pathlib import Path

import torch
from tqdm.auto import tqdm

from config import TrainConfig
from dataset import SDXLLoraDataset
from env import flush_memory


@torch.no_grad()
def warm_latent_cache(
    dataset: SDXLLoraDataset,
    vae: torch.nn.Module,
    cfg: TrainConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Pre-encode image latents to disk when disk caching is enabled."""
    if not (cfg.cache_latents and cfg.cache_latents_to_disk):
        return

    vae.eval()
    vae.to(device=device, dtype=dtype)

    pbar = tqdm(total=len(dataset), desc="Encoding Latents")
    try:
        for idx in range(len(dataset)):
            item = dataset[idx]
            cache_path = Path(item["cache_path"])

            if cache_path.exists() or item["img_type"] != "pixel":
                pbar.update(1)
                continue

            pixel_values = item["img_data"].unsqueeze(0).to(device=device, dtype=dtype)
            latent = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
            torch.save(latent.squeeze(0).detach().cpu(), cache_path)
            pbar.update(1)
    finally:
        pbar.close()
        vae.to("cpu")
        flush_memory(device)