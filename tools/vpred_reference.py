#!/usr/bin/env python3
"""Render one SDXL image with ComfyUI's own sampling recipe, outside this repo's pipeline.

A reference implementation, kept so a sample this trainer renders can be checked against the
code path ComfyUI uses. Every formula is a copy of ComfyUI's (`--comfy-dir`, default
`/home/acite/LLM/comfyui`); the model is loaded through diffusers, but the sigma table, the
scheduler and the sampler are ComfyUI's:

    comfy/supported_models.py:229   a `v_pred` marker tensor means V_PREDICTION, `ztsnr` adds zsnr
    comfy/model_sampling.py:152     ModelSamplingDiscrete(zsnr=True) sigma table
    comfy/samplers.py:671           normal_scheduler
    comfy/k_diffusion/sampling.py:216  sample_euler_ancestral
    comfy/k_diffusion/sampling.py:68   get_ancestral_step
    comfy/model_base.py:216         calculate_input (x / sqrt(sigma^2 + 1)) and timestep(sigma)
    comfy/model_sampling.py:52      V_PREDICTION.calculate_denoised
    comfy/samplers.py:592           cfg_function, applied to the denoised estimates
    comfy/latent_formats.py:25      SDXL.process_out (scale_factor 0.13025)

Measured against the trainer's own sampler (2026-09-30, NoobAI XL vPred, 1152x768, 50 steps,
cfg 5.0, seed 1234): this script 79.9 mean / 97.5 std, `trainer/sampling.py` 80.1 / 98.0.

Usage:
    python tools/vpred_reference.py --checkpoint /path/model.safetensors \
        --prompt "1girl, ..." --steps 50 --cfg 5.0 --seed 1234 --out /tmp/ref.png

Needs the `axl` env (torch, diffusers) and a ComfyUI checkout next to it for the sigma tables.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

# ComfyUI's own sigma tables, so the schedule is the one ComfyUI builds rather than a copy.
comfy_dir = Path(os.environ.get("AXL_COMFY_DIR", "/home/acite/LLM/comfyui"))
if not (comfy_dir / "comfy" / "model_sampling.py").is_file():
    raise SystemExit(f"no ComfyUI checkout at {comfy_dir} (set AXL_COMFY_DIR)")
sys.path.insert(0, str(comfy_dir))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from comfy.model_sampling import ModelSamplingDiscrete  # noqa: E402
from diffusers import StableDiffusionXLPipeline  # noqa: E402
from PIL import Image  # noqa: E402


def comfy_normal_scheduler(ms, steps: int) -> list[float]:
    """comfy/samplers.py:671 `normal_scheduler`."""
    start = ms.timestep(ms.sigma_max)
    end = ms.timestep(ms.sigma_min)
    append_zero = True
    if math.isclose(float(ms.sigma(end)), 0, abs_tol=0.00001):
        steps += 1
        append_zero = False
    timesteps = torch.linspace(start, end, steps)
    sigs = [float(ms.sigma(t)) for t in timesteps]
    if append_zero:
        sigs.append(0.0)
    return sigs


def get_ancestral_step(sigma_from: float, sigma_to: float, eta: float = 1.0):
    """comfy/k_diffusion/sampling.py:68."""
    if not eta:
        return sigma_to, 0.0
    sigma_up = min(sigma_to, eta * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5)
    return (sigma_to**2 - sigma_up**2) ** 0.5, sigma_up


def v_prediction_denoised(sigma: float, model_output: torch.Tensor, model_input: torch.Tensor):
    """comfy/model_sampling.py:52, sigma_data = 1.0."""
    return (model_input / (sigma**2 + 1.0)
            - model_output * sigma / (sigma**2 + 1.0) ** 0.5)


def encode(pipe, prompt: str, device, dtype, comfy_pad_g: bool = False):
    """comfy/sdxl_clip.py: both encoders take `hidden_states[-2]`, CLIP-G gives the pooled.

    `comfy_pad_g` reproduces ComfyUI's `SDXLClipGTokenizer` padding (token 0) instead of the
    tokenizer's own pad token. The trainer (like kohya) does not remap, so leave it off for a
    like-for-like comparison against `trainer/sampling.py`: the remap moves a sample's mean by
    about one luminance step, which is enough to make a byte comparison fail.
    """
    def one(text: str):
        def forward(tokenizer, encoder, pad_id):
            tokens = tokenizer(text, padding="max_length", max_length=77, truncation=True,
                               return_tensors="pt")
            ids = tokens.input_ids
            if pad_id is not None:
                ids = torch.where(ids == tokenizer.pad_token_id, torch.tensor(pad_id), ids)
            with torch.no_grad():
                out = encoder(ids.to(device), output_hidden_states=True)
            return out.hidden_states[-2], out[0]

        left, _ = forward(pipe.tokenizer, pipe.text_encoder, None)
        right, pooled = forward(pipe.tokenizer_2, pipe.text_encoder_2, 0 if comfy_pad_g else None)
        return torch.cat([left, right], dim=-1).to(dtype), pooled.to(dtype)

    return one(prompt)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="single-file SDXL checkpoint")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative", default="worst quality, low quality, watermark, signature")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--width", type=int, default=1152)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--zsnr", choices=["true", "false"], default="true",
                        help="zero-terminal-SNR betas (ComfyUI's zsnr switch)")
    parser.add_argument("--comfy-pad-g", action="store_true",
                        help="pad CLIP-G with token 0 as ComfyUI does (off = the trainer's own encoding)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    pipe = StableDiffusionXLPipeline.from_single_file(args.checkpoint, dtype=dtype,
                                                     feature_extractor=None)
    unet, vae = pipe.unet.to(device).eval(), pipe.vae
    pipe.text_encoder.to(device)
    pipe.text_encoder_2.to(device)

    ms = ModelSamplingDiscrete(None, zsnr=args.zsnr == "true")
    sigmas = comfy_normal_scheduler(ms, args.steps)
    print(f"[reference] {len(sigmas)} sigmas, sigma_max={sigmas[0]:.3f}", flush=True)

    cond, pooled = encode(pipe, args.prompt, device, dtype, args.comfy_pad_g)
    uncond, uncond_pooled = encode(pipe, args.negative, device, dtype, args.comfy_pad_g)
    time_ids = torch.tensor([[args.height, args.width, 0, 0, args.height, args.width]],
                            dtype=dtype, device=device)

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    latent_shape = (1, 4, args.height // 8, args.width // 8)
    noise = torch.randn(latent_shape, generator=generator, dtype=torch.float32).to(device, dtype)
    x = noise * sigmas[0]  # comfy's noise_scaling on an empty latent

    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        xc = x / (sigma**2 + 1.0) ** 0.5
        t = ms.timestep(torch.tensor(sigma)).reshape(1).to(device).float()
        preds = []
        for embeds, pool in ((cond, pooled), (uncond, uncond_pooled)):
            with torch.no_grad():
                out = unet(xc, t, encoder_hidden_states=embeds, added_cond_kwargs={
                    "text_embeds": pool, "time_ids": time_ids,
                })[0]
            preds.append(v_prediction_denoised(sigma, out.float(), x.float()))
        cfg_result = preds[1] + (preds[0] - preds[1]) * args.cfg
        sigma_down, sigma_up = get_ancestral_step(sigma, sigma_next)
        if sigma_down == 0:
            x = cfg_result.to(dtype)
        else:
            d = (x.float() - cfg_result) / sigma
            step_noise = torch.randn(latent_shape, generator=generator, dtype=torch.float32)
            x = (x.float() + d * (sigma_down - sigma)
                 + step_noise.to(device).float() * sigma_up).to(dtype)

    vae.config.force_upcast = False
    vae.to(device=device)
    # As the trainer decodes (`_decode`): tiling/slicing change the last bits, so a byte-level
    # comparison against a sample only holds with the same settings.
    vae.enable_slicing()
    vae.enable_tiling()
    with torch.no_grad():
        decoded = vae.decode((x / vae.config.scaling_factor).to(device, torch.bfloat16),
                             return_dict=False)[0]
    frame = (decoded / 2 + 0.5).clamp(0, 1)
    array = (frame[0].permute(1, 2, 0).float().cpu().numpy() * 255).round().astype("uint8")

    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(target)
    grey = np.asarray(Image.fromarray(array).convert("L")).astype(np.float32)
    print(f"[reference] wrote {target}  mean={grey.mean():.1f} black={100 * (grey < 16).mean():.1f}%",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
