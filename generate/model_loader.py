import torch
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLImg2ImgPipeline,
    AutoencoderKL,
    EulerAncestralDiscreteScheduler
)
import generate.config as config

def configure_rocm_optimizations():
    if torch.cuda.is_available():
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True

def load_base_pipeline():
    configure_rocm_optimizations()
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=config.TORCH_DTYPE
    )
    pipe = StableDiffusionXLPipeline.from_pretrained(
        config.BASE_MODEL_PATH,
        vae=vae,
        torch_dtype=config.TORCH_DTYPE,
        use_safetensors=True
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config,
    #    timestep_spacing="trailing",
    #    prediction_type="epsilon",
        use_karras_sigmas=False,
    )
    if config.LORA_PATH:
        pipe.load_lora_weights(config.LORA_PATH)
        pipe.fuse_lora(lora_scale=config.LORA_SCALE)
    pipe.vae.enable_tiling()
    if config.DEVICE == "cuda":
        pipe.enable_attention_slicing()
    return pipe


def load_inpaint_pipeline_from_base(base_pipe):
    inpaint_pipe = StableDiffusionXLInpaintPipeline(
        vae=base_pipe.vae,
        text_encoder=base_pipe.text_encoder,
        text_encoder_2=base_pipe.text_encoder_2,
        tokenizer=base_pipe.tokenizer,
        tokenizer_2=base_pipe.tokenizer_2,
        unet=base_pipe.unet,
        scheduler=base_pipe.scheduler,
        feature_extractor=None
    )

    # Apply the same VRAM memory management techniques to the detailer pipeline
    inpaint_pipe.vae.enable_tiling()

    if config.DEVICE == "cuda":
        inpaint_pipe.enable_attention_slicing()

    return inpaint_pipe


def load_img2img_pipeline_from_base(base_pipe):
    img2img_pipe = StableDiffusionXLImg2ImgPipeline(
        vae=base_pipe.vae,
        text_encoder=base_pipe.text_encoder,
        text_encoder_2=base_pipe.text_encoder_2,
        tokenizer=base_pipe.tokenizer,
        tokenizer_2=base_pipe.tokenizer_2,
        unet=base_pipe.unet,
        scheduler=base_pipe.scheduler,
        feature_extractor=None
    )

    img2img_pipe.vae.enable_tiling()

    if config.DEVICE == "cuda":
        img2img_pipe.enable_attention_slicing()

    return img2img_pipe