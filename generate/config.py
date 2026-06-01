# config.py
import torch

# Environment & Hardware Optimizations (ROCm-friendly)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# Model Paths (Change to your local absolute paths or HuggingFace repo IDs)
BASE_MODEL_PATH = "/home/acite/LLM/models/diffusers/waillu_170"
LORA_PATH = "/home/acite/LLM/models/loras/miyako-1720.safetensors"
LORA_SCALE = 0.9

# Base Image Generation Parameters
WIDTH = 1280
HEIGHT = 720
STEPS = 60
CFG_SCALE = 5.5
SEED = 8576160563625674040

# Prompts from ComfyUI workflow nodes
# POSITIVE_PROMPT = "miyako_style, newest, soft shading, large breasts, solo, looking at viewer, white thighhighs, pajamas, panties, off-shoulder, kneel-sitting, sitting on bed, open collar, cleavage, one nipple, shy, blush"
POSITIVE_PROMPT = "miyako_style, newest, soft shading, white thighhighs, large breasts, closed mouth, shy, blush, happy sex, lying on bed, breasts out, nipples, anal, sex, anus, doggystyle, panties around one leg, huge ass"

NEGATIVE_PROMPT = "worst quality, low quality, deformed, bad anatomy, out of frame, logo, watermark"

# Multi-stage Detailer Configuration (Simulating FaceDetailer node subgraph)
# Defines a list of refinement passes: (detector_model_path, denoise_strength, guide_size)
REFINEMENT_PASSES = [
    {"name": "face", "model": "generate/bbox/face_yolov8m.pt", "denoise": 0.24, "guide_size": 480},
    {"name": "hand", "model": "generate/bbox/hand_yolov8s.pt", "denoise": 0.16, "guide_size": 320},
    # {"name": "eyes", "model": "generate/bbox/Eyes.pt", "denoise": 0.20, "guide_size": 256},
    {"name": "breasts", "model": "generate/bbox/mosic.pt", "denoise": 0.18, "guide_size": 320},
    {"name": "breasts", "model": "generate/bbox/breasts_seg.pt", "denoise": 0.18, "guide_size": 512}
]

OUTPUT_FILENAME_PREFIX = "output/AXL"
REALESRGAN_MODEL_PATH = "/home/acite/LLM/Trainer/generate/upscale_models/RealESRGAN_x4plus_anime_6B.pth"


def resolve_seed(seed_override=None):
    """
    Resolve the effective seed for this run.

    Rules:
    - If an override is provided, use it exactly.
    - If no override is provided and config.SEED == 0, generate a random seed.
    - Otherwise use config.SEED.
    """
    if seed_override is not None:
        return int(seed_override)

    if SEED == 0:
        return int(torch.seed())

    return int(SEED)
