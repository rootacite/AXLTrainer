# config.py
import torch

# Environment & Hardware Optimizations (ROCm-friendly)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# Model Paths (Change to your local absolute paths or HuggingFace repo IDs)
BASE_MODEL_PATH = "/tmp/waillu_170"
LORA_PATH = "/home/acite/LLM/models/loras/miyako.safetensors"
LORA_SCALE = 0.8

# Base Image Generation Parameters
WIDTH = 1280
HEIGHT = 720
STEPS = 60
CFG_SCALE = 6.0
SEED = 4408646534401889982

# Prompts from ComfyUI workflow nodes
# POSITIVE_PROMPT = "miyako_style, (large breasts:1.2), solo, looking at viewer, school uniform, off-shoulder, kneel-sitting, sitting on bed, open collar, cleavage, one nipple, (shy:1.2)"
POSITIVE_PROMPT = "miyako_style, newest, soft shading, (large breasts:1.2), white thighhighs, large breasts, closed mouth, shy, lying on bed, breasts out, nipples, anal sex, anus, doggystyle, panties around one leg, cum in ass, huge ass, huge penis, spread ass, ass grab"


NEGATIVE_PROMPT = "worst quality, low quality, deformed, bad anatomy, disfigured, 1boy, hetero, asymmetric clothes, out of frame, logo, watermark"

# Multi-stage Detailer Configuration (Simulating FaceDetailer node subgraph)
# Defines a list of refinement passes: (detector_model_path, denoise_strength, guide_size)
REFINEMENT_PASSES = [
    {"name": "face", "model": "generate/bbox/face_yolov8m.pt", "denoise": 0.18, "guide_size": 320},
    {"name": "hand", "model": "generate/bbox/hand_yolov8s.pt", "denoise": 0.16, "guide_size": 320},
    {"name": "eyes", "model": "generate/bbox/Eyes.pt", "denoise": 0.20, "guide_size": 256},
    # {"name": "breasts", "model": "generate/bbox/mosic.pt", "denoise": 0.18, "guide_size": 480},
    {"name": "breasts", "model": "generate/bbox/breasts_seg.pt", "denoise": 0.18, "guide_size": 512}
]

OUTPUT_FILENAME_PREFIX = "output/AXL"
REALESRGAN_MODEL_PATH = "/home/acite/LLM/Trainer/generate/upscale_models/RealESRGAN_x4plus_anime_6B.pth"