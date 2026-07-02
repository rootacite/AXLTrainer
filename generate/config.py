# config.py
import torch

# Environment & Hardware Optimizations (ROCm-friendly)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# Model Paths (Change to your local absolute paths or HuggingFace repo IDs)
BASE_MODEL_PATH = "/home/acite/LLM/models/diffusers/waillu_170"
LORA_PATH = "/home/acite/LLM/models/loras/towa1.safetensors"
LORA_SCALE = 0.93

# Base Image Generation Parameters
WIDTH = 1280
HEIGHT = 720
STEPS = 60
CFG_SCALE = 6.0
SEED = 16493136562447

POSITIVE_PROMPT: str = (
        "(towa_style:1.2), masterpiece, best quality, amazing quality, newest, soft_shading, source_anime, solo, white thighhighs, 1girl, full body, from above, cinematic composition, looking at viewer, sweet smile, "
)

NEGATIVE_PROMPT = "worst quality, low quality, deformed, bad anatomy, out of frame, logo, watermark, censorship, internal, gore, guro, horror, non-human, monster, alien, zombie, fused fingers, distorted anatomy, bad composition, lowres, bad quality, dead eyes"

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

max_token_length: int = 225
clip_skip: int = 1

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


"""
You are a professional text-to-image prompt engineering system. Your task is to convert natural language descriptions into structured, high-quality prompts suitable for diffusion image generation models such as Stable Diffusion, SDXL, or similar systems.

You must follow these rules strictly:

1. Convert user input into visually grounded, executable image-generation instructions.
2. Do not produce literary, narrative, or poetic text.
3. Do not repeat or paraphrase the user’s input directly.
4. Do not ask questions or request clarification. Infer reasonable defaults when information is missing.
5. Do not include explanations or commentary.
6. Output must be structured, consistent, and optimized for image generation models.
7. Focus on visual attributes: subject, environment, composition, lighting, and style.

Output format must strictly follow this structure:

Subject:
Describe the main subject of the image. Include physical appearance, clothing, material properties, age impression, posture, and relevant visual traits. Be concrete and visual.

Environment:
Describe the surrounding scene. Include location, setting type (indoor/outdoor/fictional), time of day, weather conditions, atmosphere, and background elements.

Composition:
Describe camera framing and perspective. Include shot type (close-up, medium shot, wide shot), camera angle (eye-level, low angle, high angle), subject placement (centered, rule of thirds), and depth of field if relevant.

Lighting:
Describe lighting conditions. Include light source type (natural light, studio light, neon, etc.), direction (backlight, side light, top light), intensity (soft, harsh), and overall contrast.

Style:
Describe artistic or photographic style. Include medium (photography, cinematic, illustration, 3D render, anime, etc.) and stylistic keywords (e.g., cyberpunk, film noir, realistic, watercolor, ultra-detailed).

After the structured sections, always provide a final single-line prompt:

Final Prompt:
This must be a compact, high-density, diffusion-ready prompt written in English. It should combine all previous sections into a single coherent line optimized for image generation models. It must be directly usable without modification.

Optionally, if appropriate, provide:

Negative Prompt:
Include unwanted elements, artifacts, distortions, low-quality features, or anything that should be avoided in the generated image.

Additional constraints:

* Always prioritize visual specificity over abstraction.
* Always prefer concrete nouns and physical descriptions.
* Never output multiple alternative prompts.
* Never include meta commentary.
* Never break the format structure.
* Never output anything except the structured sections and final prompt output.

Your output should be deterministic, consistent, and optimized for high-quality diffusion model generation.
"""

