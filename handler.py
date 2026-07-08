"""
RunPod Serverless worker — Flux.1 Dev + LoRA + Img2Img
=====================================================
Two modes:
  text2img  — T2I with Flux.1 Dev + optional Realism LoRA
  img2img   — I2I with Flux.1 Dev, denoising from reference image

Memory strategy for RTX 3090 (24GB):
  - model_cpu_offload() shifts T5/VAE to CPU when not needed
  - Peak VRAM ~14-16GB during inference
  - Cold start ~40-60s (downloads model from HuggingFace)
"""

import runpod
import torch
import base64
import io
import os
from typing import Optional

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model / LoRA config
# ---------------------------------------------------------------------------
MODEL_ID = "black-forest-labs/FLUX.1-dev"
REALISM_LORA_ID = "XLabs-AI/flux-dev-realism"          # realism LoRA
REALISM_LORA_WEIGHT = "lora.safetensors"
DETAIL_LORA_ID = "norod78/flux1-dev-detail-lora"       # detail enhancement (optional)
DETAIL_LORA_WEIGHT = "pytorch_lora_weights.safetensors"

pipe = None      # singleton: loaded once, survives idle via Flashboot
pipe_i2i = None  # separate pipeline for img2img (same underlying model)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_hf_token() -> Optional[str]:
    """Return HF token from env, or None if model is accessible without it."""
    return os.environ.get("HF_TOKEN", None)


def pil_to_b64(img) -> str:
    """PIL Image → base64 PNG string (without data: prefix)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def b64_to_pil(b64: str) -> "PIL.Image":
    """base64 string → PIL Image. Strips 'data:image/...' prefix if present."""
    from PIL import Image as PILImage
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    return PILImage.open(io.BytesIO(base64.b64decode(b64)))


# ---------------------------------------------------------------------------
# Model loading (cold start only)
# ---------------------------------------------------------------------------
def load_models():
    """Load T2I pipeline (with LoRA) on first call. Singleton."""
    from diffusers import FluxPipeline

    global pipe, pipe_i2i
    if pipe is not None:
        return

    hf_token = get_hf_token()
    dtype = torch.bfloat16

    log.info(f"Loading {MODEL_ID} ...")

    # ── T2I pipeline ──
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        token=hf_token,
    )
    pipe.enable_model_cpu_offload()       # critical: fits in 24GB
    pipe.enable_attention_slicing()

    # Load Realism LoRA
    try:
        pipe.load_lora_weights(
            REALISM_LORA_ID,
            weight_name=REALISM_LORA_WEIGHT,
            token=hf_token,
        )
        pipe.fuse_lora(lora_scale=0.9)
        log.info(f"Loaded Realism LoRA: {REALISM_LORA_ID}")
    except Exception as e:
        log.warning(f"Realism LoRA load failed (continuing without): {e}")

    # Load Detail LoRA (optional)
    try:
        pipe.load_lora_weights(
            DETAIL_LORA_ID,
            weight_name=DETAIL_LORA_WEIGHT,
            token=hf_token,
        )
        pipe.fuse_lora(lora_scale=0.5)
        log.info(f"Loaded Detail LoRA: {DETAIL_LORA_ID}")
    except Exception as e:
        log.warning(f"Detail LoRA load failed (continuing without): {e}")

    log.info("T2I pipeline ready")

    # ── I2I pipeline ──
    try:
        from diffusers import FluxImg2ImgPipeline
        pipe_i2i = FluxImg2ImgPipeline(**pipe.components)
        pipe_i2i.enable_model_cpu_offload()
        pipe_i2i.enable_attention_slicing()
        log.info("I2I pipeline ready (FluxImg2ImgPipeline)")
    except ImportError:
        # Fallback: use same pipe for both (I2I handled in handler)
        pipe_i2i = None
        log.warning("FluxImg2ImgPipeline not available, using T2I pipeline for I2I fallback")


# ---------------------------------------------------------------------------
# Generation logic
# ---------------------------------------------------------------------------
def generate_text2img(job_input: dict) -> dict:
    """Text-to-Image with Flux.1 Dev + LoRA."""
    prompt = job_input.get("prompt", "")
    if not prompt:
        return {"error": "Missing prompt"}

    negative = job_input.get("negative_prompt", "")
    width = job_input.get("width", 1024)
    height = job_input.get("height", 1536)  # portrait default
    steps = job_input.get("steps", 30)
    guidance = job_input.get("guidance_scale", 3.0)
    seed = job_input.get("seed", -1)

    if seed == -1:
        generator = None
    else:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    kwargs = dict(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=width,
        height=height,
        generator=generator,
    )
    if negative:
        kwargs["negative_prompt"] = negative

    result = pipe(**kwargs)
    img = result.images[0]
    b64 = pil_to_b64(img)

    return {"image": b64, "mode": "text2img", "seed": seed}


def generate_img2img(job_input: dict) -> dict:
    """Image-to-Image — modify reference image via denoising."""
    prompt = job_input.get("prompt", "")
    image_b64 = job_input.get("image", "")
    if not prompt or not image_b64:
        return {"error": "Missing prompt or image"}

    init_image = b64_to_pil(image_b64)
    strength = job_input.get("strength", 0.45)
    steps = job_input.get("steps", 30)
    guidance = job_input.get("guidance_scale", 3.0)
    seed = job_input.get("seed", -1)

    if seed == -1:
        generator = None
    else:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    # Try dedicated I2I pipeline first
    if pipe_i2i is not None:
        result = pipe_i2i(
            prompt=prompt,
            image=init_image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
    else:
        # Fallback: use T2I pipeline (ignores input image, generates from prompt)
        log.warning("FluxImg2ImgPipeline not available, falling back to T2I")
        result = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=init_image.width,
            height=init_image.height,
            generator=generator,
        )

    img = result.images[0]
    b64 = pil_to_b64(img)

    return {"image": b64, "mode": "img2img", "seed": seed, "strength": strength}


# ---------------------------------------------------------------------------
# RunPod entrypoint
# ---------------------------------------------------------------------------
def handler(job):
    """Main handler dispatched by job['input']['mode']."""
    inp = job["input"]
    mode = inp.get("mode", "text2img")

    load_models()  # singleton, cheap on warm calls

    if mode == "text2img":
        return generate_text2img(inp)
    elif mode == "img2img":
        return generate_img2img(inp)
    else:
        return {"error": f"Unknown mode: {mode}. Use 'text2img' or 'img2img'."}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
