"""
Pre-download Flux.1 Dev model + LoRAs at Docker build time.
Saves ~5min cold start on every new worker.
"""
import torch
from diffusers import FluxPipeline, FluxImg2ImgPipeline
import os

hf_token = os.environ.get("HF_TOKEN")
print("[build] Downloading Flux.1 Dev model...")

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
    token=hf_token,
)
print("[build] Flux.1 Dev downloaded OK")

# Pre-download LoRAs
try:
    pipe.load_lora_weights(
        "XLabs-AI/flux-dev-realism",
        weight_name="lora.safetensors",
        token=hf_token,
    )
    pipe.fuse_lora(lora_scale=0.5)
    print("[build] Realism LoRA downloaded OK")
except Exception as e:
    print(f"[build] Realism LoRA download failed (continuing): {e}")

try:
    pipe.load_lora_weights(
        "norod78/flux1-dev-detail-lora",
        weight_name="pytorch_lora_weights.safetensors",
        token=hf_token,
    )
    pipe.fuse_lora(lora_scale=0.3)
    print("[build] Detail LoRA downloaded OK")
except Exception as e:
    print(f"[build] Detail LoRA download failed (continuing): {e}")

del pipe
print("[build] All models pre-cached OK")
