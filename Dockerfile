# RunPod Serverless: Flux.1 Dev + LoRA + I2I worker
# Builds on RunPod's infrastructure; models pre-cached for fast cold start.
# Base: PyTorch 2.7.1 + CUDA 12.90 + Ubuntu 22.04

FROM runpod/pytorch:1.0.7-cu1290-torch271-ubuntu2204

ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

# Install Python dependencies
COPY requirements.txt /requirements.txt
RUN python3 -m pip install --no-cache-dir -r /requirements.txt && \
    python3 -c "from diffusers import FluxPipeline; print('✅ FluxPipeline imported OK')" && \
    python3 -c "from diffusers import FluxImg2ImgPipeline; print('✅ FluxImg2ImgPipeline imported OK')" && \
    python3 -c "import runpod; print('✅ runpod imported OK')"

# Pre-download Flux.1 Dev model (with LoRA) at build time
# This turns cold start from ~5min to ~2sec
RUN python3 -c "
import torch
from diffusers import FluxPipeline, FluxImg2ImgPipeline
import os

hf_token = os.environ.get('HF_TOKEN')
print('[build] Downloading Flux.1 Dev model...')
pipe = FluxPipeline.from_pretrained(
    'black-forest-labs/FLUX.1-dev',
    torch_dtype=torch.bfloat16,
    token=hf_token,
)
print('[build] Flux.1 Dev downloaded OK')

# Also pre-download LoRAs
try:
    pipe.load_lora_weights(
        'XLabs-AI/flux-dev-realism',
        weight_name='lora.safetensors',
        token=hf_token,
    )
    pipe.fuse_lora(lora_scale=0.5)
    print('[build] Realism LoRA downloaded OK')
except Exception as e:
    print(f'[build] Realism LoRA download failed (continuing): {e}')

try:
    pipe.load_lora_weights(
        'norod78/flux1-dev-detail-lora',
        weight_name='pytorch_lora_weights.safetensors',
        token=hf_token,
    )
    pipe.fuse_lora(lora_scale=0.3)
    print('[build] Detail LoRA downloaded OK')
except Exception as e:
    print(f'[build] Detail LoRA download failed (continuing): {e}')

del pipe
print('[build] All models pre-cached OK')
"

# Copy handler
COPY handler.py /handler.py
RUN python3 -c "import py_compile; py_compile.compile('/handler.py', doraise=True); print('✅ handler.py syntax OK')"

# Startup script that logs errors
COPY start.sh /start.sh
RUN chmod +x /start.sh

# RunPod Serverless entrypoint
CMD ["/start.sh"]
