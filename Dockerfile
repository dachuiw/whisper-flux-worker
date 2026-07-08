# RunPod Serverless: Flux.1 Dev + LoRA + I2I worker
# Builds on RunPod's infrastructure; models pre-cached for fast cold start.
# Base: PyTorch 2.7.1 + CUDA 12.90 + Ubuntu 22.04

FROM runpod/pytorch:1.0.7-cu1290-torch271-ubuntu2204

ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

# Install Python dependencies
COPY requirements.txt /requirements.txt
RUN python3 -m pip install --no-cache-dir -r /requirements.txt && \
    python3 -c "from diffusers import FluxPipeline; print('import OK')" && \
    python3 -c "from diffusers import FluxImg2ImgPipeline; print('import OK')" && \
    python3 -c "import runpod; print('import OK')"

# Pre-download Flux.1 Dev model + LoRAs at build time
COPY download-models.py /download-models.py
RUN python3 -u /download-models.py

# Copy handler + start script
COPY handler.py /handler.py
COPY start.sh /start.sh
RUN chmod +x /start.sh

RUN python3 -c "import py_compile; py_compile.compile('/handler.py', doraise=True); print('handler.py syntax OK')"

CMD ["/start.sh"]
