# RunPod Serverless: Flux.1 Dev + LoRA + I2I worker
# Base: PyTorch 2.7.1 + CUDA 12.90 + Ubuntu 22.04
# Model downloads at cold start via runtime HF_TOKEN env var

FROM runpod/pytorch:2.2.1-cuda12.1.0-ubuntu22.04

# Install Python dependencies
COPY requirements.txt /requirements.txt
RUN python3 -m pip install --no-cache-dir -r /requirements.txt && \
    python3 -c "from diffusers import FluxPipeline; print('import OK')" && \
    python3 -c "from diffusers import FluxImg2ImgPipeline; print('import OK')" && \
    python3 -c "import runpod; print('import OK')"

# Copy handler + start script
COPY handler.py /handler.py
COPY start.sh /start.sh
RUN chmod +x /start.sh

RUN python3 -c "import py_compile; py_compile.compile('/handler.py', doraise=True); print('handler.py syntax OK')"

CMD ["/start.sh"]
