# RunPod Serverless: Flux.1 Dev + LoRA + I2I worker
# Builds on RunPod's infrastructure; models download at cold start.
# Base: PyTorch 2.7.1 + CUDA 12.90 + Ubuntu 22.04

FROM runpod/pytorch:1.0.7-cu1290-torch271-ubuntu2204

# Install Python dependencies
COPY requirements.txt /requirements.txt
RUN python3 -m pip install --no-cache-dir -r /requirements.txt && \
    python3 -c "from diffusers import FluxPipeline; print('✅ FluxPipeline imported OK')" && \
    python3 -c "from diffusers import FluxImg2ImgPipeline; print('✅ FluxImg2ImgPipeline imported OK')" && \
    python3 -c "import runpod; print('✅ runpod imported OK')"

# Copy handler
COPY handler.py /handler.py
RUN python3 -c "import py_compile; py_compile.compile('/handler.py', doraise=True); print('✅ handler.py syntax OK')"

# Startup script that logs errors
COPY start.sh /start.sh
RUN chmod +x /start.sh

# RunPod Serverless entrypoint
CMD ["/start.sh"]
