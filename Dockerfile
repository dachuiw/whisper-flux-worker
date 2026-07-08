# RunPod Serverless: Flux.1 Dev + LoRA + I2I worker
# Builds on RunPod's infrastructure; models download at cold start.
# Base: PyTorch 2.7.1 + CUDA 12.90 + Ubuntu 22.04

FROM runpod/pytorch:1.0.7-cu1290-torch271-ubuntu2204

# Install Python dependencies
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Copy handler
COPY handler.py /handler.py

# RunPod Serverless entrypoint
CMD ["python", "-u", "/handler.py"]
