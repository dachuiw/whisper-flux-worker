# RunPod Serverless: Flux.1 Dev + LoRA + I2I worker
# Builds on RunPod's infrastructure; models download at cold start.

FROM runpod/base:0.7.0-cuda12.4.1

# Install Python dependencies
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Copy handler
COPY handler.py /handler.py

# RunPod Serverless entrypoint
CMD ["python", "-u", "/handler.py"]
