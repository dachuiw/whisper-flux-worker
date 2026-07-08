#!/bin/bash
# RunPod entrypoint with error logging
set -e
echo "[start] Starting Flux worker..."
echo "[start] Python: $(python3 --version)"
echo "[start] Torch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "[start] Diffusers: $(python3 -c 'import diffusers; print(diffusers.__version__)')"
echo "[start] Starting runpod serverless..."
exec python3 -u /handler.py
