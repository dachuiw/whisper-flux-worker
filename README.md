# Whisper Flux Worker — RunPod Serverless

Flux.1 Dev + Realism LoRA + Img2Img 的 RunPod Serverless worker。

## 两种模式

### text2img — 文生图基础生成
```json
{
  "input": {
    "mode": "text2img",
    "prompt": "Photorealistic full-body portrait of...",
    "width": 1024,
    "height": 1536,
    "steps": 30,
    "guidance_scale": 3.0,
    "seed": -1
  }
}
```

### img2img — 图生图精调暴露度/姿势
```json
{
  "input": {
    "mode": "img2img",
    "prompt": "same woman, much more sexy and revealing outfit...",
    "image": "<base64 PNG string>",
    "strength": 0.45,
    "steps": 30,
    "guidance_scale": 3.0
  }
}
```

## 模型

- **主模型**: black-forest-labs/FLUX.1-dev (12B, bfloat16)
- **LoRA 1**: XLabs-AI/flux-dev-realism (权重 0.9)
- **LoRA 2**: norod78/flux1-dev-detail-lora (权重 0.5, 可选)

## VRAM 策略

RTX 3090 (24GB) — `model_cpu_offload()` 将 T5/VAE 卸载到 CPU，峰值 ~14-16GB。
冷启动需要下载模型 (~40-60s)，Flashboot 后续调用热启动。

## 环境变量

- `HF_TOKEN` — HuggingFace token（FLUX.1-dev 需要接受授权）
- `RUNPOD_WEBHOOK_GET_JOB` — RunPod 自动注入
