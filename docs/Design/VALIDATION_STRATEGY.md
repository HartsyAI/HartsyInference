# Validation Strategy

> Back to [Core Design](CORE_DESIGN.md)

## Principle
Every model implementation must be validated against a reference before being considered done. "It generates an image" is not sufficient — output must match a known-good reference within defined tolerances.

## Validation Matrix

| Component | Reference | Method | Tolerance |
|---|---|---|---|
| Safetensors loader | Python `safetensors` | Compare tensor values | Byte-for-byte |
| GGUF loader | llama.cpp | Compare dequantized values | Byte-for-byte |
| CLIP/T5/Whisper tokenizer | OpenAI `clip` / HF T5 / OpenAI `whisper` | Tokenize 100 strings, compare IDs | Exact |
| Whisper mel spectrogram | `whisper.cpp` | Same audio, compare mels | 1e-4 |
| CPU MatMul | NumPy `np.matmul` | Random matrices | 1e-5 (FP32), 1e-3 (FP16) |
| CPU Conv2D/GroupNorm | PyTorch `F.conv2d` / `F.group_norm` | Same inputs | 1e-5 (FP32) |
| CUDA MatMul (cuBLAS) | cuBLAS C++ | Same matrices | 1e-3 (FP16) |
| CUDA GroupNorm/Conv2D (PTX) | PyTorch CUDA | Same inputs | 1e-3 (FP16) |
| CUDA fused GroupNorm+SiLU | Sequential unfused (CUDA) | Same input | 1e-3 (FP16) |
| Vulkan kernels (SPIR-V) | CUDA equivalent | Same inputs on both backends | 1e-3 (FP16) |
| Cross-backend SD1.5 | CUDA pipeline output | Same seed/prompt Vulkan vs CUDA | SSIM > 0.95 |
| Euler/DPM++ scheduler | diffusers | Same noise → step sequence | 1e-4 |
| CLIP text encoder | diffusers `CLIPTextModel` | Same tokens → hidden states | 1e-3 (FP16) |
| SD1.5 UNet/VAE | diffusers | Same inputs → outputs | 1e-3 (FP16) |
| Full pipelines (SD1.5/SDXL/Flux) | Python `diffusers` | Same seed + prompt | Visual + SSIM > 0.95 |
| Whisper transcription | `whisper.cpp` | Same audio | WER < 1% |
| Kokoro TTS | Python Kokoro | Same text → mel | Within tolerance |
| CLIP image encoder | OpenAI `clip` | Same image → embedding | Cosine similarity > 0.999 |
| YOLO detection | Ultralytics YOLOv8 | Same image → boxes | IoU > 0.95 |
| LLM logits (per family) | HF `transformers` / llama.cpp | Same tokens → logits (greedy, fixed seed) | High correlation; greedy token match |
| GGUF dequant-matmul | llama.cpp | Same quantized weights → output | 1e-3 (dequant), token-checked decode |
| Chat template | HF `apply_chat_template` | Same messages → prompt string | Exact |

## Reference Output Generation

Python scripts in `tests/reference/` generate golden outputs committed to repo:
```
generate_tokenizer_refs.py     generate_scheduler_refs.py
generate_unet_refs.py          generate_vae_refs.py
generate_pipeline_refs.py      generate_whisper_refs.py
generate_mel_refs.py
golden/clip_tokens_100.json    golden/euler_20step_seed42.npy
golden/sd15_unet_forward.npy   golden/sd15_pipeline_seed42.png
```

## Comparison Utilities
- **TensorCompare** — element-wise abs/rel tolerance
- **ImageCompare** — SSIM + per-pixel threshold
- **TextCompare** — exact match or WER

## CI Integration
- **Unit tests** — every PR, fast, no model download
- **Integration tests** — GPU CI, `[Category("Integration")]`, require model files
- **Golden reference tests** — detect regressions vs committed refs
