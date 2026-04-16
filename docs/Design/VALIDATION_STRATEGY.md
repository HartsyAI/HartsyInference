# Validation Strategy

> Back to [Core Design](CORE_DESIGN.md)

---

## Principle

Every model implementation must be validated against a reference implementation before being considered done. "It generates an image" is not sufficient — the output must match a known-good reference within defined tolerances.

---

## Validation Matrix

| Component | Reference Implementation | Validation Method | Tolerance |
|---|---|---|---|
| **Safetensors loader** | Python `safetensors` library | Load same file in both, compare tensor values | Byte-for-byte match |
| **GGUF loader** | llama.cpp / ggml | Load same file, compare dequantized tensor values | Byte-for-byte match |
| **CLIP tokenizer** | OpenAI `clip` Python package | Tokenize 100 test strings, compare token ID arrays | Exact match |
| **T5 tokenizer** | HuggingFace `transformers` T5Tokenizer | Tokenize 100 test strings, compare token ID arrays | Exact match |
| **Whisper tokenizer** | OpenAI `whisper` Python package | Tokenize 100 test strings, compare token IDs | Exact match |
| **Whisper mel spectrogram** | `whisper.cpp` mel output | Load same audio, compare mel spectrogram | Within 1e-4 |
| **CPU MatMul kernel** | NumPy `np.matmul` | Random matrices, compare output | Within 1e-5 (FP32), 1e-3 (FP16) |
| **CPU Conv2D kernel** | PyTorch `F.conv2d` | Same input/weights/bias, compare output | Within 1e-5 (FP32) |
| **CPU GroupNorm** | PyTorch `F.group_norm` | Same input/params, compare output | Within 1e-5 (FP32) |
| **CUDA MatMul (cuBLAS)** | cuBLAS reference (C++) | Same matrices, compare output | Within 1e-3 (FP16) |
| **CUDA GroupNorm (PTX)** | PyTorch `F.group_norm` on CUDA | Same input/params, compare output | Within 1e-3 (FP16) |
| **CUDA Conv2D (cuDNN)** | PyTorch `F.conv2d` on CUDA | Same input/weights, compare output | Within 1e-3 (FP16) |
| **CUDA fused GroupNorm+SiLU** | Sequential GroupNorm then SiLU (CUDA) | Same input, compare fused vs sequential | Within 1e-3 (FP16) |
| **Vulkan MatMul (SPIR-V tiled)** | CUDA cuBLAS output | Same matrices on both backends, compare | Within 1e-3 (FP16) |
| **Vulkan GroupNorm (SPIR-V)** | CUDA PTX GroupNorm output | Same input/params on both backends | Within 1e-3 (FP16) |
| **Vulkan Conv2D (SPIR-V)** | CUDA cuDNN Conv2D output | Same input/weights on both backends | Within 1e-3 (FP16) |
| **Vulkan SDPA (SPIR-V)** | CUDA PTX SDPA output | Same Q/K/V on both backends | Within 1e-3 (FP16) |
| **Cross-backend SD1.5 pipeline** | CUDA pipeline output | Same seed/prompt on Vulkan vs CUDA | SSIM > 0.95 between backends |
| **Euler scheduler** | diffusers `EulerDiscreteScheduler` | Same noise → same 20-step denoised sequence | Within 1e-4 |
| **DPM++ 2M scheduler** | diffusers `DPMSolverMultistepScheduler` | Same noise → same step sequence | Within 1e-4 |
| **CLIP text encoder** | diffusers `CLIPTextModel` | Same tokens → same hidden states | Within 1e-3 (FP16) |
| **SD1.5 UNet forward** | diffusers `UNet2DConditionModel` | Same latent + timestep + conditioning → compare output | Within 1e-3 (FP16) |
| **VAE decoder** | diffusers `AutoencoderKL` | Same latent → compare decoded pixels | Within 1e-3 (FP16) |
| **Full SD1.5 pipeline** | Python `diffusers` with fixed seed | Same seed + prompt → visually identical output | Visual inspection + SSIM > 0.95 |
| **SDXL pipeline** | Python `diffusers` with fixed seed | Same seed + prompt → visually identical output | Visual inspection + SSIM > 0.95 |
| **Flux pipeline** | Python `diffusers` with fixed seed | Same seed + prompt → visually identical output | Visual inspection + SSIM > 0.95 |
| **Whisper transcription** | `whisper.cpp` | Same audio → same transcript | Word error rate < 1% |
| **Kokoro TTS** | Reference Python Kokoro | Same text → mel spectrogram comparison | Within tolerance |
| **CLIP image encoder** | OpenAI `clip` ViT forward | Same image → same embedding vector | Cosine similarity > 0.999 |
| **YOLO detection** | Ultralytics Python YOLOv8 | Same image → same bounding boxes | IoU > 0.95 for all detections |

---

## Validation Tooling

### Reference Output Generation

A Python script suite (`tests/reference/`) generates golden reference outputs for each component:

```
tests/reference/
├── generate_tokenizer_refs.py     # Tokenize test strings, save token ID arrays
├── generate_scheduler_refs.py     # Run schedulers with fixed seeds, save step sequences
├── generate_unet_refs.py          # UNet forward pass with fixed inputs, save output tensors
├── generate_vae_refs.py           # VAE decode with fixed latents, save output
├── generate_pipeline_refs.py      # Full pipeline with fixed seed, save final image
├── generate_whisper_refs.py       # Whisper encode/decode with fixed audio, save transcript
├── generate_mel_refs.py           # Mel spectrogram from fixed audio, save spectrogram
└── golden/                        # Saved reference outputs (committed to repo)
    ├── clip_tokens_100.json
    ├── euler_20step_seed42.npy
    ├── sd15_unet_forward.npy
    ├── sd15_pipeline_seed42.png
    └── ...
```

### Automated Comparison

Each test project includes comparison utilities:

- **TensorCompare** — element-wise comparison with configurable tolerance (absolute and relative)
- **ImageCompare** — SSIM and per-pixel difference with threshold
- **TextCompare** — exact string match or word error rate computation

### CI Integration

- **Unit tests** (kernel, tokenizer, scheduler) run on every PR — fast, no model download
- **Integration tests** (full pipeline) run on GPU CI — tagged `[Category("Integration")]`, require model files
- **Golden reference tests** compare against committed reference outputs — detect regressions
