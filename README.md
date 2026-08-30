# HartsyInference

**Run Krea 2, MiniMax-H3, Flux, Qwen-Image, Wan, Whisper, Llama and 100+ other models from C#.
No Python. No ComfyUI. No venv.**

HartsyInference is a complete AI inference engine written entirely in C#. It loads `.safetensors`,
`.gguf` and PyTorch checkpoints directly and runs them on **CUDA**, **Vulkan** or **CPU** — as NuGet
packages you reference from a .NET app. There is no Python interpreter, no `pip`, no subprocess, no GIL,
and nothing to install beyond your GPU driver.

It is not a wrapper. The CUDA kernels are hand-written PTX loaded through the Driver API, the Vulkan
kernels are SPIR-V, the CPU kernels are AVX2/AVX-512/NEON intrinsics. One engine spans image, video,
audio, LLM, vision, 3D and interactive world models.

---

## The newest models, running fastest

### Krea 2 — faster than ComfyUI, both variants

| RTX 4090 | HartsyInference | ComfyUI | |
|---|---:|---:|---|
| **Krea 2 Turbo**, 8 steps | **4.5 s** | 6.5 s | **1.4× faster** |
| **Krea 2 Base**, 28 steps | **30.3 s** | 41.5 s | **1.4× faster** |

### MiniMax-H3 — omni video *with* jointly generated stereo audio

Text, image, video **and** audio in, video with a synchronized stereo soundtrack out. Ported and verified
end-to-end within days of the weights landing.

- **1.671 s/step** at 141 frames @ 512×288, 30 steps, fp8 on an RTX 4090 — against ComfyUI's **1.660
  s/step** measured interleaved in the same session. That is 0.7% behind with overlapping ranges: parity
  within measurement resolution, and we won't claim more than that.
- **Runs on a 12 GB RTX 3060.** The 66 GB bf16 DiT is memory-mapped, so it loads at 943 MB RSS and the
  whole generation fits in 10.3 GB.

Method and the full A/B ladder: [`benchmarks/results/h3/PHASE0_BASELINE.md`](benchmarks/results/h3/PHASE0_BASELINE.md).

### And the rest of the fleet

Same GPU, same checkpoint, same scheduler, same step count. Full tables with methodology, dates and
sources: **[`benchmarks/scoreboards/`](benchmarks/scoreboards/)**.

| Image model (RTX 4090) | HartsyInference | ComfyUI | |
|---|---:|---:|---|
| Flux-Schnell, 4 steps | **2.4 s** | 3.8 s | **1.6× faster** |
| Qwen-Image 20B, 20 steps | **40.6 s** | 58.2 s | **1.4× faster** |
| Flux.2-Dev 32B (Q4 GGUF), 20 steps | **39.6 s** | 54.0 s | **1.4× faster** |
| Flux-Dev, 20 steps | **9.5 s** | 12.5 s | **1.3× faster** |
| Chroma1-HD, 30 steps | **24.6 s** | 32.6 s | **1.3× faster** |

**Text generation beats llama.cpp too** — same GGUF, same quant, on a 12 GB RTX 3060: Qwen2.5-0.5B
**435.6** vs 328.9 tok/s (1.32×), Gemma-3-1B **251.3** vs 196.8 (1.28×), Llama-3.2-1B **213.7** vs 192.0
(1.11×). Thirteen models in the core fleet are ahead; GLM-4-9B is effective parity at 0.98×.

**The losses are published in the same tables.** SDXL (1.45× slower), Lumina-Image 2.0 (1.76×),
Boogu-Base (1.49×) and Z-Image-Turbo (1.35×) still trail ComfyUI, and most video DiTs trail by 1.1–8× —
video is the current optimization frontier. Every number above is the **shipped default configuration**;
no opt-in step-caching or CFG-interval tricks are folded in to flatter the result.

---

## It runs models your GPU cannot hold

Weights are memory-mapped, so a checkpoint far larger than VRAM still runs — that is how MiniMax-H3's
**66 GB** DiT fits on a **12 GB** RTX 3060, as above.

**Or split one model across several GPUs with the VRAM pooled.** Over plain PCIe — **no NVLink and no P2P
required** — because mismatched consumer cards are the primary target, not datacenter boxes. The dev rig
is a 4090 next to a 3060.

| | What it does | Measured |
|---|---|---|
| **LLM layer split** | Text models bigger than any one card | Qwen3-32B Q4_K_M OOMs a 24 GB 4090 alone → **~12 tok/s** across 4090+3060 |
| **DiT block sharding** | Big diffusion models fully resident instead of streamed | Qwen-Image 20B at **13.4 + 6.2 GB**, SSIM 0.9734 vs single-GPU |
| **Audio-LM split** | Music LMs run **un-quantized** instead of crushed to fit | YuE 7B Stage-1 at full bf16 in **8.7 + 4.3 GB** |
| **Component placement** | Text encoder / VAE off the denoiser's card | Wan TI2V-5B **43.7 s → 32.7 s** |
| **CFG-parallel** | Both prompt branches run concurrently | **~1.8–1.9×** per-step on Wan / Flux true-CFG |

Honest framing: **sharding pools VRAM, it does not add speed.** A pipeline split runs its stages
sequentially, so per-step time is the same or a few percent worse. The win is that models that *couldn't
run* now run, and models that had to be quantized now run at full precision. Placement and CFG-parallel
are the two modes that are outright faster.

Full guide: **[`docs/MULTI_GPU.md`](docs/MULTI_GPU.md)**.

---

## One engine, every modality

| | Models |
|---|---|
| **Image** | **Krea 2** (Turbo + Base), Flux.1/.2, Qwen-Image (+Edit), Z-Image, Chroma, SD1.5, SDXL, SD3.5, HiDream, AuraFlow, Lumina 2, ERNIE, Kandinsky 5, OmniGen 2, Ideogram 4 |
| **Video** | **MiniMax-H3** (video + native stereo audio), Wan 2.1/2.2, HunyuanVideo 13B, LTX-Video + LTX-2.3/2.5, Kandinsky 5, Lance — plus **SeedVR2** video/image restoration |
| **LLM** | Llama, Qwen2/3, Gemma 2/3/4, Phi, Mistral, MoE giants (Mixtral, DeepSeek-V3, Kimi-K2, GPT-OSS), Mamba/RWKV, VLMs, embeddings and rerankers — GGUF quantized throughout |
| **Audio** | Whisper + Moonshine (STT); Kokoro, Piper, StyleTTS2, F5-TTS, CosyVoice, VibeVoice, Spark-TTS, Bark (TTS); ACE-Step, MusicGen, YuE, HeartMuLa (music); 9 neural codecs; voice conversion, stem separation, speech enhancement |
| **Vision** | CLIP / SigLIP / DINOv2 embeddings, YOLOv8/11, RT-DETR, Grounding DINO, SAM 2, Depth-Anything-V2, face detection |
| **3D** | TripoSR, Hunyuan3D-2 — image → mesh with glTF/OBJ/PLY export |
| **World** | Oasis, DIAMOND — action-conditioned, real-time interactive frame generation |

Image conditioning is first-class: **ControlNet** across SDXL/SD1.5/Flux with union types, multi-net
stacking and in-engine canny / depth / openpose / lineart / softedge / normal / segmentation
preprocessors; **IP-Adapter** including FaceID and FaceID-PlusV2; and instruction-edit models (Flux
Kontext, Qwen-Image-Edit, OmniGen2, Boogu-Edit).

Per-model status — verified end-to-end vs built-and-pending, with parity evidence — is in
[`docs/Checklists/MODEL_STATUS.md`](docs/Checklists/MODEL_STATUS.md).

---

## Get started

### With a UI (recommended)

Install the **[SwarmUI backend extension](https://github.com/HartsyAI/SwarmUI-HartsyInference-Backend)**
to register HartsyInference as a pure-C# alternative to the ComfyUI backend. You get SwarmUI's full
generation UI, model browser and parameter controls with **no Python environment at all**. Clone it into
SwarmUI's `src/Extensions/`, rebuild, and add a HartsyInference backend under Server → Backends.

### In your own app

```bash
dotnet add package HartsyInference             # everything
dotnet add package HartsyInference.Diffusion   # or just the modality you need
```

`InferenceEngine` is the facade — it owns model loading, caching and generation for every modality, so
you never touch a pipeline directly unless you want to.

```csharp
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

using var engine = new InferenceEngine("cuda");

ImageResult image = await engine.Images.GenerateAsync(
    new ModelSpec { Requested = "sdxl", Modality = Modality.Image, LocalPath = "sdxl.safetensors" },
    new ImageRequest { Prompt = "a castle on a mountain at sunset, oil painting", Steps = 25 });

// image.Rgb is RGB24, image.Width × image.Height
```

Same shape for everything else — `engine.Text.StreamAsync(...)` streams tokens,
`engine.Transcribe.RunAsync(...)` returns a transcript, `engine.Video`, `engine.Speech`, `engine.Music`,
`engine.Vision`, `engine.Mesh`, `engine.World` and `engine.Embeddings` follow the same pattern.

### From the terminal

```bash
dotnet run -c Release --project src/HartsyInference.Cli -- \
  image "a castle on a mountain at sunset" --model-path sdxl.safetensors -b cuda
```

The `hartsy` CLI covers every modality — `text`, `image`, `video`, `transcribe`, `speak`, `music`,
`vision`, `3d`, `world`, `restore`, `convert`, `fx` — plus `pull` / `list` / `models` for the catalog. Run
it with no arguments for an interactive REPL. It is a developer and validation tool, not the intended
end-user surface.

### As an HTTP server

`src/HartsyInference.API` hosts an OpenAI-shaped REST API: `/v1/chat/completions` (streaming, JSON-mode,
with **continuous batching** and a **paged KV cache**), `/v1/images/generations`, and model
load/list/pull/unload. Runs from source; not published as a package.

Native video has a separate preflight contract. `POST /v1/native/video/plan` resolves the checkpoint
profile, effective geometry/sampling values, component formats, and every compatibility issue without
constructing model weights. `POST /v1/native/video/stream` runs that same plan and returns named SSE
events (`progress`, repeated `frame`, optional `audio`, then successful terminal `complete`); a generation failure
after streaming begins emits terminal `error`. For MiniMax-H3, `complete` includes the exact profile-resolved
`VideoExecutionSummary` (legacy families leave it null until their pipeline-specific normalization is exposed).
Unsafe profile/adapter/control combinations return a typed HTTP 422 before
SSE begins. Base64 guide and control clips are bounded by `HartsyInference:MaxVideoRequestBodyBytes`
(256 MiB by default).

---

## Requirements

**.NET 8 or .NET 10**, then pick a backend:

- **CUDA** (fastest) — NVIDIA GPU with compute capability 8.0+ (RTX 30xx/40xx, A100, H100) and CUDA 12.x
  or 13.x userspace libraries. FP8 tensor-core paths need 8.9+ (Ada). cuDNN 9.21+ is optional and enables
  fused flash attention; without it the engine falls back to materialized attention — slower, identical
  output.
- **Vulkan** (cross-vendor — NVIDIA / AMD / Intel) — a Vulkan 1.3+ runtime, which your GPU driver almost
  certainly installed already, plus FP16 compute (`shaderFloat16`). Most discrete GPUs from 2019+ qualify.
- **CPU** (anywhere) — no GPU needed. AVX2 / AVX-512 / NEON with a scalar fallback. Slow, but universal.

---

## Packages

`HartsyInference` is a meta-package pulling in the core, all three backends and every modality. Reference
the individual packages instead if you only need one:

`Core` · `ModelAssets` (+ `.Tokenizers`) · `Cpu` · `Cuda` · `Vulkan` · `LLM` · `Diffusion` · `Audio`
(+ `.Phonemizer`) · `Vision` · `Video` · `ThreeD` · `World`

Dependencies flow one way — `Core` ← modality packages ← `Engine` ← consumers — and GPU code stays behind
`IBackend`, so a CPU-only package never drags in CUDA or Vulkan. `HartsyInference.Cli` and
`HartsyInference.API` are applications, run from source.

---

## How it's built

- **Pure C#** — no native shared libraries, no C++ wrappers, no ONNX Runtime, no managed GPU wrappers
- **CUDA via PTX** — kernels loaded from disk and JIT-compiled through the CUDA Driver API by P/Invoke
- **Eager execution** — no computation graphs; ops run immediately, so memory and debugging stay predictable
- **Zero-allocation hot paths** — tensors in `NativeMemory.AlignedAlloc`, weights memory-mapped, `Span<T>`
  throughout, no GC pressure during inference
- **Validated against references** — every component is diffed against its Python/C++ original within
  documented tolerances before it ships. Evidence:
  [`PARITY_VERIFICATION.md`](docs/Checklists/PARITY_VERIFICATION.md)

---

## Status

**Alpha.** The coverage and numbers above are real and verified, but this is a young engine: public APIs
still change between releases, some models are built but not yet verified end-to-end, and video
performance is the current optimization frontier. What works and what doesn't is written down rather than
glossed over — per-model status in [`MODEL_STATUS.md`](docs/Checklists/MODEL_STATUS.md), open engineering
work in [`ROADMAP.md`](docs/Checklists/ROADMAP.md), measured numbers in
[`benchmarks/scoreboards/`](benchmarks/scoreboards/).

## Documentation

[`docs/README.md`](docs/README.md) is the map. Entry points:
[**Multi-GPU guide**](docs/MULTI_GPU.md) ·
[**Model status**](docs/Checklists/MODEL_STATUS.md) ·
[**Benchmarks**](benchmarks/scoreboards/) ·
[**Parity evidence**](docs/Checklists/PARITY_VERIFICATION.md) ·
[**Troubleshooting**](docs/Checklists/TROUBLESHOOTING.md) ·
[**Code style**](docs/CODE_STYLE.md) ·
[**Contributing**](CONTRIBUTING.md)

## License

HartsyInference's source code and NuGet packages are [MIT licensed](LICENSE). Model weights are separate
works under their publishers' terms. In particular, MiniMax-H3 is distributed under the
[MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE), whose
applicable territory excludes the EU, UK, South Korea, and USA absent separate authorization and which
includes notice, disclosure, acceptable-use, commercial-display, and revenue-related obligations.
HartsyInference downloads or redistributes no H3 expansion weights; using local files does not remove the
operator's licensing obligations.
