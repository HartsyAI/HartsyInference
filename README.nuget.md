# HartsyInference

**Run Flux, Qwen-Image, Wan, Whisper, Llama and 100+ other models from C#. No Python. No venv. No subprocess.**

A complete AI inference engine written entirely in C#. It loads `.safetensors`, `.gguf` and PyTorch
checkpoints directly and runs them on **CUDA**, **Vulkan** or **CPU** — as NuGet packages you reference
from a .NET app. Nothing to install beyond your GPU driver.

Not a wrapper: the CUDA kernels are hand-written PTX loaded through the Driver API, the Vulkan kernels are
SPIR-V, the CPU kernels are AVX2/AVX-512/NEON intrinsics.

> ## ⚠️ Alpha
>
> Verified across a wide model set and in daily use, but **public APIs still change between releases**.
> Pin an exact version. Some models are built but not yet verified end-to-end, and video performance is
> still being optimized. Status is tracked openly — see the links at the bottom.

## Install

```bash
dotnet add package HartsyInference             # everything
dotnet add package HartsyInference.Diffusion   # or just the modality you need
```

## Generate an image

`InferenceEngine` is the facade — it owns model loading, caching and generation for every modality.

```csharp
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

using var engine = new InferenceEngine("cuda");   // "cpu", "vulkan", "cuda:1", "auto"

ImageResult image = await engine.Images.GenerateAsync(
    new ModelSpec { Requested = "sdxl", Modality = Modality.Image, LocalPath = "sdxl.safetensors" },
    new ImageRequest { Prompt = "a castle on a mountain at sunset, oil painting", Steps = 25 });

// image.Rgb is RGB24 bytes, image.Width × image.Height
```

## Stream tokens from a local LLM

```csharp
var spec = new ModelSpec { Requested = "qwen3", Modality = Modality.Text, LocalPath = "Qwen3-4B-Q4_K_M.gguf" };
var request = new TextRequest
{
    Messages = [new TextMessage { Role = TextRole.User, Content = "Explain flow matching." }],
};

await foreach (TextChunk chunk in engine.Text.StreamAsync(spec, request))
    if (chunk.Kind == TextChunkKind.Chunk)
        Console.Write(chunk.Text);
```

Every modality follows the same shape: `engine.Images`, `engine.Text`, `engine.Video`, `engine.Speech`,
`engine.Transcribe`, `engine.Music`, `engine.Vision`, `engine.Mesh`, `engine.World`, `engine.Restore`,
`engine.VoiceConversion`, `engine.Fx`, `engine.Embeddings`.

## Performance

Faster than ComfyUI on the models most people run — same GPU, same checkpoint, same steps (RTX 4090):
Flux-Schnell **2.4 s** vs 3.8 s, Krea2-Turbo **4.5 s** vs 6.5 s, Qwen-Image 20B **40.6 s** vs 58.2 s,
Flux-Dev **9.5 s** vs 12.5 s. SDXL and Lumina 2 still trail; most video DiTs still trail.

Text decode beats llama.cpp at matched GGUF and quant on a 12 GB RTX 3060: Qwen2.5-0.5B **435.6** vs
328.9 tok/s, Gemma-3-1B **251.3** vs 196.8, Llama-3.2-1B **213.7** vs 192.0.

Full tables, including the losses, are in the repo under `benchmarks/scoreboards/`.

## What it can run

- **Image** — SD1.5, SDXL, Flux.1/.2, SD3.5, Qwen-Image (+Edit), Chroma, Krea 2, Z-Image, HiDream,
  AuraFlow, Lumina 2, Kandinsky 5, OmniGen 2, Ideogram 4. ControlNet, IP-Adapter (incl. FaceID), LoRA,
  img2img and inpaint.
- **Video** — Wan 2.1/2.2, HunyuanVideo 13B, LTX-Video + LTX-2.3, MiniMax-H3, Kandinsky 5, plus SeedVR2
  video/image restoration.
- **LLM** — Llama, Qwen2/3, Gemma 2/3/4, Phi, Mistral, MoE giants, Mamba/RWKV, VLMs, embeddings and
  rerankers. GGUF quantized throughout, device-resident KV cache, chat templates.
- **Audio** — Whisper and Moonshine (STT); Kokoro, Piper, StyleTTS2, F5-TTS, CosyVoice, VibeVoice, Bark
  (TTS); ACE-Step, MusicGen, YuE (music); 9 neural codecs; voice conversion, stem separation, enhancement.
- **Vision** — CLIP / SigLIP / DINOv2, YOLOv8/11, RT-DETR, Grounding DINO, SAM 2, Depth-Anything-V2.
- **3D** — TripoSR, Hunyuan3D-2 (image → mesh, glTF/OBJ/PLY).
- **World** — Oasis, DIAMOND (action-conditioned real-time frame generation).

## Beyond one GPU

Memory-mapped weights run models far larger than VRAM — MiniMax-H3's **66 GB** bf16 DiT is verified
end-to-end on a **12 GB RTX 3060**, loading at 943 MB RSS. One model can also be **split across several
GPUs with the VRAM pooled**, over plain PCIe with no NVLink or P2P required: LLM layer splits, DiT block
sharding, text-encoder/VAE placement, and CFG-parallel branches.

## Packages

`HartsyInference` is a meta-package pulling in the core, all three backends and every modality. Reference
individually if you only need one:

`Core` · `ModelAssets` (+ `.Tokenizers`) · `Cpu` · `Cuda` · `Vulkan` · `LLM` · `Diffusion` · `Audio`
(+ `.Phonemizer`) · `Vision` · `Video` · `ThreeD` · `World`

Dependencies flow one way and GPU code stays behind `IBackend`, so a CPU-only package never drags in CUDA
or Vulkan.

## Requirements

**.NET 8 or .NET 10.** For CUDA: an NVIDIA GPU with compute capability 8.0+ and CUDA 12.x/13.x userspace
libraries (FP8 paths need 8.9+; cuDNN 9.21+ optional for fused flash attention). For Vulkan: a Vulkan 1.3+
runtime and FP16 compute — most discrete GPUs from 2019+. The CPU backend needs nothing.

## Links

[GitHub](https://github.com/HartsyAI/HartsyInference) ·
[SwarmUI backend extension](https://github.com/HartsyAI/SwarmUI-HartsyInference-Backend) ·
[Model status](https://github.com/HartsyAI/HartsyInference/blob/main/docs/Checklists/MODEL_STATUS.md) ·
[Benchmarks](https://github.com/HartsyAI/HartsyInference/tree/main/benchmarks/scoreboards) ·
[Roadmap](https://github.com/HartsyAI/HartsyInference/blob/main/docs/Checklists/ROADMAP.md)

Prefer a UI? The **SwarmUI backend extension** registers HartsyInference as a pure-C# alternative to the
ComfyUI backend — full generation UI, model browser and parameter controls, with no Python environment.

## License

MIT
