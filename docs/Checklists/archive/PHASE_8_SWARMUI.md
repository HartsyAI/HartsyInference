# Phase 8 — SwarmUI Extension (primary consumption path)

> **Goal:** The recommended way to run HartsyInference. The extension registers the engine as a
> SwarmUI backend (an alternative to the ComfyUI backend) so users drive every modality from
> SwarmUI's UI and API.
> **Status:** SHIPPED and actively maintained. Lives in its own repo, installed as a SwarmUI
> extension (not a NuGet package in this repo).
> **Repo:** https://github.com/HartsyAI/SwarmUI-HartsyInference-Backend
> **Local dir:** `SwarmUI/src/Extensions/SwarmUI-HartsyInference`

---

The extension is real and in use: it registers a `HartsyInferenceBackend`, provides per-architecture
loaders (SD1.5/SDXL/SD3/Flux/Flux.2/Chroma/AuraFlow/Ideogram4/Anima/ERNIE/Boogu/Krea2/HiDream/Lens
and more), video loaders (Wan, LTX, WanAnimate), audio (ACE-Step/MusicGen), LoRA passthrough,
ControlNet / IP-Adapter / img2img / inpaint mask resolvers, live previews, and checkpoint
conversion. Multiple image and video models generate end-to-end through SwarmUI on this backend
(the diffusion/video e2e benchmark routes the same SwarmUI request to ComfyUI vs the HartsyInference
backend on one GPU). Remaining work is breadth and polish, not bring-up.

## 1. Research

- [x] SwarmUI backend extension API, model management conventions, parameter passing
- [x] Required interface methods, expected output format/metadata

## 2. Planning

- [x] Map SwarmUI API → HartsyInference pipeline calls
- [x] Model format compatibility, LoRA/ControlNet passthrough, progress reporting, error surfacing

## 3. Implementation

- [x] Backend registration, model discovery (`HartsyInferenceBackend`, per-architecture loaders)
- [x] Text-to-image, image-to-image, inpainting translation (`Img2ImgResolver`, `MaskResolver`)
- [x] Video generation translation (Wan / LTX / WanAnimate loaders + video param resolver)
- [x] LoRA + ControlNet passthrough (`LoraApplier`, `ControlNetResolver`, `IpAdapterResolver`)
- [x] Progress reporting / live previews (`PreviewEncoder`), model load/unload (backend lifecycle)
- [ ] Cancellation mid-generation, settings UI polish

## 4. Testing

- [ ] Unit: registration, model list
- [ ] Integration: t2i, i2i, LoRA, progress events, cancellation, model hot-swap
- [x] Manual: full SwarmUI workflow verified (image + video models generate e2e through the backend)

## 5. Documentation

- [x] Extension README + per-topic docs (architecture, integration, pipeline translation, video plan)
- [ ] End-user installation guide + troubleshooting + performance comparison writeup

## 6. Review & Merge

- [ ] Code review (all interfaces implemented, no memory leaks on model switch)
- [ ] Multi-version SwarmUI compatibility test
- [x] Shipped in the extension repo (maintained out-of-tree, not merged into this repo)
