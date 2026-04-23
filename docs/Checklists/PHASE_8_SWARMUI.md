# Phase 8 — SwarmUI Extension

> **Goal:** Register SharpInference as an in-process SwarmUI backend.
> **Packages:** SharpInference.SwarmUI

---

## 1. Research

- [ ] SwarmUI backend extension API, model management conventions, parameter passing
- [ ] Required interface methods, expected output format/metadata

## 2. Planning

- [ ] Map SwarmUI API → SharpInference pipeline calls
- [ ] Model format compatibility, LoRA/ControlNet passthrough, progress reporting, error surfacing

## 3. Implementation

- [ ] Backend registration, model discovery
- [ ] Text-to-image, image-to-image, inpainting translation
- [ ] LoRA + ControlNet passthrough
- [ ] Progress reporting, cancellation, model load/unload, settings UI

## 4. Testing

- [ ] Unit: registration, model list
- [ ] Integration: t2i, i2i, LoRA, progress events, cancellation, model hot-swap
- [ ] Manual: full SwarmUI workflow, output quality vs Python backend

## 5. Documentation

- [ ] Installation guide, supported models/formats, troubleshooting, performance comparison

## 6. Review & Merge

- [ ] Code review (all interfaces implemented, no memory leaks on model switch)
- [ ] Multi-version SwarmUI compatibility test
- [ ] Merge to main branch
