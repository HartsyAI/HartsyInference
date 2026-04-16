# Phase 8 — SwarmUI Extension

> **Goal:** Register SharpInference as an in-process SwarmUI backend.
> **Packages:** SharpInference.SwarmUI

---

## 1. Research

- [ ] Study SwarmUI backend extension API surface
- [ ] Study SwarmUI model management conventions
- [ ] Study SwarmUI parameter passing (how prompts, settings, LoRAs are sent to backends)
- [ ] Identify all SwarmUI backend interface methods that must be implemented
- [ ] Document SwarmUI's expected image output format and metadata

## 2. Planning

- [ ] Map SwarmUI backend API → SharpInference pipeline calls
- [ ] Plan model format compatibility (SwarmUI expects certain model file conventions)
- [ ] Plan LoRA/ControlNet passthrough from SwarmUI settings to SharpInference adapters
- [ ] Plan progress reporting (SwarmUI expects step-by-step updates)
- [ ] Plan error handling (how to surface SharpInference errors in SwarmUI UI)
- [ ] Write agent instructions for Phase 8

## 3. Implementation

- [ ] SwarmUI backend registration — register SharpInference as an available backend
- [ ] Model discovery — expose SharpInference model registry to SwarmUI model list
- [ ] Text-to-image — translate SwarmUI generation request → SharpInference pipeline call
- [ ] Image-to-image — translate SwarmUI img2img request → SharpInference pipeline call
- [ ] Inpainting — translate SwarmUI inpaint request → SharpInference pipeline call
- [ ] LoRA support — pass SwarmUI LoRA selections through to SharpInference LoRA manager
- [ ] ControlNet support — pass SwarmUI ControlNet settings through
- [ ] Progress reporting — stream step progress back to SwarmUI UI
- [ ] Cancellation — forward SwarmUI cancel requests to SharpInference pipeline
- [ ] Model loading/unloading — respond to SwarmUI model switch requests
- [ ] Settings UI — expose SharpInference-specific settings in SwarmUI backend config

## 4. Testing

- [ ] Unit test — backend registration succeeds
- [ ] Unit test — model list populated correctly
- [ ] Integration test — text-to-image through SwarmUI backend interface
- [ ] Integration test — img2img through SwarmUI backend interface
- [ ] Integration test — LoRA application through SwarmUI
- [ ] Integration test — progress events received by SwarmUI
- [ ] Integration test — cancellation stops generation
- [ ] Integration test — model hot-swap works without crash
- [ ] Manual test — full SwarmUI workflow with SharpInference backend
- [ ] Manual test — compare output quality vs Python backend

## 5. Documentation

- [ ] SwarmUI extension installation guide
- [ ] Supported models and formats guide
- [ ] Troubleshooting guide (common issues and solutions)
- [ ] Performance comparison vs Python backend

## 6. Review & Merge

- [ ] Code review — correct implementation of all SwarmUI backend interfaces
- [ ] Code review — no memory leaks during model switching
- [ ] Test with multiple SwarmUI versions for compatibility
- [ ] Merge to main branch
