# Multi-GPU Component Placement — the recipe pattern

**Status:** Live. First wave (2026-08-02): Wan video, Flux.1, SDXL. Second wave (2026-08-04): Qwen-Image
(TE + VAE incl. the edit-reference encode; note the `_cachedPackedRef` pin-owner hazard annotated in the
pipeline), Chroma (TE + VAE; the cached-conditioning frees STAY on the denoise backend — the step-graph
warm-up pins them there), HunyuanImage (TE both encoder paths + VAE, including the explicit host bridge for
its deliberately device-resident latent→VAE chain), LTX-1 (Wan-style ctor threading), LTX-2 (TE incl. the
Gemma-vs-prefix evict skip — the biggest single win — plus video AND audio VAE/vocoder on `VaeDevice`).
Flux/SDXL residual img2img/inpaint/control VAE-encode gaps were closed in the same pass. Remaining
unported: Krea2, Ideogram4, and the smaller single-file families. `VaeDevice` is now settable from the
SwarmUI extension (`VaeGpuId`) and the CLI (`--vae-gpu`).

## Why this works with zero copy machinery

Every pipeline already **host-materializes** its stage boundaries, for VRAM reasons that predate placement:

- Encoder → denoiser: the pre-loop `_ = embeddings.DataPointer` sweeps (Flux) or host post-processing
  (`CfgHelper.SliceBatchElement`, `ZeroPaddedRows` — Wan) force the conditioning to host so encoder
  activations can be reclaimed.
- Denoiser → VAE: `UnpackLatent` and friends are host-side.

A component placed on another GPU therefore just uploads from the host copy — no peer copy needed. **That
makes the host materialization load-bearing**: annotate it (`// LOAD-BEARING for TextEncoderDevice placement`)
so a future "keep embeddings device-resident" optimization doesn't silently break cross-GPU placement. Any
such optimization must go through `IBackend.CopyFromPeer` instead.

## The four-step pattern

1. **Backends from context.** `RecipeContext` carries `TextEncoderBackend` / `VaeBackend` (null = primary).
   `DiffusionPipelineBase` exposes `TextEncoderBackend` / `VaeBackend` init-properties defaulting to
   `Backend`; the recipe sets them at construction:
   ```csharp
   FooPipeline pipeline = new FooPipeline(context.Backend, ...)
   {
       TextEncoderBackend = context.TextEncoderBackendOrDefault,
       VaeBackend = context.VaeBackendOrDefault,
   };
   ```
   Recipe-level pipelines that hold their own encoder backend field (Wan's `_textBackend`, SDXL's weighted
   conditioning) take it from the same source.

2. **Preload/encode/free against the owning backend.** Every `PreloadWeights`/`Encode`/`Sync`/`FreeWeights`/
   `FreeActivations` in the text-encode phase moves from `Backend` to `TextEncoderBackend`; every VAE
   encode/decode (and its preload/free) moves to `VaeBackend`. `GpuTransferHelper` state is per-backend, so a
   free issued on the wrong backend is a silent no-op — the leak shows up as VRAM growth on the encoder's GPU.

3. **Skip cross-device contention hacks when split.** Evict-DiT-for-T5 dances
   (`if (_ditResident) { FreeWeights(transformer); }`) only make sense when encoder and denoiser share a
   device — gate them with `ReferenceEquals(TextEncoderBackend, Backend)`.

4. **Flags to every backend.** Recipes that set per-backend flags (`CacheWeightCasts`, fp8 toggles) must
   apply them over `context.AllBackends`, not just `context.Backend` — the encoder's device needs the same
   policy or it re-inflates there (see `Flux1Recipe`).

## Verification per pipeline

- Same-seed generation with and without `TextEncoderDevice` must be **bit-identical** (placement moves math,
  it doesn't change it). Compare output hashes.
- `GetD2hSyncCount()` during the denoise loop stays ~0 — the boundary D2H happens once, pre-loop.
- VRAM watermarks per device via `CudaTopology.Probe()`: the encoder's bytes must move off the primary.

## What stays on the primary (deliberately)

Block streaming (lowvram) binds denoiser weights on `Backend` only — TE-on-B composes with
lowvram-denoiser-on-A for free. Composition machinery (ControlNet residuals, IP-Adapter, refiner) stays on
`Backend`; Wan's CLIP-vision + first-frame VAE-encode currently ride the primary too (split further only if
profiling shows a win).
