# Image deferred-wave verification — 2026-07-17

Engine `1.0.0-alpha.49.24-local`, extension wired (patches applied on `a91c56f`), RTX 4090, live
SwarmUI API (same request path as the 07-16 campaign). Follow-up to
`image_conditioning_2026-07-16.md` — closes every deferred item from that campaign that is
closable on this box. Wall times are single runs via the API (includes HTTP + queue overhead).

## Flagship regression gate (per-deploy requirement)

| Model | Bar | Measured | Result |
|---|---|---|---|
| Krea2-Turbo 1024²/8st warm | < 6.5 s + clean | 4.50 s / 4.50 s back-to-back, clean apple | PASS |
| Z-Image-Turbo 1024²/8st warm | ≤ 3.2 s + clean | 2.78 s, clean | PASS |

## New feature verifications (all live via SwarmUI)

| Feature | Checkpoint | Evidence |
|---|---|---|
| SD1.5 ControlNet **segmentation** | control_v11p_sd15_seg_fp16 + in-engine UperNet-ConvNeXt-Small ADE20K | watercolor street: bus and pedestrians match the reference photo's segment layout; annotator parity 100.000% class agreement vs transformers |
| SDXL **Union** ControlNet (canny type) | controlnet-union-sdxl-promax @0.8 | bronze apple, contour-exact vs the same reference used for the plain-CN run |
| SDXL **Union** ControlNet (depth type) | controlnet-union-sdxl-promax @0.8 | glass apple at reference position/scale, depth-consistent |
| IP-Adapter **FaceID-PlusV2** SDXL | ip-adapter-faceid-plusv2_sdxl.bin @0.9, V2 weight 1.0 | log confirms variant=FaceID-PlusV2 + ArcFace + CLIP-Vision + 560-weight companion LoRA merge; simple-prompt portrait carries reference identity clearly (engine e2e identity cosine 0.3456 vs 0.224 for FaceID base) |
| **Flux-Depth conditioning fix** | flux1-depth-dev-fp8, byte-identical rerun of the 07-16 seed-11 request | 07-16 output had heavy web-image-prior junk (UI chrome, red caption text, filmstrip borders); 49.24 output is a photorealistic depth-faithful apple with only faint chalk-text residue on flat background bands. Cause: engine now reproduces BFL's bicubic-antialias depth upsample (training distribution includes its edge ringing); map corr 1.000000 vs the BFL reference pipeline |

## Regression spot-checks (unchanged behavior confirmed)

| Check | Result |
|---|---|
| SDXL plain canny (diffusers_xl_canny_full @0.9, 07-16 request replay) | contour-locked bronze apple, matches 07-16 |
| IP-Adapter Plus SDXL @0.8 (07-16 request replay; affected by the ResamplerLayer GELU tanh→erf parity fix) | near-identity transfer, quality ≥ 07-16 |
| Flux warm-repeat bit-stability, Kontext, Fill | not re-run (no code touched those paths this wave; 07-16 evidence stands) |

## Lens / Lance perf (live via SwarmUI, warm = second identical request)

| Model | Config | 07-16 (bring-up) | 49.24 live warm | Engine-side s/step |
|---|---|---|---|---|
| Lens Turbo | 1024², 8 steps, cfg 1 | ~25-30 s/step (≈200 s+ denoise alone) | **63.6 s total** (GPT-OSS TE encode dominates; denoise+decode ≈ 6.8 s) | 14.4 → **0.55 s** (26×) |
| Lance 3B | 768², 20 steps, cfg 4 | ~260 s total | **14.8 s total** | 12.8 → **0.33 s** (39×) |

Parity gates for the perf work: Lens velocity corr 0.999973 / Lance 20-step image corr 0.999950 vs
pre-change baseline (see `2026-07-16_lens_lance_genperf.md` for the full fix list).

## Regression coverage — shared logic touched by the wave

Every shared primitive the five branches changed was traced to its consumers and each consumer
re-verified (live gen and/or numeric parity). No model regressed.

| Shared change (file) | Blast radius (consumers) | Regression evidence |
|---|---|---|
| `IpAdapterPlusResampler.ResamplerLayer` GELU tanh→erf | IP-Adapter Plus family ONLY (grep-confirmed: no other ResamplerLayer users) | Live: IPA-Plus SDXL, IPA-Plus-Face SDXL, FaceID-PlusV2 SDXL — all coherent, identity carried |
| `IpAdapterLoader.Detect`/`BuildConfig` (+isFaceIdV2, hoisted hasPerceiver) | ALL IP-Adapter variants | Live: IPA std SDXL, IPA std SD1.5, IPA Plus, plus-face, FaceID base, FaceID-PlusV2 — all coherent; `isPlus` logic byte-preserved |
| `ControlNet`/`ControlNetLoader`/`Config`/`Conditioning` union path (null-guarded) | ALL SDXL + SD1.5 ControlNets | Live: SDXL plain canny, SD1.5 canny/depth/normalbae — all follow the reference; union path branches on null so plain path untouched at runtime |
| `TimestepEmbedding` (Forward→ForwardEmbedding extraction) | ALL SDXL UNet models (runs every step) | Diff is a pure refactor (identical op order); exercised by every SDXL live gen above + flagship gate |
| `AdditionEmbedding.EmbedScalar` (private→internal) | SDXL UNet | Signature/behavior unchanged; same coverage |
| `DepthAnythingPreprocessor` (flux-mode kernel only) | Flux-Depth (changed) + SD/SDXL depth CN (min-max path) | Min-max bilinear loop extracted verbatim into `ResizeBilinearAlignCorners`, called only in the unchanged branch → SD1.5 depth CN live gen confirms; Flux-Depth live A/B confirms the intended change |
| `ConvBnFold` (new) + `NormalBaeModel` refactor | NormalBAE annotator | Numeric parity re-verified (out_res1 7.9e-6, unchanged) + SD1.5 normalbae CN live gen: clay apple matches reference shape |
| `SiglipImagePreprocessor` bilinear→AA-bicubic + `Resample.cs` (new) | SigLIP vision path (grep: zero non-test callers; deployed Redux uses its own resize) | No live blast radius; `ResampleTests` (Unit, torch-golden) + SigLIP/Redux parity tests green |

Test tiers after the wave: **Vision unit 103/103**; **Diffusion shared-code sweep (adapters/CN/IPA/
embeddings/Lens/Lance/SDXL/UNet) 193/194** with parallelism ON — the 1 failure is
`AnimaLlmAdapterDiffTests`, a **pre-existing mis-tagged test** (its `t5_input_ids.bin` fixture is
gitignored / never committed, fails on any fresh checkout; the Anima model was not touched by this
wave). The earlier `LanceLatentPatchTests` parallel-suite corruption was root-caused (undisposed
test Tensors finalized mid-assert → glibc tcache stomp) and fixed — repro filter now 103/103 5×/5.

**Test-hygiene flag (not this wave's code, surfacing for the owning sessions):** at least three
Unit-tier tests fail on a clean checkout because they depend on gitignored reference fixtures or
OOM the host — `AnimaLlmAdapterDiffTests`, `AceStep15GuidanceParityTests` (music), and the ACE-Step
e2e (music, 17 GB RSS). Per `CODE_STYLE.md` these need `[Trait("Category","Integration")]` + env
gates so `main` stays green on the hosted runner. Left untouched here since they belong to the
audio/Anima sessions; easy to tag on request.

## Notes / still open

- Swarm scheduler quirk (pre-existing, observed on every boot tonight): the FIRST request for a
  not-yet-loaded model gets a spurious ~2 s "backend does not have that model" refusal while the
  load actually proceeds; an immediate retry rides the load and succeeds. Worth a look in the
  extension's model-list advertisement vs BackendHandler timing.
- ReduxResolver still uses its own ImageSharp resize (antialiased, close to reference); switching
  it to the engine's now-HF-exact `SiglipImagePreprocessor` is a cheap exactness win next deploy.
- FaceID-Plus/PlusV2 SD1.5 variants: engine parity corr 1.000000 on the real checkpoints; live
  SD1.5 gen not re-run (SDXL PlusV2 path is the live-verified representative).
- GPT-OSS TE per-prompt encode (~55 s) now dominates Lens totals — candidate: pipeline-level
  prompt cache like F-Lite/Lumina.
- Union ControlNet segment/tile/repaint types: raw-map pass-through wired; dedicated live checks
  deferred with the preprocessing follow-ups.
