# Video gen-perf — full audit + optimization plan (2026-07-08)

> ## Phase 0+1 SHIPPED for LTX-2.3 (2026-07-08) — steps ~30 s → **~5.5 s (5.4×)**, coherent video+audio, test PASSES
> Clean-GPU rerun: steps 2..20 = **5.5 s** flat (cond 2.76 s / uncond+euler 2.77 s — pure streaming floor now; the
> first run's 8 s had Swarm contention). Phase probes (20 steps 25f 512×320): TE(Gemma)+connectors 11.4 s ·
> preload 0.5 s · denoise 111 s · latent unpack 37 ms · **video VAE decode 18.1 s (the predicted host-loop killer
> → Phase 4)** · rgb 62 ms · audio decode 1.7 s (vocoder 1.5 s). E2E ≈ 145 s vs 451 s baseline = **3.1×**.
> Block GPU-residency port (`LtxVideo2Block` Modulation/ShiftScale/GatedAdd via Add+SliceRows/
> AffineBroadcastLastDim/GatedResidualLastDim; `LtxVideo2Attention.ApplyGate` via sigmoid+0/1-GEMM-expand+Mul),
> per-gen RoPE table cache, device OutputLayer, AdaLnSingle host-memcpy removal, device `CfgEulerStep(−dt)` Euler,
> `[ltx2-phase]` probes. **Proven bit-identical** to the pre-port code (tiny-config CPU dump old-vs-new, byte-equal;
> `LtxVideo2PortNumericsDumpTests`, env `LTX2_PORT_DUMP`). GPU e2e (20 steps 25f 512×320, real 22B fp8): steps
> 2..20 ≈ **7.9–8.5 s** vs ~30 s baseline; coherent cat-in-garden + audio decoded. 165 s denoise vs ~600 s.
> Gotcha found: transformer `Dispose()` must NOT dispose GPU-promoted cache tensors (context may already be torn
> down) — null-only, dispose on mid-session key change instead.
> Next lever (Phase 2): the remaining ~8 s/step is the unpinned, unoverlapped 19 GB×2 weight stream.
>
> **Fleet quick wins shipped same day — Wan 1.3B GPU-verified:** Wan `ConditionTimeGroups` temb/proj device
> `CopyInto` (G=1), Wan+Hunyuan device final layer (G=1/B=1), Wan RoPE cos/sin memo + cross-step text-context cache
> (host-materialized, survives per-step `FreeActivations`), S2V per-step `TrimMemoryPool` → `trimPool:false`.
> A/B on `WanVariant_Gpu_E2E` (1.3B fp16, 33f 832×480, 20 steps, same seed, coherence-asserted PASS both):
> **4.45–4.61 → 4.09–4.11 s/step (~9%)** and step jitter ±80 ms → ±6 ms (the drains were the jitter). Hunyuan
> final-layer + S2V trim still need their own GPU e2e (weights/session-time bound, harnesses identified).
> Pre-existing (NOT ours): `HunyuanVideoDitTests.Forward_ProducesFiniteVelocityOfLatentShape` fails at HEAD with
> missing `txt_in.input_embedder.weight` synthetic key.

Applying the Krea2/Z-Image playbook (`KREA2_ARENA_GRAPH_F16_PLAN.md`, memories `vae-host-loops-hidden-20s`,
`cuda-graph-step-capture-recipe`, `image-genperf-host-glue-wins`) to the video fleet, with **LTX-2.3 22B
(dual-stream video+audio)** as the priority target. Audit only — no code changed yet.

## The playbook, distilled (what won the image war, in the order it wins)

1. **Wall-clock phase probes FIRST** (`[model-phase]` logs: TE / preload / per-step / VAE / post). Op profiles
   only see instrumented ops — the Krea2 20s VAE hid from 4 op-level experiments. Never optimize before phase
   attribution.
2. **GPU-residency ports** — replace every host `DataPointer` loop in blocks AND VAEs with existing device ops
   (`AffineBroadcastLastDim`, `GatedResidualLastDim`, `SliceRows`, `WanRmsNormChannel`, `Modulate`,
   `LayerNormNoAffine` backend op). Bit-identical, zero numerical risk, biggest wins (Chroma 4.6×, VAE 1250×).
3. **cuDNN fused SDPA** (`HARTSY_SDPA_CUDNN=1`) — 11× per call; **mask-null only**, D∈{64,128}; zero-cast when
   Q/K/V are native F16.
4. **F16 activations with packed-fp8 weights** (`HARTSY_DIT_F16` + F16→e4m3 activation-quant GEMM) — halves the
   bandwidth-bound elementwise/norm traffic, no VRAM regression; per-arch opt-in (audit FFN/SwiGLU overflow).
5. **CUDA-graph step capture** (`HARTSY_DIT_GRAPH`) — needs a drain-free fixed-shape step (device Euler via
   `CfgEulerStep`, fixed boundary buffers); wins wall only when launch-bound; **incompatible with weight
   eviction/streaming** (graph bakes weight pointers — the 43.145 crash).
6. **Caching + residency** — prompt-embedding cache, RoPE-table memoization, timestep-independent projections
   computed once/gen, `HARTSY_KEEP_MODELS`.
7. **Async hygiene** — no sync `cuMemcpyDtoD`, no per-step `TrimMemoryPool`, no pageable H2D of per-step tensors
   (one 2 KB host read of a per-block-consumed tensor = full pipeline stall — the Kandinsky temb bug).

## Current standings (video_comfy-vs-hartsy_2026-07-03.md + status docs)

| Model | Comfy | Hartsy now | Regime |
|---|---|---|---|
| LTX-2.3 22B (audio) | n/a | ~30 s/step, 451 s/clip | **streaming-bound + host-glue** (both, stacked) |
| LTX-0.9/0.9.5 2B | 2.84 s | 12–13 s | launch-bound (~640 tokens) — the graph-capture case |
| LTX-0.9.7 13B | — | ~25 s/step | fp8 per-GEMM transient dequant + host glue |
| Wan 2.1 1.3B | 6.28 s | 23.7 s | compute+glue mix |
| Wan 14B fp8 / S2V / Animate | 30.6 s (T2V) | 5.9× / 23.5 s/step / 12.5 s/step | fp8 dequant per GEMM |
| HunyuanVideo 13B fp8 | — | 2.15 s/step | already good; SDPA unfused ~1 s/step |
| Kandinsky-5 Lite 2B | — | 2.9 s/step | already good; SDPA unfused ~1.2 s/step |

## Audit — LTX-2.3 22B (the audio model; in-tree code targets LTX-2.3)

Two stacked bottleneck regimes, each individually fatal:

### Regime A — host glue (the Krea2 disease, worst case in the fleet)
- **`LtxVideo2Block`**: `ApplyShiftScale` (6×), `GatedAddInto` (8×), `ModulateRows` (4–6×), `Modulation`
  (26 small vectors) are ALL host `DataPointer` loops (`LtxVideo2Block.cs:209-275`) + `LtxVideo2Attention.ApplyGate`
  host loop 6×/block (`LtxVideo2Attention.cs:106`). ≈ **24 full-`[S,dim]` D2H/H2D round-trips per block-forward**
  × 48 blocks × 2 CFG × steps ≈ **1.1 M excursions/gen**. Every sync also stalls the weight prefetch stream.
  The device equivalents already exist (Krea2/Ideogram4Block pattern).
- **RoPE tables rebuilt on host every forward** (`LtxVideo2Transformer.cs:180-182` → `LtxVideo2Rope.BuildVideo/
  BuildAudio`) though grid-invariant across the whole gen (~300 rebuilds+uploads/gen). Apply is already GPU.
- **Per-step host Euler** (`LancePipelineCommon.EulerCfgStep`) drains latent + both velocities every step;
  final `OutputLayer` host LayerNorm + shift/scale; `AdaLnSingle` host sinusoid + memcpy drains ×8/forward.
- **`ModulateRows(ctx.Encoder …)` re-derives the constant connector features every block × every step.**

### Regime B — weight streaming (unique to LTX-2, the dominant floor)
- 48 blocks × ~19 GB fp8 streamed **per forward**, ×2 sequential CFG forwards/step = **~38 GB/step**.
- `CudaStreamingWeightCache.PinUploadSource` is **never enabled by any pipeline** → pageable H2D = silently
  synchronous staged copies, **zero compute overlap** (its own doc-comment says so). This is why deepening
  `prefetchAhead` beyond 2 "gave no win".
- This box: 4090 on **PCIe gen3 x16** (~12–13 GB/s pinned ceiling; 3060 shares the bus). Full-stream floor
  ≈ 1.5 s/forward pinned — vs ~15 s/forward today.
- `retainBehind: 0` — nothing stays resident between forwards even though ~16+ GB of VRAM could hold most
  of the DiT once activations shrink (F16 + fused SDPA).
- Host RAM 64 GB (checkpoint already in page cache) → pinned staging is feasible.

### Decode tail (once/gen, but the literal Krea2-20s pattern ×3)
- **Video VAE**: `LtxVaeResnetBlock3d.ChannelRms` host loop 2×/resnet (~15 resnets, near-full-res tensors) while
  GPU `WanRmsNormChannel` exists; host `Denormalize`/`PixelUnshuffle`/upsampler shuffle+residual adds.
- **Audio VAE**: all norms host (`LtxAudioPixelNorm`), host causal pads/crops between GPU convs.
- **Vocoder**: GPU convs/Snake, but host full-waveform MRF residual sums (`LtxBigVganGenerator.AddInPlace`
  ×18), host mel matmul + magnitude + pad/crop (`LtxAudioVocoder.MelSpectrogram`).
- No `DitDtype`/`DitStepGraph` opt-in; activations F32 everywhere; no prompt cache (Gemma-3-12B re-encoded +
  freed every gen); attention F32 → cuDNN path pays 3+1 casts (needs `allowF16`, has QK-norm so it's safe).

## Audit — rest of fleet (remaining items only)

**Wan family** (`WanVideoBlock` interior already GPU on the G=1 path):
- W1 RoPE cos/sin rebuilt on host every forward (`WanRope.BuildCosSin`, no memo — Kandinsky has the memo pattern).
- W2 text/image projections (`WanDitOps.TextEmbed`, i2v img proj) timestep-independent but re-run 2×/step.
- W3 temb `Buffer.MemoryCopy` drains in the condition embedder (`WanDitOps.cs:94,98`) — the Kandinsky stream-drain bug.
- W4 final layer: full-hidden D2H + host LayerNorm/modulate + re-upload (`WanDitOps.cs:120-151`) — every forward.
- W5 host patchify/unpatchify; W6 **TI2V multi-group (per-frame-timestep) block path fully host ×30 blocks/step**
  (`WanVideoBlock.cs:270-348`); W7 CFG sequential; W8 variant host glue (VACE pad, S2V AppendRows, Animate fusers);
  W9 **S2V calls `TrimMemoryPool()` every step** (`WanS2VPipeline.cs:171`).
- Weights upcast to F32 at load (`WanVideoTransformer.cs:145`) — no bf16/fp8 residency; 14B/S2V/Animate pay
  per-GEMM fp8 transient dequant (`HARTSY_FP8_NATIVE` recipe exists, validated 3× on T2V-14B, still opt-in).

**HunyuanVideo**: H1 final-layer host LayerNorm+Modulate D2H/forward (`HunyuanVideoDit.cs:156,241-254`);
H2 host patchify/unpatchify; H3 per-step re-upload of host-materialized prompt/pooled (pageable — the graph-
capture invalidator pattern); H4 shared head weights F32; V1 **VAE tiled path: per-tile `backend.Sync()` +
host feather accumulate** (`HunyuanVideoVaeDecoder.cs:197-256`). Single forward/step (embedded guidance),
fp8-resident, GPU RoPE → **best graph-capture candidate in the fleet**.

**Kandinsky-5**: K1 CFG 2× sequential forwards (rope `ApplyGpu` is batch-1-only, blocks B=2 batching);
K2 host patch-embed/unpatchify/final permute; K3 host sinusoid (small). Rope memoized, temb fixed — closest
to done.

**Cross-cutting engine facts (verified)**:
- cuDNN SDPA fires only when `mask is null` (`CudaBackend.cs:2408,2416`). Video models passing text-padding
  masks never reach flash attention. Preferred fix at B=1: **trim padded conditioning tokens** (LTX-0.9 already
  does `SliceBatchElementPrefix`) instead of masking; cuDNN varlen support is the bigger later option.
- No sync `CopyDeviceToDevice` left in any video path (the Concat fix inherited). Clean.
- No video model opts into `DitDtype.Act` / `DitStepGraph`.

## Plan — LTX-2.3 first (phases ordered by leverage; each phase deployed + coherence-checked video AND audio)

**Phase 0 — instrument.** `[ltx2-phase]` wall-clock probes: TE(Gemma) / connectors / per-step (per-forward) /
VAE-video / VAE-audio / vocoder / mux. Plus one instrumented run logging per-forward upload-vs-compute overlap
(event timings around `BeginUploadAsync`/`AwaitWeights`). This calibrates every estimate below. ~1 deploy.

**Phase 1 — block GPU-residency port (Regime A).** Straight Krea2Block/Ideogram4Block replay, bit-identical:
- `Modulation`/`Slice` → device `Add` + `SliceRows`; `ApplyShiftScale`/`ModulateRows` → `AffineBroadcastLastDim`;
  `GatedAddInto` → `GatedResidualLastDim`; `ApplyGate` → device (sigmoid-gate = `Sigmoid`+`Mul`, per-head layout
  as in Krea2Attention); OutputLayer → backend `LayerNormNoAffine` + `Modulate`.
- Cache RoPE tables per (grid,fps)-sig for the gen; **hoist the per-block `ModulateRows(ctx.Encoder…)`** — the
  prompt-mod scale/shift is per-block but the encoder is constant: still compute on device, and only if probes
  show it matters, precompute all 48 per-block modulated encoders once/step.
- Device Euler: `CfgEulerStep` (cond,uncond,g,dt) on resident latents; kill the per-step D2H (keep the optional
  preview drain gated on `onProgress`).
- **DoD:** bit-identical-class output (relL2 vs baseline at 2 steps), coherent clip + audio, per-step probe delta.
  Expected: removes ~117 k syncs/gen AND unblocks prefetch overlap (prereq for Phase 2 to show its full win).

> ### ✅ Phase 2 SHIPPED (2026-07-08) — steps 5.5 s → **2.6 s** (30 s at audit start = **11.5×**), PASSES, coherent
> (a) **Pinned staging ring** (`CudaStreamingWeightCache`): 3 × block-size `cuMemHostAlloc` slots; weights memcpy →
> pinned slot → async HtoD, slot reuse event-gated. Direct `cuMemHostRegister` of mmap'd weights is a TRAP —
> unaligned ranges fail INVALID_VALUE, contiguous weights share boundary pages (ALREADY_REGISTERED aborts all-or-
> nothing), and a copy straddling pinned/unpinned pages fails INVALID_VALUE at `cuMemcpyHtoDAsync`. Auto-enabled by
> `BlockStreamingController` (kill: `HARTSY_STREAM_PIN=0`), so Hunyuan/Flux streaming inherits it.
> (b) **VRAM-gated resident block prefix** (`LtxVideo2Pipeline`): 15 of 48 blocks resident on this 24 GB card
> (`HARTSY_LTX2_HEADROOM_MB`, default 4096). Prefix alone: 5.5 → 4.2 s/step.
> (c) **CFG pairing** (`LtxVideo2Transformer.ForwardCfgPair`): cond+uncond through each block back-to-back → streamed
> weights upload once/step (24.3 GB/step instead of 48.7). Proven bitwise-identical to two Forwards
> (`CfgPair_MatchesTwoForwards`); real-weight seed-42 frame matches Phase 1.
> Step floor is now ~24.3 GB ÷ ~13 GB/s ≈ 1.9 s transfer + tail — still stream-bound. Next tiers: bigger prefix via
> F16 activations (Phase 3), VAE decode 18.5 s (Phase 4 — NOW the biggest single phase), TE 11.4 s (Phase 5 cache).

**Phase 2 — streaming overhaul (Regime B; the LTX-2-specific lever).**
1. **Pin the upload path**: set `PinUploadSource=true` for block-swap (weights are mmap'd — if `cuMemHostRegister`
   on file-backed pages misbehaves, fall back to a 2-block pinned staging ring: worker-thread memcpy mmap→pinned,
   async H2D from staging). Re-tune `prefetchAhead` (now it will matter).
2. **Batch CFG cond+uncond into one B=2 forward** → halves streamed bytes/step (38→19 GB). Needs B=2 through
   block/attention (currently B=1-only). If B=2 is too invasive, alternative: retain-window ping-pong —
   run cond+uncond back-to-back per *block* (block-level CFG interleave) so each block's weights upload once/step.
3. **Maximize residency**: size a resident prefix of blocks to free VRAM (24 GB − activations − VAE headroom),
   stream only the remainder (`retainBehind`/partition change in `BlockStreamingController` usage). With F16
   activations (Phase 3) ~16 GB of the 19 GB DiT can sit resident → ~3 GB streamed/forward ≈ 0.25 s, fully
   hidden under compute.
- Floor math (gen3 x16 ≈ 13 GB/s): today ~15 s/forward unpinned+serial → pinned+overlapped full stream 1.5 s
  → B=2 one forward/step → resident-prefix ~0.25 s hidden. **DoD:** probe shows upload time hidden under
  compute; step time within ~20% of pure-compute time.

**Phase 3 — F16 activations + zero-cast cuDNN SDPA.** Opt LTX-2 blocks into `DitDtype.Act` (QK-normed → safe;
audit the GELU FFN inner dim like Z-Image's SwiGLU). Native-F16 SDPA path removes the 3+1 casts ×6 attentions
×48 blocks. Halves activation VRAM → directly buys Phase 2.3 residency budget. Verify text/cross-modal attn
masks are null at runtime (trim padded text if not — cuDNN is mask-null-only).

**Phase 4 — decode-tail ports.** Video VAE `ChannelRms`→`WanRmsNormChannel` (drop-in, the exact Krea2 fix),
device Denormalize/PixelUnshuffle/upsampler; audio VAE `LtxAudioPixelNorm`→`WanRmsNormChannel`, device pads;
vocoder mel matmul → `backend.Linear`, MRF sums → `backend.Add`, pad/crop → device. Probes decide priority
inside this phase.

**Phase 5 — pipeline parity + (conditional) graph.** Gemma prompt-embedding cache keyed on tokens (12B encode
is expensive, TE freed every gen); `HARTSY_KEEP_MODELS` semantics for the resident block prefix. Step-graph
capture ONLY for the resident prefix region if Phase 2.3 leaves streaming for a minority of blocks — a captured
graph cannot span streamed (re-pointered) weights (the 43.145 eviction-crash rule).

> ### ✅ Hunyuan+Kandinsky fused-SDPA round (2026-07-08) — both PASS, coherent (cat / snow leopard verified)
> `allowF16: true` at all 5 QK-normed, mask-null SDPA sites (HunyuanImageBlock/SingleBlock joint attn;
> Kandinsky5Block encoder-self/self/cross) + `HARTSY_SDPA_CUDNN=1`:
> **HunyuanVideo 2.15 → 1.29 s/step (1.67×), clip 86 → 56 s** (fp8-resident + FP8_NATIVE recipe; also GPU-verifies
> the device final-layer quick win). **Kandinsky-5 2.9 → 0.83 s/step (3.5×), 30-step clip 102 → 42 s** — the
> 7168-token SDPA was even more dominant than profiled. Remaining per the audit: Hunyuan patchify/unpatchify host
> loops + per-step prompt re-upload + VAE tiled per-tile Sync; Kandinsky patch glue; graph capture for both.

> ### I2V/S2V round (2026-07-08, engine 44.9-local deployed)
> **Wan I2V-14B through Swarm was BROKEN on 44.8** (blotchy un-denoised output, composition preserved —
> reproduced 2×): the per-forward conditioning recompute read CLIP/text tensors whose device buffers the
> per-step `FreeActivations` had dropped. **FIXED by the I2V context cache in 44.9** (compute once,
> host-materialize): output now pristine (astronaut-on-horse gallop, faithful to init). Lesson: any tensor
> consumed across steps in a pipeline that calls `FreeActivations` per step MUST be host-materialized or
> recomputed from host-backed sources — silent garbage otherwise.
> **I2V perf**: 200 s warm vs T2V-14B's 37 s at the same size/steps — ~160 s of I2V-specific overhead
> (init VAE-encode, concat-conditioning build, per-step mask?) → needs phase probes in
> `GenerateImageToVideoConcat` (next target).
> **S2V trim REVERTED**: `FreeActivations(trimPool:false)` OOM'd mid-run (40 blocks × 13824 FFN) even with
> FP8_NATIVE — the audio injector's churn needs the per-step pool trim until its host glue is ported.
> **Wan 14B T2V at cfg 6 (fp8)** shows rainbow glitch patches through Swarm — the known fp8-CFG amplification;
> extension should clamp/renorm CFG for fp8 Wan (detector CfgRescale exists engine-side; verify it engages
> through the extension path).

## Plan — rest of fleet (interleave as capacity allows; ordered by value)

1. **Quick wins, one deploy each:** S2V per-step `TrimMemoryPool` → `trimPool:false` (W9); Wan RoPE memoization
   (copy `Kandinsky5Rope` pattern, W1); Wan text/img projection cache across steps (W2); temb drain fix (W3,
   the proven Kandinsky fix); HunyuanVideo VAE per-tile `Sync()` + host feather-blend → device accumulate (V1).
2. **Shared final-layer port** (W4 + H1): `WanDitOps.FinalLayer` and `HunyuanVideoDit` tail → device
   `LayerNormNoAffine` + `Modulate` — one fix, every Wan variant + Hunyuan inherits.
3. **fp8 residency defaults**: promote the validated `HARTSY_FP8_NATIVE` + no-cache-casts recipe for Wan-14B/
   S2V/Animate/LTX-0.9.7 (the per-GEMM transient dequant is their stated dominant cost); resolve the fp8-CFG
   parity item with `CfgRescale` where needed.
4. **cuDNN SDPA engagement**: Hunyuan (~1 s/step) + Kandinsky (~1.2 s/step @7168 tokens) — verify mask-null
   (trim padded text at B=1), pass F16 QKV where QK-normed. Then Wan.
5. **Graph capture** where launch-bound: **LTX-0.9 2B first** (12–13 s vs Comfy 2.84 s, explicitly diagnosed
   launch-bound — the exact case where capture wins wall), then HunyuanVideo (single-forward, fp8-resident,
   fixed shapes), then Kandinsky/Wan after their loops go drain-free (device Euler each).
6. **Wan TI2V multi-group host path port** (W6) + patchify/unpatchify device ports (W5/H2/K2) + CFG batching
   (W7/K1; K1 needs batched rope apply).

## Targets (calibrate with Phase-0 probes before promising)

| Model | Now | After plan | Basis |
|---|---|---|---|
| LTX-2.3 22B clip (25f/512×320/20 steps) | 451 s | **~60–100 s** | glue-free compute + hidden ≤0.25 s/step stream |
| LTX-0.9 2B | 12–13 s | **~4–6 s** | graph capture on a launch-bound loop |
| Wan 1.3B | 23.7 s | **~8–12 s** | rope/proj caches + final-layer + F16 + graph |
| HunyuanVideo | 2.15 s/step | **~1.2–1.5 s/step** | fused SDPA + graph + tail ports |
| Kandinsky-5 | 2.9 s/step | **~1.7–2 s/step** | fused SDPA + patch glue + graph |

## Method rules (from the war, non-negotiable)

- Phase probes before any op-level work; wall ≠ op-profile means an un-instrumented host phase.
- Every deploy: verify coherence (video frames AND decoded audio waveform stats) — fast+garbage is worse than slow.
- One `--filter` per GPU test process; never rebuild a test project mid-bg-run; regression-check the Krea2
  flagship configs after ANY engine-level change (`always-regression-check-flagship`).
- Bit-identical ports first (residency/caching), numeric-risk changes (F16) behind flags with per-stage relL2.
