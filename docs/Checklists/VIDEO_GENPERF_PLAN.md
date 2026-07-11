# Video gen-perf — full audit + optimization plan (2026-07-08)

> ## Round 11 (2026-07-11) — Wan-Animate CHECKERBOARD SOLVED + motion-encoder OOM fix (`44.84-local`, details in E2E_PARITY_WORKLOG round 11 / MODEL_STATUS_VIDEO)
> The Animate checkerboard was a **converter bug, not perf**: `ApplyFp8ScaledDequant` dropped the KJ v2 Animate
> checkpoint's **BF16** `.scale_weight` companions (guarded `== F32`; all other Wan fp8 ckpts are F32) → fp8
> weights ran ~5× hot → block-0 activations exploded (stage dump rms 0.17→4.4e5) → token collapse = the tile.
> Fixed by accepting BF16/F16 scalar scales (F32 path byte-identical, zero regression). **Perf-relevant second
> fix:** `WanAnimateTransformer.BuildMotion` now runs the StyleGAN motion encoder in **8-frame chunks** (comfy
> `encode_bs=8`) with per-chunk `FreeActivations` — the all-frames 512² conv stack OOM'd beside the resident
> 14B fp8 DiT (real-input 17f 480² e2e now completes). Animate is now the shared-`WanVideoBlock` fast path +
> correct; no per-step perf pass done yet (`WanAnimateLoader` also still lacks the round-6 warm cache).

> ## Round 10 (2026-07-11) — Wan T2V step-floor: **COMPUTE-FLOOR EARLY-BAIL, no code shipped** (details in E2E_PARITY_WORKLOG round 10)
> Profiled `wan2.1_t2v_14B_fp8_scaled` (25f 512×320) under `HARTSY_PROFILE_SYNC=1`: a `cuStreamSynchronize` after
> EVERY op left the step at **~1.79 s/step, identical to the un-profiled 1.82 s baseline** — the definitive
> compute-bound signature (no async overlap lost, launch overhead ~0%). True-GPU-time op table: **Linear (native
> fp8 GEMM) 4930 ms + SDPA (cuDNN fused flash) 2003 ms = ~68% of GPU time**, the AdaLN/norm/rope/permute glue
> already device-resident (~0.4 s/step, ~22%), and `H2D_MISS_BIG` 1335 ms is **one-time cold-start** (umT5 8.4 GB
> encode + DiT 16.4 GB preload), NOT a per-step drain. The host `scheduler.Step`/CFG-combine round-trip is a tiny
> 1.1 MB latent (a few ms/step, <0.3%) — killing it saves nothing and does not unlock graph capture. **Graph
> capture would not help** (removes only the already-negligible launch overhead; would fight the resident 14B DiT
> and bake per-forward cache re-faults). **fp8 coverage audit CLEAN:** `NativeFp8Gemm=True`, every Wan GEMM shape
> clears the `K%16 && N·outBytes%16` guards (dims 5120/13824), and the profile shows NO `Cast`/dequant-to-F16 op →
> zero fp8→F16 recasts. No engine edit; shared kernel untouched. `44.82-local` stands. The only remaining
> theoretical lever is F16 activations (`HARTSY_DIT_F16`) for the ~0.4 s/step glue — minority, fp8-CFG-risky,
> out of scope. Wan T2V is at its fp8 compute floor on the 4090.

> ## Round 9 (2026-07-11) — batched-CFG step-floor lever: **NO-GO, no code shipped** (details in E2E_PARITY_WORKLOG round 9)
> The "fuse cond+uncond into one B=2 forward to stream weights once, ~2×" thesis does NOT hold for either video
> target. **LTX-2.3** already captures the stream-once win bit-exactly via block-level CFG interleave
> (`LtxVideo2Transformer.ForwardCfgPair` runs cond then uncond back-to-back per streamed block — weights upload
> once/step, each sample stays a separate per-tensor-quantized GEMM = bit-identical); a true M=2·sv merge adds no
> streaming and breaks fp8 exactness. **Wan** DiT is **fully resident** (no `BlockStreamingController` — grep NONE;
> whole 16.4 GB fp8 DiT preloaded + KEEP_MODELS), so there is no weight stream to fetch once; at S=4480 tokens the
> fp8 GEMMs are large-M compute-bound and B=2 is FLOP-identical (no speedup). And the fp8 **activation quant is
> per-tensor** (`native/cuda/dequant/fp8_quant.cu` absmax over the whole tensor) → batching cond+uncond shares one
> scale = NOT bit-exact (round-7 precedent on this path: temb relL2 3.5e-2, 17× over the ~2e-3 fp8 floor). Wan's
> real step lever is CUDA-graph capture (resident = capturable; needs a device Euler/UniPC step to drop the host
> `scheduler.Step` drain) + a native-fp8-path K%16 coverage audit — NOT CFG batching.

> ## Round 7 SHIPPED (2026-07-11, `44.81-local`) — S2V multi-group WanDitOps port: steps **4.15 → 3.44 s/step (17%)**
> The W4 shared-final-layer port + the G>1 `ConditionTimeGroups` drain kill, S2V-verified (full details in
> `E2E_PARITY_WORKLOG.md` round 7). The "audio injector host glue" framing was STALE — the injector has been
> GPU-resident since 07-02; the real glue was `WanDitOps` G>1: per-group host copies of temb/proj (2G stream
> drains/forward) + `FinalLayer` full-hidden D2H+CPU-modulate+H2D (×2 CFG/step). Port = per-group M=1 GEMMs
> kept bit-exact + device `Concat` gather; device FinalLayer via SliceRows/AffineBroadcastLastDim/AddScalar/
> LayerNormNoAffine rank-3 (host fallback when G∤S). Parity: all stages through block_39 **bit-identical**;
> velocity relL2 2.5e-4 (LN order × fp8 bucket edges, 10× under the fp8 noise floor). TI2V/Matrix-Game inherit.
> **W9 trim RESOLVED — trim stays default-on:** A/B showed trim ON 3.44 s/step / 68 async-OOM retries vs
> trim OFF 4.23 / 169 — beside a near-capacity pool the per-step trim PREVENTS allocation-retry stalls
> (`WAN_S2V_TRIM=0` knob kept). Swarm 49f 480²: 2.38 min gen (was 2.66), gens md5-identical, sound muxed;
> T2V 30.04 s warm (no regression); flagships PASS. NEXT S2V lever: `WanS2VLoader` has no warm cache (gen 2 =
> full pipeline rebuild + TE re-encode; transplant round-6 `WanVideoCacheEntry`).

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

> ### ✅ Phase 4 (audio half) SHIPPED (2026-07-11, 44.77-local) — audio decode **1.7 s → 0.13 s (~13×)**;
> warm same-prompt Swarm gens **41.7/40.9 → 39.1/38.7 s**. All remaining audio-tail host loops GPU-ported with
> EXISTING ops (new shared helper `LtxAudioDeviceOps`, zero backend/kernel edits): vocoder anti-alias
> replicate-pad → Transpose2D + SliceRows(first/last row) + one contiguous Concat + Transpose2D back (no
> per-channel copies); crops → SliceLastDim; MRF residual sums + resblock chain → `backend.Add` (Clone gone,
> ownership-tracked); mel flatten → batched-transpose `Permute0213(s=frames,h=bins,d=1)`; the BWE input's
> transpose-then-flatten proved an IDENTITY relabel of the log-mel memory → eliminated (MelSpectrogram emits
> `[1,C·mel,frames]` directly, unit-tested); STFT input relabel via device `Scale(…,1f)` (no waveform drain);
> tail add+clamp+crop → SliceLastDim+Add+Clamp; audio VAE `LtxAudioPixelNorm` → `WanRmsNormChannel` with eps
> folded `eps′=sqrt(C·eps)` (exact in signal+silence limits, relL2 ≤1e-6 at test magnitudes); causal top-pad →
> Permute0213+Fill+Concat(contiguous)+Permute0213; upsampler first-row drop → Permute0213+SliceRows chain.
> Only remaining host math: the once-per-clip magnitude+mel matmul (tiny, one small drain) and the final PCM
> read. Parity: `LtxAudioDevicePortParityTests` (10 theories/facts, pre-port loops verbatim, EXACT except
> pixel-norm ≤1e-5 relL2) + Swarm A/B vs 44.76 same-seed: video frames BYTE-IDENTICAL, audio 48 kHz stereo
> present, waveform cos=0.9999 / rms+peak dB within 0.03 dB (residual = AAC re-encode + documented eps fold);
> 3 same-prompt gens md5-identical. Probes now: audio VAE 24 ms · vocoder 101 ms. Phase 4 fully closed
> (video 44.74 + audio 44.77).
>
> ### ✅ Phase 4 (video-VAE half) SHIPPED (2026-07-11, 44.74-local) — video VAE decode **18.5 s → 0.77 s (24×)**,
> PASSES, coherent (cat-in-garden seed-42, 25f 512×320 + audio; e2e test wall 97 s). All decode-tail host loops
> GPU-ported with EXISTING ops, zero backend/kernel edits: resnet + norm_out `ChannelRms` → `WanRmsNormChannel`
> (gamma-null; timestep-conditioned v1 path kept host because `ApplyShiftScale` mutates the host buffer in place);
> upsampler pixel-shuffle → batched `Permute0213` adjacent-group-swap chain + one `SliceRows` for the leading
> (st0−1)-temporal-frame drop (identity swaps skipped per stride); `RepeatChannels` → `Concat(dim:1)`; residual
> add → `backend.Add`; final `PixelUnshuffle` → existing `UnpatchifyVae` (its `oc=c·p²+r·p+q` channel unpack is
> exactly LTX-2's `(c·p+pa)·p+pb` layout — verified + unit-tested). Parity: new
> `LtxVaeDevicePortParityTests` (all 4 22B stride geometries EXACT-equal vs the pre-port host loops on CPU);
> real-weight CUDA-vs-CPU diag decode max px diff 1/255 (GEMM rounding). LTX v1 (0.9/0.9.5/0.9.7) upsamplers/
> resnets inherit the win. Remaining in this phase (probes: now minor): host `Denormalize` (once, tiny latent,
> boundary), audio VAE + vocoder 1.5 s total (`LtxAudioPixelNorm`, host pads, mel matmul, MRF sums).
>
> **Phase 4 — decode-tail ports (original plan).** Video VAE `ChannelRms`→`WanRmsNormChannel` (drop-in, the exact Krea2 fix),
device Denormalize/PixelUnshuffle/upsampler; audio VAE `LtxAudioPixelNorm`→`WanRmsNormChannel`, device pads;
vocoder mel matmul → `backend.Linear`, MRF sums → `backend.Add`, pad/crop → device. Probes decide priority
inside this phase.

> ### ✅ Phase 5b (resident-prefix persistence, KEEP_MODELS) SHIPPED (2026-07-11, 44.76-local) — prefix
> pinned at max across gens, re-upload gone (engine roundtrip: preload+prime 7.8 s → 0.15–0.24 s on gen 2+).
> `LtxVideo2Pipeline` keeps the shared weights + the VRAM-sized block prefix device-resident across
> generations (the Flux `HARTSY_KEEP_MODELS` idiom, default ON): post-loop frees ONLY the streamed suffix's
> lingering cache entries + `TrimMemoryPool` (the VAE decode ran in 7.7 GB free beside the kept 12-block
> prefix, 0.64 s, zero OOM). The prefix COUNT is sized once and pinned (`_residentPrefixBlocks`), killing the
> 12→10→9 pool-slack drift; geometry growth (token load > sized) releases + re-sizes. A prompt-cache MISS
> measures whether the ~14 GB Gemma fits beside the prefix (it doesn't on 24 GB: free 5.6 GB < TE 14.1 GB +
> 2 GB margin → logged eviction), re-uploads after the TE frees at the pinned count — squeezed to what fits
> THIS gen if the auto-promoted VAE/vocoder residency tightened VRAM (observed 12→9 squeeze), and the next
> generation TOPS BACK UP to the pinned max (gen-4 roundtrip leg asserts byte-identical frames vs gen 3).
> Model-switch eviction: pipeline `DisposeCore` frees the prefix eagerly (plus the extension's
> `MakeRoomForLoad`→`FreeAllDeviceMemory` wipe when the cache empties). Engine 4-gen roundtrip (25f
> 512×320/20st seed-42, gen-4 top-up leg + byte-identity assert, PASSED): gen-2 HIT **45.0 s** (was 48.3 s),
> gen-3 MISS squeeze 12→9, gen-4 HIT top-up→12 + frames byte-identical to gen-3. **Swarm 44.76-local:**
> gen 1 miss 61.5 s → same-prompt gens 2-3 **41.7 / 40.9 s** (was 70.1 s in Phase 5 — target <70.1 ✓),
> all three mp4s md5-identical; prefix constant **15 (persistent, no re-upload)** with preload+prime
> 164/152 ms and free-for-decode stable 9382→9386 MB (drift GONE); different-prompt miss 47.6 s
> (squeeze 15→13), next HIT 41.1 s topped back up to 15, mp4 md5-identical to the miss gen across the
> different prefix split. LTX→Z-Image switch: `FreeAllDeviceMemory: free 14591 → 22549 MB`, no OOM;
> flagships Z-Image-Turbo 2.83 s / Krea2-Turbo 4.55 s PASS. Remaining in Phase 5: nothing — Phase 4 audio
> half and Phase 3 F16 are the open LTX-2 items.
>
> ### ✅ Phase 5 (Gemma prompt cache) SHIPPED (2026-07-11, 44.75-local) — cache HIT skips the whole TE phase
> (Gemma encode ×2 + connectors + ~12 GB TE weight upload, 5–22 s/gen), PASSES, byte-identical output.
> `LtxVideo2Pipeline` prompt-embedding cache (the FLite/Flux2 pattern): all four paired-CFG embeddings
> (video/audio × pos/neg) cached under one (posTokens,negTokens) key; on miss: encode → `FreeWeights(gemma)` →
> host-materialize all four (a never-host-read tensor loses its only copy in FreeActivations) →
> `FreeActivations`+`TrimMemoryPool` (also gives the resident-prefix sizing the reclaimed VRAM) → cache;
> on hit: skip TE entirely, embeddings re-fault to device on first block use. Streaming/pinned-ring lifecycle
> untouched. Engine e2e roundtrip (25f 512×320/20st seed-42, `LTX2_CACHE_ROUNDTRIP=1` in
> `LtxVideo2_Gpu_T2VA_ShortClip`): gen 1 miss TE 7.4 s; gen 2 HIT wall **48.3 s** with frames BYTE-IDENTICAL
> to gen 1; gen 3 different-prompt miss 53.0 s (TE 4.9 s), prompt-faithful (red-car sunset clip). Noticed:
> resident prefix drifts 12→10→9 across gens (pool slack) — small; KEEP_MODELS prefix persistence still open.
>
> **Phase 5 — pipeline parity + (conditional) graph (original plan).** Gemma prompt-embedding cache keyed on tokens (12B encode
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

> ### ✅ I2V STRUCTURAL BUG FIXED (44.12-local, Swarm-verified): the cond-latent must be the causal-VAE encode
> of the WHOLE padded pixel clip (init + mid-gray frames), not first-frame + zero latents (`BuildCondClip` +
> full-latent `BuildI2VCondition`). Swarm-API gen 1349001: real galloping motion faithful to the init.
> ### ✅ I2V FULL AUDIT COMPLETE (2026-07-08 night) — DiT PARITY-PROVEN, root causes enumerated
> **Layer-by-layer parity vs a faithful ComfyUI-math torch oracle (real fp8 ckpt, identical inputs,
> `tests/python-reference/i2v_reference/`): ALL stages ≤6.5e-3 relL2** (patch_embed 1.8e-4, temb 3.7e-3,
> textProj 2.3e-3, imgProj 8.1e-4, blocks 0/20/39 = 1.5e-3/6.5e-3/4.0e-3). The C# Wan I2V transformer —
> incl. the k_img/v_img dual cross-attention — is numerically CORRECT. UniPC scheduler formula-audited
> against the reference: faithful. Reference specs (diffusers/Comfy/sd.cpp, code-quote level) captured in
> the research-agent outputs; key confirmations: our cond construction now matches all three (padded-clip
> causal-VAE encode, mask-first concat, image-tokens-first context, shared-query summed img attention).
> **Why the engine harness showed "all noise" while Swarm showed real video:** harness input defects, not
> engine bugs — (1) it fed ~460 GARBAGE umT5 pad rows of 512 (unzeroed; the documented drowns-the-prompt
> failure; FIXED — test now zeroes pads like the extension), (2) synthetic gradient init is off-distribution
> for CLIP conditioning, (3) sub-native res. Rule: judge I2V quality ONLY via Swarm gens with real inputs.
> **Measured fp8-native cost on small conditioning GEMMs:** temb 5.7%→0.4%, textProj/imgProj similar when
> HARTSY_FP8_NATIVE=0 → exclude tiny GEMMs (M·N below threshold) from the fp8-native path (quality item).
> Remaining open (quality/perf, not correctness): fp8-CFG collapse ≥cfg5 (renorm insufficiency vs Comfy),
> comb-texture polish, I2V ~200 s vs T2V 37 s overhead, harness natural-init + WAN_CFG knob.
>
> **Quality attribution COMPLETE (official-spec runs, 2026-07-08 late):**
> • VAE exonerated: real-image encode→decode roundtrip is pixel-faithful (new `WanVae_Roundtrip_Quality`
>   harness, `Output/wan_vae_roundtrip_*`).
> • Blur/halos = RESOLUTION under-spec: at native 0.4 MP (Image Aspect, Model Res) frame 0 is tack-sharp
>   (gallery 1553001); at 512×320 (40% area) everything is soft with comb-edge halos. Official diffusers
>   recipe: 81f / cfg 5 / ~50 steps / native area; our FlowShift 3.0 for 480p already matches.
> • Flat-collapse = fp8-CFG DC-bias integration: cfg 5 at native res → frames 1+ collapse to a flat colour
>   field (renorm 0.7 insufficient); cfg 3.5/2.0 keep real motion (1349001/1406001). Comfy handles cfg 6 on
>   the SAME fp8 checkpoint → our fp8 CFG path has a correctable residual bias (velocity DC/renorm work).
> • OOM fix shipped 44.13: the (correct) whole-clip conditioning encode needs headroom — pipeline now evicts
>   the resident DiT pre-encode when free VRAM is short (loader preloads DiT before the pipeline runs).
> • Sweet-spot probe in flight: native res + cfg 3.0. Practical recipe pending; extension should default
>   fp8 Wan I2V cfg ≤3.5 (or land stronger renorm) + prefer native-area video resolution.
> Residual (open): engine
> I2V harness needs WAN_CFG knob + natural init (its cfg-5+gradient regime shows noise even when Swarm is
> healthy); I2V wall ~250 s vs T2V 37 s (conditioning/overhead probes). History below.
>
> ### I2V/S2V round (2026-07-08, engine 44.9-local deployed) — earlier state (superseded above)
> **Wan I2V-14B produces garbage beyond frame 0 — at the ENGINE level, independent of Swarm.** The engine
> e2e (synthetic gradient init, cfg 5, renorm 0.7) yields blotch noise on EVERY frame, and
> `AssertFramesCoherent` PASSES on it (only catches flat/degenerate) — so the 07-02 "validated" status never
> verified I2V visually; it has plausibly never worked. The I2V test now DUMPS FRAMES (`Output/wan_i2v_*`).
> Through Swarm, frame 0 looks perfect (real-photo conditioning anchors it) which masked the bug in previews.
> Debug log (2026-07-08 afternoon): knob matrix ALL-ELIMINATED — identical deterministic noise with
> solver-order 1, fp8-native off, cuDNN off. **Conditioning fix SHIPPED** (reference-faithful: VAE-encode the
> whole padded pixel clip [init + mid-gray frames] via causal `IWanVaeEncoder.Encode` → full-latent
> `BuildI2VCondition`; the old single-frame encode left latent frames 1+ literal ZEROS on 16/36 input
> channels — a genuine deviation from diffusers/Comfy, kept regardless) — but noise persists.
> **Attribution correction:** engine T2V-14B ALSO fails the coherence assert (near-flat/black) at BOTH
> f596cef AND 8027968 (bisect run) — NOT a new regression from the wan-grind commits; the engine-test
> regime (detector cfg 5 + rescale 0.7 @ 512×320) sits in the DOCUMENTED fp8-CFG low-res darkening zone,
> while the same model at cfg 6 through Swarm is fine. Next: engine T2V-14B probe at WAN_CFG=6 (queued on
> GPU-quiet); if clean, re-judge I2V at matched cfg, then layer-diff `GenerateImageToVideoConcat` vs
> diffusers `WanImageToVideoPipeline` (clone `tests/python-reference/s2v_reference`) as the definitive tool.
> The I2V test now honors frame dumps; consider adding a `WAN_CFG` override to it (T2V/loader tests have one).
> Side facts: the 44.8→44.9 ctx cache DID fix a real cross-step conditioning-corruption bug (FreeActivations
> drops device buffers of never-host-read tensors — general rule stands), and `FreeBackendMemory` via the
> Swarm API works for freeing held models between engine tests.
> **I2V perf**: 200 s warm vs T2V-14B's 37 s at the same size/steps — ~160 s of I2V-specific overhead
> (init VAE-encode, concat-conditioning build, per-step mask?) → needs phase probes in
> `GenerateImageToVideoConcat` (next target).
> **S2V trim REVERTED**: `FreeActivations(trimPool:false)` OOM'd mid-run (40 blocks × 13824 FFN) even with
> FP8_NATIVE — the audio injector's churn needs the per-step pool trim until its host glue is ported.
> **Wan 14B T2V at cfg 6 (fp8)** shows rainbow glitch patches through Swarm — the known fp8-CFG amplification;
> extension should clamp/renorm CFG for fp8 Wan (detector CfgRescale exists engine-side; verify it engages
> through the extension path).

> ### ✅ Wan warm-path residency round (2026-07-11, 44.80-local) — same-prompt+image I2V warm **52.6 → 31.9 s**
> (mp4 md5-IDENTICAL to the 44.78 seed-42 baseline), different-prompt warm 52.6 → **39.7 s**; T2V inherits.
> Three cross-generation caches + KEEP_MODELS DiT residency:
> (1) **umT5 prompt cache** (extension `WanVideoLoader`, the LTX/Flux pattern): zero-padded embeds cached per
> (pos,neg) token key on the `WanVideoCacheEntry`; HIT skips the encode AND the ~8.4 GB umT5 upload. MISS
> decides DiT coexistence from MEASURED free VRAM (umT5 8.4 GB + 2 GB never fits beside the resident 16.4 GB
> fp8 DiT → logged eviction, denoise re-uploads at 1.9-3.4 s).
> (2) **CLIP image cache** keyed on SHA-256 of the raw init-image bytes (2.4 GB CLIP-ViT-H fits beside the
> resident DiT when it does miss — measured gate).
> (3) **I2V conditioning cache** (`WanVideoPipeline`): the `[mask, cond-latent]` tensor cached per
> (init/last-frame hash, geometry) — a same-image repeat skips the whole-padded-clip VAE encode ENTIRELY.
> This matters doubly: the encode's REAL conv peak is **~7.5 GB at 25f 512×320 (measured OOM 44.79: consumed
> 6 GB then requested 1.5 GB more beside the kept DiT)** — ~153 F32 copies/frame, not the old 24-copies
> estimate (which only ever ran against a freed DiT). `EnsureVaeEncodeHeadroom` now uses ×160 + trims the
> pool BEFORE reading free VRAM (pool slack made the old check pessimistic).
> (4) **KEEP_MODELS DiT residency** (`ReleaseOrKeepTransformer`): single-expert DiT stays device-resident
> post-gen unless measured free < the decode estimate `max(3 GB, f·h·w·160)`; warm DiT preload **2.2 s → 0 ms**
> (logged `DiT preload: 0 ms`). MoE always frees (two 14 GB experts never co-reside); LoRA gens free the
> resident base up front; `DisposeCore` frees on model switch (Wan→T2V switch verified: free 22.58 GB at next
> load, no OOM). V2V whole-clip encode gained the same measured guard.
> Warm all-HIT profile: TE/CLIP/cond ≈ 0 · DiT preload 0 · steps 27.4 s · decode 3.8 s (+mux) = **31.9 s**;
> peak VRAM 23.0 GB (unchanged). Next levers: steps 27.4 s (Axis-B fp8 transient dequant, batched/graphed CFG),
> decode 3.8 s, T2V same treatment inherited free.
>
> ### ✅ Wan I2V ~200s overhead SOLVED (2026-07-11, 44.78-local) — warm **234/218 s → 52.6/51.9 s (4.3×)**,
> steps **13.1 → 1.82 s/step (7.2×, the T2V floor)**, seed-42 frames BYTE-IDENTICAL to the pre-fix engine.
> The "~160 s of I2V-specific overhead" was NOT conditioning/setup — it was the CLIP-image branch of
> `WanVideoBlock.CrossAttention` (the ONLY block path T2V never executes): (1) **host `AddInPlace`
> pointer-loop summing the two [sq,dim] cross-attn branch outputs** — a full D2H of BOTH ~92 MB tensors +
> CPU add + re-upload per block per forward (×40 blocks ×2 CFG ≈ 15 GB/step of drained PCIe traffic that
> also serialized the async pipeline); (2) **host `SliceRows`** building fresh host text/img context rows
> per block per forward → device cache MISS → full stream drain ×80/forward. Fixes (bit-exact, existing
> ops): `backend.SliceRows` device slice + `GatedResidualLastDim`(ones) device add (`AddRows`); plus the
> I2V loop gained the T2V loop's per-step `FreeActivations()` (rope/ctx caches are host-materialized —
> safe) and `[wan-phase]` probes (cond-encode / DiT-preload / per-step / VAE-decode). Bench (25f 512×320,
> 15 steps, cfg 3.5, `wan2.1_i2v_480p_14B_fp8_scaled`, real init image, seed-42 A/B + 2 warm): cold 259 →
> 65.2 s; peak VRAM 23.2 → 20–23 GB; real motion from the init verified (fox turns + trots toward camera).
> WanAnimate shares the block path and inherits the fix. Warm phase profile NOW (the next levers, in
> order): cond VAE-encode+evict 7.4 s (whole-padded-clip encode + resident-DiT evict/re-upload cycle every
> gen), TE/CLIP/mux ~11 s (umT5 re-encode per gen — the LTX prompt-cache pattern applies), DiT preload
> 2.2 s, steps 27.4 s (T2V-parity floor: Axis-B fp8 transient dequant + graph territory), decode 3.8 s.
> Flagship gate PASS: Z-Image-Turbo 2.77/2.79 s, Krea2-Turbo 4.50/4.51 s, both viewed pristine;
> Wan→Z-Image eviction clean (FreeAllDeviceMemory 17.4 → 22.6 GB, no OOM).

## Plan — rest of fleet (interleave as capacity allows; ordered by value)

1. **Quick wins, one deploy each:** ~~S2V per-step `TrimMemoryPool` (W9)~~ **RESOLVED round 7: trim is a WIN,
   stays default-on (3.44 vs 4.23 s/step; `WAN_S2V_TRIM=0` knob)**; Wan RoPE memoization
   (copy `Kandinsky5Rope` pattern, W1); Wan text/img projection cache across steps (W2); temb drain fix (W3,
   the proven Kandinsky fix — **G>1 half DONE round 7**, G=1 shipped earlier); HunyuanVideo VAE per-tile
   `Sync()` + host feather-blend → device accumulate (V1).
2. **Shared final-layer port** (W4 + H1): ~~`WanDitOps.FinalLayer`~~ **DONE round 7 (G=1 was already device;
   G>1 device path shipped, S2V-verified, TI2V/Matrix-Game inherit)**; `HunyuanVideoDit` tail (H1) still open.
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
