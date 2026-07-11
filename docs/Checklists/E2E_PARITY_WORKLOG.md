# Image-model e2e + parity grind — session worklog

## ▶▶ PERF+CORRECTNESS PASS QUEUE (user directive 2026-07-10 pt4) — proven fleet wins per new model

Playbook order per model: glue-scan → GPU-residency block port (copy QwenImageBlock/Lumina2Block idioms)
→ device RoPE → cuDNN SDPA (allowF16 where QK-normed, D∈{64,128,256... D=120 gate widened}) → TE prompt
cache → drain-free CfgEulerStep loop → KEEP residency → wall-clock phase probes before op-level work →
step graph last. After EACH: visual verify + warm bench + scoreboard + flagship gate (Z<3.2s, Krea2<6.5s).

**Methodology per model (user directive pt5):** correctness by LAYER-BY-LAYER / BLOCK-BY-BLOCK numerical
diff against a Python reference (the proven loop: env-gated stage dumps in the engine — copy the
`OMNIGEN2_DEBUG_DIR`/`HIDREAM_DEBUG_DIR`/`HunyuanImageDebugDump` pattern — vs a hand-written torch/numpy
reference fed the engine's EXACT dumped inputs; localize first divergence by relL2, fix, repeat until
corr≈1.0). Perf to conclude with the FULL kit including per-step CUDA GRAPH capture (the
`DitStepGraph`/Flux2 fixed-buffer recipe; watch the 2 capture-invalidators: pageable re-upload of
host-materialized caches + sync DtoD copies; FreeActivations resets the graph slot; graph pool needs
`cuDeviceGraphMemTrim` on reset). Iterate each model until fully working AND as fast as achievable.

**Python references to diff against:** F-Lite = `github.com/fal-ai/f-lite` (or diffusers FLitePipeline)
run in the ComfyUI venv on CPU with streamed fp16 weights (the Mistral/HyVAE oracle pattern);
Zeta-Chroma = lodestones' Zeta-Chroma repo sampling code (x0-parameterization!) + Z-Image reference for
the shared trunk; Radiance = lodestones Chroma-Radiance reference (pixel head); HunyuanImage/Flux.2 Dev
are already visually correct — perf-only.

**VIDEO ARC round 11 — Wan-Animate CHECKERBOARD SOLVED (44.84-local, 2026-07-11).** The ~16 px halftone
tile (period = one DiT token = 2×2 latent patch × 8× VAE) was a **dropped fp8 weight scale**, not an
Animate-arch bug. `CheckpointConvertUtils.ApplyFp8ScaledDequant:354` folded the `.scale_weight` companion
into `Fp8ScaleFactor` only when the companion was **F32** — but the Kijai `wan2.2_animate_14B_fp8_scaled_KJ_v2`
checkpoint ships them **BF16** (all earlier-validated Wan fp8 ckpts use F32, which is why only Animate
broke). Guard dropped all 514 scales (~0.13–0.46) → raw fp8 weights (±448) ran ~5× hot → stage dump
(`HARTSY_ANIMATE_DUMP`) showed block-0 out exploding **rms 0.17 → 4.4e5** (→2.7e6 by block-1), collapsing
every token to the dominant direction (head interTokenStd/rms **0.005**) = identical patches = the tile.
Fix = read the scalar as F32 for F32/F16/**BF16** companions (F32 branch byte-identical → **zero regression**
to F32-scale Wan variants; confirmed t2v/i2v/s2v_14B all F32). Post-fix: block-0 rms **0.72**, head
interTokenStd/rms **0.92**, tile gone. Also chunked `WanAnimateTransformer.BuildMotion` to comfy's
`encode_bs=8` (the all-frames 512² StyleGAN stack OOM'd beside the resident 14B DiT). **Swarm real-input
e2e (17f 480²) completes → smooth checkerboard-free video.** Flagships on 44.84: Z-Image 2.82 s, Krea2
4.52 s; Wan T2V ti2v-5B regen coherent. LESSON: fp8 scale-companion guards must accept BF16/F16 scalars
(the Zeta/F-Lite BF16-as-float* class, here in the converter). Diagnostics kept: `HARTSY_ANIMATE_DUMP`,
`HARTSY_ANIMATE_NO_POSE/NO_FACE`. Patches: `scratchpad/video_arc_round11.patch` (+extension pin note).

1. **F-Lite — ✅ CORRECT + FAST (44.66-local, 07-11).** Coherent reference-quality astronaut @1024²/30st
   (verified vs a same-recipe python GPU reference gen — flite_pygen.py in scratchpad). Single gen 67.9s
   (1.9s/step, was 90s/step + AV crash). THE bugs (found via byte-dump oracles, in order): (1) reference
   T5 conditioning = NO attention mask + zeros-tensor negative (not encoded-empty); (2) dynamic-shift
   alpha uses the LATENT grid (128² → alpha 4 @1024²), not the token grid; (3) **register_tokens is BF16
   read as float*** (16 garbage tokens → bubble grid); (4) **lambda_param is BF16 read as float*** (wrong
   V-residual mix in blocks 1..39 — found because block0 matched the oracle at 0.002 while block10 jumped
   to 0.093, and block 0 is the only mix-free block). t×1000 is CORRECT (the reference caller scales; the
   embedding fn takes raw t — do not remove again). REMAINING: warm gen #2 fails in PreloadWeights (T5
   re-preload beside the resident 20GB DiT) → needs the Flux2Pipeline TE⇄DiT staging + prompt cache =
   also the perf lever (~10s T5 re-encode/gen); then warm bench + scoreboard row. Block-39 residual drift
   vs python ~0.34 relL2 = F16-SDPA compounding (image-quality unaffected; retest allowF16:false if ever
   suspected). OLD notes: GPU-residency port shipped: block
   AdaLN/modulate/gated-residual + attention slice/permute/head-norm + device ApplyRope (F-Lite's rotation
   = llama split-half with NEGATED sin → `FLiteRope.BuildDeviceTables` emits [c|c]/[−s|−s] full-dim tables,
   applied pre-permute) + allowF16 SDPA + cached lambda. **Result: 90s/step → 1.9s/step (47×), AV crash GONE,
   completes 30 steps in 68s.** Real bug found+fixed via reference diff: `ComputeTimeEmbedding` scaled t
   ×1000 but fal-ai `timestep_embedding` takes t RAW in [0,1] (kept — correct per reference). **REMAINING:
   output still FLAT GRAY.** Probes (`HARTSY_FLITE_PROBE=1`, 44.57): context absmax 2.04 / temb 21.98 /
   x-embed rms 1.00 all sane; block0-out absmax 438; blocks-out rms 17.3; **velocity rms 0.449 ≈ 3× weak**.
   VAE scaling verified correct (0.3611/0.1159 = the file's config). NEXT = the block-0 oracle: engine-dump
   x/context/temb/rope + block0 out (add a FLite dump env like HunyuanImageDebugDump), run fal-ai
   `f_lite/model.py` DiTBlock-0 on CPU from the fp16 shards (references already fetched:
   scratchpad `flite_ref.py` + `flite_model.py`), relL2-bisect. PRIME SUSPECTS in order: (a) rope
   convention vs `TwoDimRotary` (verify their attention's rotate form + our negated-sin equivalence
   numerically); (b) cross-attn fused `context_kv` chunk order (we slice K=[0,H), V=[H,2H)); (c) the
   vPrev/lambda residual-V mixing; (d) register-token prepend order vs `extend_with_register_tokens`;
   (e) CFG combine; (f) final-layer modulate. Old glue-scan text follows for reference: FLiteBlock cpu=24,
   FLiteAttention cpu=13, FLiteRope cpu=19 (HOST rope), FLiteTransformer cpu=40 — the full un-ported
   profile; expect Lumina2-class gains (10-30×). Also no allowF16 SDPA anywhere.
2. **Chroma1-Radiance — ✅ PERF PASS DONE (44.73-local, 07-11): 36.9 → 2.65 s/step (14×), warm 54.4s
   @1024²/20st.** The cost was NOT the token count (4096 img tokens = classic Chroma @1024²) — it was
   the NeRF head's host tile loop (~50k tiny Linears + host loops per forward). GPU-ported with existing
   ops + ForwardPaired shared embed/patchify/modTable + prompt cache + DiT residency + drain-free
   CfgEulerStep loop. Parity corr 0.99996 vs old engine (4-step A/B). See the dated section below.
   Next levers: F16 head/backbone (damp-kit audit), Radiance step graph, fp8 requant of the 19 GB BF16.
3. **Zeta-Chroma — ✅ VERIFIED CLEAN (44.70, 07-11).** Confetti = BF16 `dec_net` in_ln read as host
   float* in the decoder head (see the 07-11 section below); sampling recipe was already correct.
4. **HunyuanImage 2.1 — TE cache + token-space loop SHIPPED (44.71, 07-11): 77 → 74.1s @2048².**
   Remaining perf levers: F16 stream absmax probe (DiT block-input probe), cuDNN SDPA engagement check
   at 4096+1000 tokens — the 40-forward CFG loop is the wall now. ByT5 glyph branch still unwired.
5. **Flux.2 Dev** (2.64 s/step eager): step-graph capture OOMs on 24GB → eager; needs a capture-friendly
   memory plan (or partial-graph). ~~GGUF-eviction model-switch NRE~~ **FIXED + VALIDATED 07-11 (44.71)**
   — Dev→Krea2→Dev in one process, no restart, no NRE/finalizer lines.

**PTX toolchain flag (surface to the LLM session):** commit `9762b94` regenerated `lm_f32.ptx` as PTX ISA
9.2 (CUDA 13.2 toolchain) — this box's driver JIT tops out at 9.0 → backend init failed fleet-wide until
the `.version` header was downpatched 9.2→9.0 in BOTH copies (native/cuda/lm/ + src/HartsyInference.Cuda/Ptx/).
No 9.2-only instructions were present. Future PTX regens must target ≤9.0 or ship SASS.

---


## 2026-07-11 — VIDEO ARC round 10: Wan T2V step-floor profile → **COMPUTE-FLOOR EARLY-BAIL** (no code shipped, `44.82-local` unchanged)

**Mission:** kill Wan's per-step host `scheduler.Step` latent drain with a device Euler/UniPC step (drain-free
loop), then CUDA-graph-capture the resident DiT — OR early-bail if the profile shows the step is already at the
fp8 compute floor. **Verdict after profiling: EARLY-BAIL — Wan T2V is genuinely compute-bound on fp8 tensor-core
GEMM + fused flash attention. No host drain worth killing, and graph capture would not help. No engine edit.**

**The decisive experiment — per-op stream-sync leaves the step time UNCHANGED.** Relaunched Swarm with
`HARTSY_PROFILE=1 HARTSY_PROFILE_SYNC=1` (a `cuStreamSynchronize` after *every* op) and ran `wan2.1_t2v_14B_fp8_scaled`
(25f 512×320, 15 steps, cfg 3.5, seed 42). Steps ran at **~1.79 s/step — identical to the un-profiled 1.82 s
baseline** (step 1 @16:36:42 → step 15 @16:37:07 = 25 s / 14 intervals). Forcing a full drain after each of the
~2500–3000 kernel launches per step added **~0 ms**. This is only possible if (a) there is **no async overlap being
exploited** (the step is already a serial chain of dependent GPU-compute kernels) and (b) **kernel launch/host
overhead is negligible** vs the big-kernel execution. Both are the textbook signatures of a compute-bound loop —
and both are exactly what would make CUDA-graph capture (which only removes launch overhead) a no-op here.

**Op breakdown (`HARTSY_PROFILE_SYNC`, true GPU time, cumulative over a cold 4-step gen = umT5 encode + 4 steps +
VAE decode; `scratchpad/wan_t2v_profile_round10.txt`):**

| op | calls | total_ms | avg_ms | note |
|---|---|---|---|---|
| **Linear** | 3412 | **4930** | 1.445 | native fp8 tensor-core GEMM — the floor |
| **SDPA** | 664 | **2003** | 3.017 | cuDNN fused flash attn (mask-null, F16) — already optimal |
| H2D_MISS_BIG | 303 | 1335 | 4.407 | **one-time cold-start**: umT5 8.4 GB encode+upload + DiT 16.4 GB preload (NOT per-step) |
| Permute0213 | 2560 | 408 | 0.159 | patchify/rope layout |
| GatedResidual | 2880 | 362 | 0.126 | AdaLN gated residual (device) |
| RmsNorm | 1329 | 252 | 0.190 | QK-norm |
| Gelu | 346 | 250 | 0.722 | FFN |
| AffineBroadcast | 968 | 221 | 0.228 | AdaLN shift/scale (device) |
| LayerNormNoAffine | 968 | 217 | 0.224 | pre-norms |
| RopeInterleaved | 640 | 135 | 0.211 | 3D RoPE apply (device) |
| H2D_MISS_SMALL | 2313 | 36 | 0.016 | tiny cache re-faults |
| Silu | 16 | 0.3 | 0.017 | — |

Linear + SDPA = **~68% of all GPU time** and an even larger share of the *step* (H2D is cold-start, not per-step —
confirmed from the phase log: `umT5 prompt cache MISS — encode+free 8889ms` + a cold DiT preload of the resident
16.4 GB fp8 DiT; `free 22779 MB` before the loop = DiT not yet on device). The AdaLN/norm/rope/permute glue is
**already device-resident** (rounds 5–7 ports) and sums to only ~0.4 s/step (~22%).

**Step 1 (host drain) — measured negligible, NOT worth killing.** The host `scheduler.Step` (UniPC) +
`CfgCombineRenormInPlace` read the latent + both velocities to host each step, but the latent is tiny
(16·7·40·64 = 287 K floats ≈ 1.1 MB; D2H+H2D+CPU-UniPC ≈ a few ms/step, <0.3% of 1.79 s). The per-op-sync result
already proves it: if the host round-trip were stalling the pipeline, forcing sync everywhere would not have left
the step time unchanged. **No device-Euler port shipped** — it would save a few ms and carries fp8-CFG numeric
risk (UniPC is a stateful predictor/corrector; round-9 showed this path is sensitive), for no graph-capture payoff.

**Step 3 (graph capture) — would not help, so not attempted.** Graph capture removes *launch* overhead; the
per-op-sync experiment shows launch overhead is already ~0% of the step. A captured graph over a 14B fp8 DiT would
also fight the resident-DiT VRAM and bake the per-forward host-materialized cache re-faults (cos/sin/encoderProj —
graph capture-invalidators). Zero upside, real cost. Same class of outcome as the Flux.2 Dev graph opt-out.

**Step 4 (fp8 GEMM coverage audit) — CLEAN, no fallbacks, no fix needed.** Native fp8 is default-on
(`[Cuda] perf flags: … NativeFp8Gemm=True`), and the fallback guards in `CudaBackend.cs:511-515` are `K % 16 == 0
&& (N · outElemBytes) % 16 == 0`. Every Wan-14B GEMM shape clears both: block dims are 5120 (attn q/k/v/o, FFN out)
and 13824 (FFN inner), both ≡ 0 (mod 16); N·4 (F32 out) = 20480 / 55296, both ≡ 0 (mod 16); the embedders/final-proj
(K = 64/256/4096, N = 5120/64) also clear. **The profile table contains NO `Cast`/dequant-to-F16 op** — direct
confirmation that zero fp8 weights are recast to F16 on the native path (the only per-GEMM fp8 cost is the cheap
memory-bound activation absmax+quant, folded into `Linear`). No surgical guard-widening is warranted; the shared
kernel is left untouched (an image flagship must not regress for a Wan win that does not exist).

**Conclusion:** Wan T2V at 1.82 s/step is the fp8 tensor-core compute floor for this DiT on a 4090. The remaining
theoretical lever is F16 *activations* (`HARTSY_DIT_F16`) to trim the ~0.4 s/step of norm/glue traffic — a minority
of the step, numerically risky on the fp8-CFG-sensitive Wan path, and out of scope for a step-floor round. No code
changed; `44.82-local` stands. Correctness unchanged (no edit; gen produced a valid 25-frame seed-42 mp4 at the
expected 1.79 s/step, matching the round-5/6 verified baseline). Patch: `video_arc_round10.patch` (empty). No deploy
→ no flagship gate (round-9 precedent). Next lever if ever revisited: F16-activation A/B behind the existing flag
with per-stage relL2, expected ≤~15% step win at best.


## 2026-07-11 — VIDEO ARC round 9: batched-CFG step-floor investigation → **documented NO-GO** (no code shipped, `44.82-local` unchanged)

**Mission:** fuse the CFG pair (cond+uncond) into ONE batch-2 forward to stream the weights once and ~2× the
step floor. **Verdict after profiling both targets: no-go — the premise does not hold for either model, and
the change would break fp8 exactness for ~zero real win.** This is a valid documented result (the mission's
own timebox rule): a stream-floor win that corrupts output is a fail, and there is no stream floor to win here.

**Finding 1 — LTX-2.3 ALREADY captures the stream-once win, bit-exactly (`LtxVideo2Transformer.ForwardCfgPair`,
lines 325-332).** The block loop runs cond then uncond **back-to-back per block** (`_blocks[i].Forward(...ctxC)`
then `...ctxU`), with the `BeforeBlockForward` streaming hook uploading each streamed block's weights ONCE
before both consume it. This is block-level CFG interleave: streamed weights hit the bus once/step (the win the
mission wanted) while cond and uncond stay **separate M=sv GEMMs, each with its own per-tensor activation quant
→ bit-identical to two Forwards** (the notes' "proven bitwise-identical"). Merging them into one M=2·sv GEMM
would add nothing to streaming (already once/step) and would BREAK the per-tensor quant exactness (Finding 3).
LTX has no remaining CFG-batching win.

**Finding 2 — Wan's DiT is FULLY RESIDENT, not streamed → the mission's premise is factually wrong for Wan.**
`WanVideoPipeline` `PreloadWeights(_transformer.EnumerateWeights())` loads the whole 16.4 GB fp8 DiT and
KEEP_MODELS pins it (round-6: "the resident 16.4 GB fp8 I2V DiT"); `WanVideoTransformer.ForwardCore` has **no
`BlockStreamingController`/`CudaStreamingWeightCache`/streaming hook** (grep = NONE — that machinery is LTX/Flux
only). So there is no per-forward weight stream to fetch once. The two sequential forwards (`WanVideoPipeline.cs`
228-229 / 238-239 / 367-369 / 373-375) re-read the *resident* fp8 weights from VRAM, and at S=4480 tokens
(25f 512×320: gt·gh·gw = 7·20·32) the GEMMs are large-M compute-bound — the resident weight read is amortized
over M, so it is not the bottleneck. Batching to M=8960 is FLOP-identical (2·S·N·K either way) and gives no
GEMM speedup. Wan's 1.82 s/step floor is fp8 tensor-core **compute**, which B=2 cannot reduce. (The recurring
"Axis-B fp8 transient dequant" framing in the notes is loose: on Ada `EnableNativeFp8Gemm` defaults ON
(`CudaBackend.cs:232`), so fp8-scaled Wan weights are consumed DIRECTLY by fp8 tensor cores staying packed —
`CudaBackend.cs:505-529` — there is no per-call fp8→F16 weight recast on the native path; the only remaining
per-GEMM cost is the cheap memory-bound activation absmax+quant.)

**Finding 3 — fp8 activation quant is PER-TENSOR (the round-7 trap, confirmed at kernel source).**
`native/cuda/dequant/fp8_quant.cu`: `absmax_f32`/`absmax_finalize_scale` reduce over the ENTIRE activation
tensor `n` → ONE dequant scale (amax/448) consumed via `B_SCALE_POINTER`. Batching cond+uncond into one
`[2S, dim]` activation makes both samples share one absmax → cond's fp8 quantization changes vs its solo run
(NOT bit-identical). Prior in-repo measurement on this exact path (round 7, batching the timestep MLP to one
M=G GEMM): **temb relL2 3.5e-2** — 17× over the documented ~2e-3 fp8 noise floor. Making it exact would require
a segmented (per-sample) absmax in the shared CUDA fp8 kernel — a numeric-risk change to the path every fp8
image model rides — for a Wan win that Finding 2 already shows is ~zero.

**Conclusion:** no code changed; `44.82-local` stands. Wan's real step lever is **CUDA-graph capture** (it is
fully resident = graph-capturable, unlike streaming LTX; blocked today by the host-side `scheduler.Step`
latent drain — needs a device Euler/UniPC step first) + a native-fp8-path coverage audit (ensure every Wan
GEMM meets the K%16 / 16-byte-ldc alignment so none falls to the F16 recast). LTX is already optimal on this
axis. Patches: `video_arc_round9.patch` (empty — no engine edit). No deploy → no flagship gate needed.

## 2026-07-11 — VIDEO ARC round 8: S2V warm cache (gen 2 **2.39 → 2.04 min**, md5-exact) + MatrixGame2 testhost-crash root cause (`44.82-local`)

**Item 1 — S2V warm cache (round-7 finding (a)).** The "Model cache MISS on gen 2" root cause was NOT a
missing cache: `HartsyInferenceBackend.IsCached` only probed `TryGetWanVideo` for the shared Wan compat
classes, so a cached S2V/VACE/Animate entry was invisible → every repeat gen rebuilt the full pipeline
(~9 s DiT convert+load + VAE + Wav2Vec2 + umT5) and re-encoded everything. Fixed (probe all four Wan
variant slots) + transplanted the round-6 `WanVideoCacheEntry` pattern into `WanS2VLoader`, all three
phases logged HIT/MISS with `EnsureEncoderHeadroom` measured-VRAM TE⇄DiT staging on every miss:

- **umT5 prompt cache** per (pos,neg) token key (round-6 pattern verbatim; a HIT skips the 8.4 GB upload +
  9.2 s encode).
- **Wav2Vec2 stacked-feature cache** keyed on SHA-256 of the decoded 16 kHz waveform — caches
  `EncodeAllLayers` `[T50, layers, dim]` (frame-count-INDEPENDENT; the cheap host 50 Hz→16 fps resample
  runs per gen), a HIT skips the Wav2Vec2 upload+encode (1.8 s). Loader now drives
  `GenerateFromAudioFeatures` directly instead of `GenerateFromWaveform`.
- **Reference-latent cache** keyed on SHA-256(init image) + target res via new engine
  `WanS2VPipeline.EncodeReferenceImage` (hoisted from the ref-image branch) + a caller-owned
  `referenceLatent` param on `GenerateFromAudioFeatures` (0.7 s VAE encode skipped on HIT).

**Bench (44.82-local, round-7 recipe: jfk.wav + Z-Image portrait 1123001, 49f 480², 15 st, cfg 2, seed 42):**
gen 1 all-MISS 2.46 min; gen 2 Model cache HIT + umT5/audio/ref HIT → **2.04 min** (round-7 gen-2 baseline
2.39 min; steps 7.27 s/step ×15 + decode now dominate). Gen 1, gen 2 AND round-7's 1127001 mp4 are all
**md5-IDENTICAL** (`d9204313…`) — the cache path is byte-exact vs the pre-change engine. Frames VIEWED
(talking head articulating, identity-faithful); h264 480² 16 fps 3.06 s + AAC 16 kHz speech-level
(mean −15 dB). **Regressions clean:** T2V warm 30.5 s (baseline 30.0, mp4 md5 == round-7 1131001), I2V warm
32.0 s (baseline 31.9). **Flagship gate PASS:** Z-Image-Turbo 2.78/2.80 s, Krea2-Turbo 4.52 s warm, both
VIEWED pristine; video→Z-Image eviction clean.

**Item 2 — MatrixGame2 testhost crash (round-7 finding (b), merge blocker) ROOT-CAUSED + FIXED.**
`Transformer_Forward_RespondsToContextActionsAndClip` malloc-corruption was **latent since the test was
introduced (cb0f1fe)** — bisect via throwaway worktrees proved it fails at cb0f1fe/028013a/b314a87/HEAD, so
it is NOT rounds-5/6 fallout. Root cause: `MatrixGame2Transformer.ProjectClipContext` (a private duplicate
of the Wan I2V MLPProj) hardcoded the intermediate buffer to `[L, clipDim]` while `backend.Linear` derives N
from the WEIGHT — the synthetic fixture ships `ff.net.0.proj` as `[dim=24, clip=16]`, so the GEMM wrote
L×24 floats into an L×16 heap block (the OmniGen2 FFN-overflow bug class; invisible until glibc's next
consistency check, hence "testhost crashed"). Fix: deleted the duplicate and wired the shared shape-driven
`WanImageEmbedder` (inner dim from `ff.net.0.proj.Shape[0]`; handles both the fixture layout and the real
1280→1280→dim checkpoints, plus BF16→F32 affine casts and pos_embed). Test passes standalone; all 7
Diffusion MatrixGame tests, 2 Interactive MatrixGame2 tests, 26 Wan Video CPU tests, 8 WanDitOps/WanVideo
tests green (no DupUp3D flake observed this run).

**Next levers:** S2V steps (7.27 s/step at 12.6k tokens) are now ~89% of gen-2 wall — fp8 requant of the
per-GEMM transient dequant (Axis-B) or batched/paired CFG is the next S2V win; FramePackMotioner extend
still TODO. Patches: `video_arc_round8.patch` (engine) + `video_arc_round8_extension.patch` (scratchpad).

## 2026-07-11 — VIDEO ARC round 7: S2V multi-group host-glue port — steps **4.15 → 3.44 s/step (17%)**, single-forward parity bit-identical through block 39 (`44.81-local`)

**Plan framing was stale:** the "audio injector host glue" (round-6 assessment, `WanS2VPipeline` comment) was
already gone — `WanS2VAudioInjector` has been fully GPU-resident since `c30cf43` (07-02), and W8's "S2V
AppendRows" is a device `Concat`. Code audit found the REAL remaining per-step S2V host glue in the
**multi-group (G>1) branches of `WanDitOps`** — the branches only S2V (ref-image ⇒ G=gt+refT), TI2V and
Matrix-Game take:
- `ConditionTimeGroups` G>1: per-group `Buffer.MemoryCopy` of the GPU Linear outputs (a DataPointer read of an
  op result = full stream drain, ×2G ≈ 40/forward) AND left temb/proj host-side for every downstream consumer.
- `FinalLayer` G>1: drained the FULL final hidden D2H (~123 MB @480×320, mid-forward sync + activation
  eviction), CPU LayerNorm+modulate, re-upload for proj_out — ×2 CFG forwards per step. This churn is what the
  per-step `TrimMemoryPool` was papering over.

**Port (existing IBackend ops only):** `ConditionTimeGroups` keeps the per-group M=1 GEMM loop EXACTLY and
gathers with one device `Concat` per output; `FinalLayer` G>1 (guard `g·tokensPerGroup == s`, host fallback
kept) = `SliceRows` table rows + ones-row `AffineBroadcastLastDim` adds + `AddScalar(+1)` +
`LayerNormNoAffine` into rank-3 `[G,tokens,dim]` + per-group `AffineBroadcastLastDim` modulate + `Linear`.
**Rejected variant (lesson):** batching the timestep MLP to one M=G GEMM changed the fp8-native PER-TENSOR
activation-quant scales (temb relL2 3.5e-2 vs baseline) — never change GEMM grouping on the fp8-native path
when bit-parity matters.

**Parity:** new `WanDitOpsMultiGroupTests` (CPU, Unit tier): ConditionTimeGroups multi-group == per-group G=1
path bit-exact; FinalLayer device path == host reference ≤1e-4. GPU old-vs-new dump A/B (`WAN_DEBUG_DIR`,
1 step, default fp8-native profile, real S2V ckpt, 480×320/33f/seed 42): **every stage through block_39
bit-identical (rms=0)** incl. temb/timestep_proj/audio-injector deltas; velocity relL2 2.5–2.8e-4,
cfg_combined 4.4e-4 — the FinalLayer LN kernel's reduction order crossing fp8 quant-bucket edges, 10× below
the documented fp8 noise floor (~2e-3). Multi-step dumps drift to ~2e-2 by step 2 via cross-step fp8 requant
amplification (expected; same class as the fp8-CFG sensitivity).

**Trim verdict — trim STAYS default-on (flip attempted, reverted on data):** 8-step harness A/B
(480×320/33f, 15.7 GB resident DiT, Swarm holding 4.8 GB): baseline 4.14–4.17 s/step (169 async-OOM
retries) → port + trim ON **3.44 s/step / 68 retries (17%)**; port + trim OFF 4.23 s/step / 169 retries —
beside a near-capacity pool the per-step trim PREVENTS allocation-retry stalls, it is not merely an OOM
band-aid. `WAN_S2V_TRIM=0` knob kept. Peak VRAM 24 020 MB in all three runs (pool grows to card capacity).
`HARTSY_FP8_NATIVE=0` hard-OOMs BOTH builds at this ambient VRAM (config unusable here, not a trim artifact).

**Swarm validation (44.81-local, backend 2):** S2V jfk.wav + Z-Image portrait, 49f 480², 15 steps, cfg 2,
seed 42 ×2 — gen 1 wall 143 s (2.38 min gen, 7.36 s/step at the 12.6k-token production geometry; prior
production wall 2.66 min), gen 2 2.39 min, **both mp4s md5-IDENTICAL** (deterministic path). Frames VIEWED:
identity-faithful talking head, mouth articulating across frames; ffprobe: h264 480² 16 fps 3.06 s + AAC
16 kHz stereo, muxed audio is speech-level RMS and correctly trimmed from the 11 s source. **T2V regression
clean:** `wan2.1_t2v_14B_fp8_scaled` 25f 512×320/15st seed 42 → 1.74 s/step, warm 30.04 s (round-6: 1.82 /
30.3 s), fox clip coherent. **Flagship gate PASS:** Z-Image-Turbo 3.04 s warm (9st/cfg1 1024²), Krea2-Turbo
4.49 s warm (8st/cfg1 1024²), both VIEWED pristine; video→Z-Image eviction clean.

**Found, not fixed (pre-existing):** (a) `WanS2VLoader` has NO warm cache — gen 2 logs `Model cache MISS`
with the same model resident and rebuilds the whole pipeline (~9 s load + TE re-encode per gen); the round-6
`WanVideoCacheEntry` pattern is the next S2V lever. (b)
`MatrixGame2ModelTests.Transformer_Forward_RespondsToContextActionsAndClip` crashes the testhost (malloc
corruption) at HEAD too — verified by swapping baseline `WanDitOps` back in; likely interacts with the
uncommitted rounds-5/6 `WanVideoBlock` work.

## 2026-07-11 — VIDEO ARC round 6: Wan warm-path residency — same-prompt+image I2V **52.6 → 31.9 s**, T2V warm **~37 → 30.3 s** (`44.80-local`)

**Plan items** (round-5 "next levers"): umT5 prompt cache (~11 s/gen) + skip the per-gen resident-DiT
evict→cond-encode→re-upload cycle (7.4 + 2.2 s). Three cross-generation caches + KEEP_MODELS DiT residency:

- **umT5 prompt cache** (extension `WanVideoLoader`, the LTX/Flux pattern): the two zero-padded host embed
  tensors cached on `WanVideoCacheEntry` per (pos,neg) token key — a HIT skips the encode AND the 8.4 GB
  umT5 upload. A MISS decides DiT coexistence from MEASURED free VRAM (`EnsureEncoderHeadroom`: trim pool →
  weight-bytes + 2 GB margin vs free): 8.4 GB umT5 never fits beside the resident 16.4 GB fp8 I2V DiT →
  logged eviction, the denoise re-uploads at 1.9–3.4 s; on a fresh card it logs "fits" and skips the evict.
- **CLIP image cache** keyed on SHA-256 of the raw init-image bytes (embed depends only on the image; the
  2.4 GB CLIP-ViT-H fits beside the resident DiT when it does miss — same measured gate). LoRA gens free
  the resident base DiT up front (merged transformer can't coexist).
- **I2V conditioning cache** (`WanVideoPipeline`): the `[mask(4), cond-latent(16)]` tensor (~1.4 MB host)
  cached per (init/last-frame SHA-256, w×h×frames) — a same-image repeat skips the whole-padded-clip VAE
  encode entirely. **The 44.79 OOM that forced this design:** the encode's REAL conv peak is ~7.5 GB at
  25f 512×320 (consumed 6 GB then requested 1.5 GB more beside the kept DiT) ≈ 153 F32 copies/frame — the
  old `×24` "empirical ceiling" had only ever run against a freed DiT. `EnsureVaeEncodeHeadroom` now uses
  ×160 and trims the pool BEFORE the free-VRAM read (prev-gen pool slack made it pessimistic — that slack,
  not the DiT, is what the old unconditional evict was really clearing). V2V encode gained the same guard.
- **KEEP_MODELS DiT residency** (`ReleaseOrKeepTransformer`): the single-expert DiT stays device-resident
  post-gen unless measured free < decode estimate `max(3 GB, f·h·w·160)` — warm `DiT preload: 0 ms` (was
  2.2 s + the 7.4 s evict cycle). MoE experts always free; `DisposeCore` frees the resident DiT + caches on
  model switch (I2V→T2V switch measured: next load starts at free 22.58 GB, no OOM).

**Bench** (Swarm 44.80-local, `wan2.1_i2v_480p_14B_fp8_scaled`, 25f 512×320/15st/cfg 3.5, fox init, seed 42,
backend 2): gen 1 miss 64.6 s (= round-5 cold parity), gen 2 same-prompt+image (all three caches HIT, log
lines confirmed) **31.9 s** — profile TE/CLIP/cond ≈ 0 · preload 0 ms · steps 27.4 s (1.82 s/step flat) ·
decode 3.8 s; gen 3 different-prompt (umT5 MISS + CLIP/cond HIT) **39.7 s**. Peak VRAM 23.0 GB (unchanged).
**gen-1, gen-2 AND the 44.78 round-5 seed-42 baseline mp4s are md5-IDENTICAL (183a3cad…)** — the whole
residency+cache path is byte-exact. Frames VIEWED: fox turns + trots toward camera (the known seed-42
motion); gen-3 prompt-faithful (fox sits still in gentle snowfall). **T2V** `wan2.1_t2v_14B_fp8_scaled`
same geometry: gen 1 (cold, model switch) 60.6 s → gen 2 all-HIT **30.3 s** (warm baseline ~37 s — inherits
the cache + residency free), both mp4s md5-identical, golden-hour fox clip coherent with real motion.

**44.79 lesson (one repack):** gen-2 OOM'd mid-cond-encode with all caches HIT because the ×24 encode
estimate said "fits" beside the kept DiT — the engine recovered gracefully (request errored, gen 3
completed after the measured TE evict). Fixed by the conditioning cache (skips the encode) + the ×160
estimate (a true cond miss now evicts correctly).

**S2V trim (plan item 3): assessed, not taken.** The per-step `TrimMemoryPool` is already default-on and
REQUIRED (the 07-08 `trimPool:false` experiment OOM'd; `WAN_S2V_TRIM=0` knob exists) — the real fix is
porting the audio injector's host glue, a full port job outside this round's timebox.

**Flagship gate PASS** (post-deploy): Z-Image-Turbo and Krea2-Turbo warm within gates, viewed coherent;
Wan→Z-Image eviction clean. Deploy: engine `44.80-local` (15 nupkgs) + extension pin bump + forced
extension rebuild; `[Cuda] perf flags` verified on relaunch.

## 2026-07-11 — VIDEO ARC round 5: Wan I2V-14B warm **234/218 s → 52.6/51.9 s (4.3×)**, steps 13.1 → 1.82 s/step (`44.78-local`)

**Plan item** (GENPERF: "I2V ~200 s vs T2V 37 s overhead — needs phase probes in `GenerateImageToVideoConcat`,
next target"). Picked over the S2V trim (smaller win) and Animate checkerboard (oracle campaign, heavier).

**Diagnosis** (code audit confirmed by baseline probes): the ~160 s of "I2V-specific overhead" lived in the
CLIP-image branch of `WanVideoBlock.CrossAttention` — the only block code T2V never executes:
- `AddInPlace(flat, flatImg)` host pointer-loop summed the text and image cross-attn branch outputs: full D2H
  of BOTH `[4480, 5120]` ≈ 92 MB tensors + CPU add + re-upload on next use, per block per forward
  (×40 blocks ×2 CFG ≈ 15 GB/step of synchronous PCIe round-trips, each one draining the async stream).
- `SliceRows` host `Buffer.MemoryCopy` produced FRESH host text/img context tensors per block per forward →
  first device use = cache MISS = full stream drain + pageable H2D (×80/forward, the sync-H2D disease).
- The I2V loop also lacked the T2V loop's per-step `Backend.FreeActivations()`.

**Changes** (existing ops only, zero kernel/backend edits):
- `src/HartsyInference.Diffusion/Models/Denoisers/DiTBlocks/WanVideoBlock.cs`: `SliceRows` → device
  `backend.SliceRows`; `AddInPlace` → `AddRows` (`GatedResidualLastDim` with the ones-gate, exact 1·x add).
  WanAnimate shares this path and inherits.
- `src/HartsyInference.Video/Pipelines/WanVideoPipeline.cs` (`GenerateImageToVideoConcat`): per-step
  `FreeActivations()` (rope/ctx caches host-materialized — safe) + `[wan-phase]` probes (cond-encode /
  DiT-preload / per-step Verbose / VAE-decode).

**Bench** (Swarm, `wan2.1_i2v_480p_14B_fp8_scaled`, 25f 512×320, 15 steps, cfg 3.5, real 512×320 fox init,
backend 2, quiet GPU; scratchpad `bench_i2v.py`): 44.77 baseline cold 259.0 s / warm 234.4, 218.3 s @
13.1 s/step, peak 23.2 GB → **44.78 cold 65.2 s / warm 52.6, 51.9 s @ 1.82 s/step (flat)**, peak 20–23 GB.
**Seed-42 frames BYTE-IDENTICAL to the 44.77 output** (4 sampled frames, max px diff 0) — exact port.
Real motion from the init verified on seed 42 (fox turns + trots toward camera) and a warm seed.

**Warm phase profile now** (the next levers): cond VAE-encode incl. resident-DiT evict 7.4 s → could keep the
VAE encoder staged / skip the evict-reupload cycle when headroom allows; TE/CLIP/mux ~11 s → umT5 prompt
cache (the LTX/Flux pattern); DiT preload 2.2 s/gen (paid because the cond encode evicts); steps 27.4 s =
the T2V floor (Axis-B fp8 transient dequant + step graph territory); VAE decode 3.8 s.

**Flagship gate PASS** (post-deploy): Z-Image-Turbo **2.77/2.79 s**, Krea2-Turbo **4.50/4.51 s**, both viewed
pristine astronaut-on-horse; Wan→Z-Image switch eviction clean (`FreeAllDeviceMemory: 17436 → 22580 MB`,
no OOM). Deploy note: `pkill -f SwarmUI` matches your own shell if the command line contains the Swarm PATH
(cd "<swarm>" in the same compound command) — pkill pattern must target the dotnet process, then relaunch
in a separate command.

## 2026-07-11 — VIDEO ARC round 4: LTX-2.3 audio decode **1.7 s → 0.13 s (~13×)** — GENPERF Phase 4 (audio half) shipped (`44.77-local`); Phase-3 F16 absmax probe recorded

**Plan item** (Phase 4 remainder: "audio VAE + vocoder 1.5 s total — `LtxAudioPixelNorm`, host pads, mel
matmul, MRF sums"). **Diagnosis:** same disease as the video half — the vocoder's two BigVGAN generators run
~110 anti-aliased SnakeBeta activations each, and every one did TWO host replicate-pads + a host crop between
GPU convs (~440 full-tensor D2H drains/clip), plus host MRF sums (`AddInPlace` ×18), resblock `Clone`s, host
mel flatten/transpose, and the audio VAE's host pixel-norms/causal-pads/row-drops.

**Changes** (existing IBackend ops only, zero kernel/backend edits; new shared helper
`src/HartsyInference.Diffusion/Models/Music/LtxAudioDeviceOps.cs`):
- **Replicate-pad-time** (`AntiAlias.Up/Down`, `Resampler`): Transpose2D → SliceRows(first/last time row) →
  ONE Concat whose repeated edge parts share a single device buffer (contiguous dim-0 copies, no per-channel
  loop) → Transpose2D back. **Crop** → one `SliceLastDim` (time is innermost).
- **Vocoder** (`LtxBigVganGenerator`): MRF fusion accumulates via `backend.Add` into the first resblock's
  output; resblock residual chain drops `Clone` (ownership-tracked, `Add(h2,h2,cur)`); pre-STFT `PadRight` →
  `Fill`+`Concat`.
- **`LtxAudioVocoder`**: mel flatten → batched-transpose `Permute0213(s=frames, h=bins, d=1)` (batch inferred
  = C·bins — exact layout match, unit-proven); **the BWE input's transpose-then-flatten is an IDENTITY relabel
  of the log-mel memory** (proven in `BweInputBuild_IsIdentityRelabelOfLogMelMemory`) → MelSpectrogram now
  emits the generator-ready `[1, C·mel, frames]` directly; STFT input `[1,C,S]→[C,1,S]` relabel via device
  `Scale(…, 1f)` (same contiguous bytes, kills the waveform drain); tail `clamp(residual+skip)`+crop →
  `SliceLastDim`+`Add`+`Clamp`. Only host math left: the once-per-clip magnitude + mel matmul (tiny).
- **Audio VAE** (`LtxAudioVaeDecoder`): `LtxAudioPixelNorm` → `WanRmsNormChannel` with **eps folded
  `eps′ = sqrt(C·eps)`** (reference `x/sqrt(mean_C+eps)` = `x·√C/sqrt(sumSq+C·eps)`; the backend's
  `x·√C/max(L2,eps′)` matches EXACTLY in both the signal limit — rel diff ≤ eps/(2·mean_C(x²)) — and the
  silence limit `x/sqrt(eps)`); causal top-pad → Permute0213(time-outer)+Fill+contiguous Concat+Permute back;
  upsampler first-time-row drop → Permute0213+SliceRows(rowOffset=c)+Permute back.

**Parity** — new `LtxAudioDevicePortParityTests` (22 cases incl. theories, CPU, pre-port host loops copied
verbatim): all EXACT-equal except pixel-norm (relL2 ≤ 1e-6 at signal magnitudes, tol 1e-5). Existing
`LtxVocoderTests` pass on the new path. Engine e2e `LtxVideo2_Gpu_T2VA_ShortClip` PASSED (1 m 44 s, real 22B).

**Swarm `44.77-local`** (25f 512×320/20st seed-42, backend 2, same prompt as round 3): warm same-prompt gens
**39.1 / 38.7 s** (was 41.7/40.9 — ≤41 gate beat); audio decode probe **1.7 s → 129 ms** (audio VAE 24 ms ·
vocoder 101 ms); 3 gens md5-IDENTICAL; **video frames BYTE-IDENTICAL to the 44.76 output** (video path
untouched, f01/f13/f24 PNG-equal); audio 48 kHz stereo present, waveform vs 44.76: **cos = 0.99990, rms/peak
dB within 0.03 dB** (residual = AAC re-encode of both + the documented eps fold), listened character identical
(soft garden ambience). Flagship gate PASS: Z-Image-Turbo **2.82 s**, Krea2-Turbo **4.55 s**, both viewed
coherent; LTX→Z-Image switch eviction clean (`FreeAllDeviceMemory: free 14622 → 22580 MB`, no OOM).

**Phase-3 F16 verdict (probe numbers recorded, implementation deferred):** new env-gated first-forward
residual-stream probe `HARTSY_LTX2_PROBE=1` in `LtxVideo2Transformer.ForwardCfgPair` (the F-Lite/HiDream
idiom). Real-weight results @512×320 step 1: video stream grows to a **15k–18.2k absmax plateau over blocks
37–45 (peak 18,201 @ block 42, rms ~20)**; audio stream peaks 1,989 (rms ~61 @ block 47); proj_in ~18/8.7.
No stream rides >60k → F16 does NOT hard-overflow (gate passes), but headroom to F16-max 65,504 is only
~3.6× at ONE geometry/prompt — a naive flip is low-margin, and LTX-2 blocks have plain (non-sandwich) norms
so the exact Chroma/Z-Image 1/32 residual-damp trick needs the no-affine-LN cancellation audit first. Wall
math at the validated geometry also caps the win: steps are weight-stream-bound (~1.87 s ≈ 24.3 GB ÷ ~13 GB/s
+ tail) and activations already fit (prefix 15/48 with 7.5 GB spare for decode) — F16 activations shrink
neither. The F16 lever pays at bigger geometries (activation-VRAM-bound) or combined with a residency win;
carry these probe numbers into that round.

## 2026-07-11 — VIDEO ARC round 3: LTX-2.3 resident-prefix persistence (KEEP_MODELS) — warm gens 70.1 → **41 s** (`44.76-local`)

**Plan item** (Phase 5 leftover: "prefix drifts 12→10→9 across gens; KEEP_MODELS prefix persistence").
**Change** (`src/HartsyInference.Video/Pipelines/LtxVideo2Pipeline.cs`): the Flux `HARTSY_KEEP_MODELS`
idiom adapted to the hybrid resident-prefix + streamed-suffix DiT — post-loop no longer
`FreeWeights(ALL)`: shared weights + the block prefix stay device-resident across gens (next gen's
`PreloadWeights` = cache-hit no-op), only the streamed suffix's lingering cache entries are freed +
`TrimMemoryPool` hands the streaming window's VRAM to the VAE decode. Prefix COUNT sized once and PINNED
(`_residentPrefixBlocks`) — kills the pool-slack drift; geometry growth (token load > sized) releases +
re-sizes. Prompt-cache MISS decides TE coexistence from MEASURED free VRAM (Gemma 14.1 GB never fits
beside the prefix on 24 GB → logged eviction), re-uploads after the TE frees — squeezed to what fits that
gen (the auto-promoted VAE/vocoder residency tightens VRAM vs the virgin gen-1 sizing), pin kept, next gen
TOPS BACK UP. Decode safety valve: evict prefix pre-decode if free < max(3 GB, grid-scaled estimate).
Model switch: new `DisposeCore` frees the prefix eagerly (+ existing `MakeRoomForLoad`→`FreeAllDeviceMemory`).

**Validation** — engine 4-gen roundtrip (`LTX2_CACHE_ROUNDTRIP=1`, now with a gen-4 top-up leg + byte-identity
assert; 25f 512×320/20st seed-42, PASSED): gen 1 sizes prefix 12, kept (`kept across generations: 12 blocks,
free 7.7 GB for decode`, VAE 0.64 s, zero OOM); gen 2 HIT `resident prefix 12 (persistent, no re-upload)`,
preload+prime **7.8 s → 0.15 s**, wall 48.3 → **45.0 s**; gen 3 MISS evicts for TE (measured: free 5.6 GB <
14.1+2 GB), re-uploads squeezed 12→9, pin kept; gen 4 HIT tops back up to 12 (preload 0.30 s), wall 44.9 s,
frames BYTE-IDENTICAL to gen 3 (Assert). Swarm `44.76-local` (25f 512×320/20st seed-42, backend 2):
gen 1 miss 61.5 s → gens 2-3 same-prompt **41.7 / 40.9 s** (target <70.1 ✓, equal ✓), all three mp4s
**md5-identical**; prefix log constant `15 (persistent, no re-upload)`, preload+prime 164/152 ms, free-for-
decode stable 9382→9386 MB (drift GONE); gen 4 different-prompt miss 47.6 s (squeeze 15→13 logged), gen 5
HIT 41.1 s topped back up to 15 + mp4 md5-identical to gen 4 ACROSS the different prefix split. Frames
viewed coherent + prompt-faithful (cat garden / red car sunset); mp4a audio atom present. **Model-switch
eviction PASS**: LTX→Z-Image logged `FreeAllDeviceMemory: free 14591 MB → 22549 MB`, no OOM. Flagship gate
PASS: Z-Image-Turbo warm **2.83 s** (≤3.2), Krea2-Turbo warm **4.55 s** (<6.5), both 1024²/8st astronaut
outputs viewed coherent. Audio half (Phase 4) not started — out of timebox after the squeeze/top-up
iteration; still the next Phase-4 item (audio VAE norms/pads + vocoder mel/MRF, ~1.4 s).

## 2026-07-11 — VIDEO ARC round 2: LTX-2.3 Gemma prompt-embedding cache — GENPERF Phase 5 shipped (`44.75-local`)

**Plan item** (Phase 5: "Gemma prompt-embedding cache keyed on tokens — 12B encode is expensive, TE freed
every gen"). **Change** (`src/HartsyInference.Video/Pipelines/LtxVideo2Pipeline.cs`): the FLite/Flux2
prompt-cache pattern, LTX-flavored — the paired-CFG denoise consumes FOUR embeddings (video/audio ×
pos/neg), so all four are cached under one (posTokens,negTokens) key. Miss path: `TrimMemoryPool` →
encode ×2 → `Sync`+`FreeWeights(gemma)` → host-materialize all four (`DataPointer` read — a
never-host-read tensor loses its only copy in `FreeActivations`, the Wan ctx-cache rule) →
`FreeActivations`+`TrimMemoryPool` (bonus: the resident-prefix sizing right after now sees the reclaimed
Gemma VRAM) → stash. Hit path: skip the whole TE phase; embeddings re-fault to device on first block use.
The Phase-2 streaming/pinned-staging-ring lifecycle is untouched (cache section ends before
`PreloadWeights(shared)`). The end-of-gen `enc*.Dispose()` calls removed — cache owns them now (freed on
the next miss). RoPE table cache is host-built, so the new `FreeActivations` is safe for it (verified).

**Validation** (engine e2e `LtxVideo2_Gpu_T2VA_ShortClip` + new env-gated `LTX2_CACHE_ROUNDTRIP=1`
3-gen roundtrip; real 22B fp8 + Gemma-3-12B fp8, 25f 512×320/20st seed 42, 4090):
- gen 1 (cold miss): TE(Gemma)+connectors+free **7.36 s**, steps ~2.09 s, video VAE 0.63 s, audio 1.45 s.
- gen 2 (same tokens): logs `prompt cache HIT — skipping Gemma encode`, wall **48.3 s**, frames
  **BYTE-IDENTICAL** to gen 1 (cmp on frame 12) — the cache is exact, and step 1 also dropped 7.1→2.2 s
  (no cold prime). Viewed: coherent cat-walking-through-garden motion.
- gen 3 (different tokens): miss path re-encodes (TE 4.9 s), wall 53.0 s, prompt-faithful red-vintage-car
  coastal-sunset clip with strong motion. Viewed.
- Swarm-deployed as `44.75-local` (extension pin bumped, forced rebuild, `[Cuda] perf flags` verified).
  Swarm API 25f 512×320/20st seed-42 mp4s (audio muxed, AAC stereo 48 kHz confirmed via ffprobe):
  gen 1 miss **84.2 s** (TE 7.66 s) → gen 2 same-prompt **70.1 s** with `prompt cache HIT` in the Swarm log,
  identical scene + identical audio stats (mean −40.0 / max −16.9 dB); gen 3 different-prompt **70.9 s**
  (TE re-encode 3.3 s warm), prompt-faithful red-car clip with DIFFERENT audio (engine rumble, mean
  −12.4 dB) — the audio-connector embeddings re-encode correctly on miss. All frames viewed coherent.
- Flagship gate PASS (post-deploy): Z-Image-Turbo warm **2.79 s** (≤3.2), Krea2-Turbo warm **4.45 s**
  (<6.5), both 1024²/8st snow-leopard outputs viewed coherent.
Observed (next lever): resident prefix drifts 12→10→9 across sequential gens (pool slack accumulates);
KEEP_MODELS-style prefix persistence across gens (skip per-gen prefix re-upload) is the remaining Phase-5
item. Audio tail now: audio VAE 0.14 s + vocoder ~1.2 s (Phase 4 audio half still open — out of this
round's timebox; `LtxAudioPixelNorm`/causal-pad/upsample host loops + vocoder mel matmul/MRF sums are the
targets, same existing-ops approach as the video half).

## 2026-07-11 — VIDEO ARC round 1: LTX-2.3 video-VAE decode **18.5 s → 0.77 s (24×)** — GENPERF Phase 4 (video half) shipped (`44.74-local`)

**Plan item** (VIDEO_GENPERF_PLAN.md's own post-Phase-2 ordering: "VAE decode 18.5 s = NOW the biggest
single phase"). **Diagnosis:** the whole LTX decode tail was host `DataPointer` loops interleaved between
GPU convs (the literal Krea2-VAE disease): per-resnet `ChannelRms` ×2 (~36 calls, tensors up to ~130M
elements), upsampler `PixelShuffle`/`RepeatChannels`/raw-pointer residual add ×4 stages, decoder-final
`ChannelRms` + `PixelUnshuffle` at full res — each one a D2H sync + H2D re-upload.

**Fix — GPU-ported with EXISTING IBackend ops only (zero backend/kernel edits):**
- `LtxVaeResnetBlock3d.cs`: non-timestep `ChannelRms` → `WanRmsNormChannel` (gamma-null, same math to float
  rounding). Timestep-conditioned (v1 0.9.5/0.9.7) path deliberately stays host: `ApplyShiftScale` mutates
  the host buffer in place — done to a GPU-cached tensor that's the stale-device-copy bug.
- `LtxVaeUpsampler3d.cs`: pixel-shuffle → batched `Permute0213` adjacent-group-swap chain (≤6 swaps,
  identity swaps skipped per stride) + one `SliceRows` for the leading (st0−1)-temporal-frame drop;
  `RepeatChannels` → `Concat(dim:1)` (identical block layout); residual add → `backend.Add`. B>1 keeps host.
- `LtxVideo2VaeDecoder.cs`: final norm → `WanRmsNormChannel`; final `PixelUnshuffle` → existing
  `UnpatchifyVae` — the Wan channel unpack `oc = c·p² + r·p + q` (q→H, r→W) is EXACTLY LTX-2's
  `(c·p + pa)·p + pb` layout (verified index-by-index + unit-tested). Host `Denormalize` kept (tiny latent,
  once, already at the host boundary).

**Validation:** new `LtxVaeDevicePortParityTests` — all four 22B upsampler geometries (spatio-temporal
8/up2 + 8/up1, temporal 2/up2, spatial 4/up2) EXACT-equal vs the pre-port host loops (copied verbatim as
test references); unpatchify exact; RMS ≤1e-5. Real-weight `LtxVideo2_VaeDecode_CudaVsCpu_Diagnostic`:
max px diff 1/255. GPU e2e (25f 512×320, 20 steps, seed 42): **video VAE decode 773 ms (was
18,100–18,500 ms)**, steps ~2.0 s, e2e test wall **97 s**, frames VIEWED coherent (cat advancing through a
sunlit garden, real temporal motion), audio decoded (1.5 s, untouched path). LTX v1 (0.9/0.9.5/0.9.7)
decoders inherit the upsampler/resnet wins automatically.

**Deploy + flagship gate (44.74-local):** `[Cuda] perf flags` line verified, Backend #2 live on the 4090.
Z-Image-Turbo warm **2.74 s** (≤3.2 ✓), Krea2-Turbo warm **4.46 s** (<6.5 ✓), both VIEWED coherent. No
shared image paths touched (LTX-only files). Swarm-side LTX-2.3 gen also verified live (see plan doc).

**Next levers (per plan):** Phase 4 audio half (audio VAE `LtxAudioPixelNorm` + host pads, vocoder mel
matmul + MRF sums — 1.5 s total); Phase 5 Gemma prompt cache (TE ~11–22 s/gen is now the biggest one-shot
phase); Phase 3 F16 activations → bigger resident block prefix (steps still stream-bound ~2 s).

## 2026-07-11 — Chroma1-Radiance perf pass: 36.9 → 2.65 s/step (14×), warm **54.4s** @1024²/20st (`44.73-local`)

**Profile-first (HARTSY_PROFILE relaunch, 4-step wall probes):** step wall 35.5-36.5s, T5 encode 10.4s
per gen, backbone = the already-resident Chroma ForwardCore (~1.5s of it). The other ~34s/step (95%) was
the **NeRF head host-glue tile loop**: per forward, 128 host tiles × 4 depths × 32 patches × (host
TransposeNormalizeChunk ×3 + 3 tiny `backend.Linear` + host silu/residual loops) ≈ **50k tiny Linear
calls + host loops per forward, ×2 for CFG** — plus host `X0Prediction.ToVelocity`, host
`scheduler.Step`, and per-gen T5⇄DiT re-upload churn (19 GB BF16 DiT + 5 GB T5 every gen).

**Fixes (all existing backend ops — no new kernels):**
1. `ChromaRadianceNerfHead` — full GPU port. Embed Linear **folded into a stride-16 Conv2d at load**
   (RGB columns → kernel taps; the constant positional-feature term pre-summed into the bias) + one
   `Permute0213` → `[N, 256, 64]` tokens. Per depth: ONE `param_generator` GEMM per patch tile, chunk
   split via `SliceLastDim`, L2-normalize as **batched-transpose → RmsNorm(scale=1/√dim) → transpose
   back** (exact), then per-patch `BatchedMatMul` (batch = patches) for gate/value/out + `Silu`/`Mul`/
   `Add`. Final RmsNorm → `UnpatchifyTokens(innerChannelFastest)` → final conv. GLU blocks run in
   **1024-patch tiles** (round 2) — the untiled version's 1 GB gate/value/glu transients beside the
   19 GB resident DiT caused ~12 async-pool OOM-retries/step (all recovered but stalled); tiling
   bounds transients ≈268 MB and the warnings vanished.
2. `ChromaRadianceTransformer.ForwardPaired` — CFG pair shares ONE modTable build, ONE conv patchify,
   ONE NeRF pixel embed per step (each was ×2).
3. `ChromaRadiancePipeline` — ChromaPipeline round-3 kit: **prompt-embedding cache** (repeat prompts
   skip T5 entirely), **context trim to kept tokens** (mask-free SDPA, exact), **DiT residency**
   (`_ditResident`, evicted only for a new-prompt T5 phase; + `TrimMemoryPool` after T5 free),
   **drain-free device loop**: `X0Prediction.ToVelocityDevice` (new device twin) + in-place
   `CfgEulerStep(pixels, vCond, vUncond, cfg+1, dt)` — cond-anchored CFG maps onto the uncond-anchored
   kernel via guidance = cfg+1. Previews every 4th step (D2H sync each read). Masked inpaint keeps the
   host branch.

**Parity:** 4-step seed-42 A/B old-vs-new engine: corr **0.99996**, mean |Δ| 0.3/255 — port is exact
(1e-20-eps L2 + op-order noise only). 13/13 CPU unit tests pass. 20-step seed-42 astronaut VIEWED:
coherent, the checkpoint's known dark/backlit/vignette WIP character preserved.

**Numbers @1024²/20st/cfg3.5:** first-e2e 735.6s → cold 64.1s (load-warm, new prompt: 7.0s T5 + 20
steps) → **warm 54.4s (2.65-2.7 s/step, prompt-cache hit, DiT resident, GPU ~91% util)**. No ComfyUI
baseline for this arch in the bench file yet.

**Next levers (documented, not done):** F16 the NeRF head + backbone (Chroma damp kit — needs a
Radiance-head audit, `ChromaTransformer._f16Mode` is classic-only by design), per-generation step
graph (ForwardPaired is Radiance-local, no graph plumbing), fp8 requant of the 19 GB BF16 checkpoint
(would free ~9 GB → headroom + bandwidth), batched CFG.

## 2026-07-11 — Zeta-Chroma confetti ROOT-CAUSED + FIXED: BF16 `in_ln` misread as F32 in the decoder head

**Verdict: engine bug (NOT checkpoint character).** ComfyUI baseline (Swarm comfy self-start backend 0
on the 3060, same settings: seed 42, 1024², 20 steps, cfg 3) rendered a clean astronaut-on-horse
(`Output/local/raw/2026-07-11/0515001-*.png`) while engine 44.68 produced the same composition buried
under fine periodic RGB confetti (`0502001-*.png`).

**Oracle method (no guessing):** added env-gated stage dumps `HARTSY_ZETA_DUMP=<dir>` to
`ZetaChromaTransformer` + `ZetaChromaDecoderHead` (FLite dump pattern; step-0 cond pass only, raw F32
`.bin` + shapes manifest), then replayed `dec_net.*` in torch F32 from the checkpoint fed with the
dumped `pixel_patches`/`img_tokens`:
- `dec_embed_input` 0.0 / `dec_h_embed` 1.6e-3 / `dec_cond_silu` 1.1e-3 → inputs + Linears fine.
- **`dec_h_block0` relL2 = 1.24 — first AdaLN res block diverges.** No shift/scale/gate chunk
  permutation nor no-affine variant reproduced the engine values.
- Replaying with the in_ln weight/bias **BF16 bytes reinterpreted as F32** (mmap: the OOB half reads
  the next tensor in the file) matched the engine at relL2 3.5e-3/3.9e-3/4.9e-3/4.4e-3 for blocks 0-3
  → smoking gun.

**The bug:** checkpoint ships every `dec_net.*` tensor BF16; `ZetaChromaDecoderHead.
ApplyAffineAndModulation` (host loop) read `_lnWeight/_lnBias` via `(float*)DataPointer` — garbage
per-channel affine in all 4 res blocks → per-patch garbage → 32-px periodic RGB confetti after
unpatchify. Same class as the F-Lite `register_tokens` misread (fleet lesson: audit host `float*`
weight reads for BF16).

**Fix (44.70-local, deployed):** `EnsureF32HostReadable` in `ZetaChromaDecoderHead.LoadWeights` —
casts in_ln weight/bias to F32 at load (owned casts disposed with the head). Verified: corrected-math
step-0 x0 renders a clean blurry martian composition; full Swarm gen `0548001-*.png` = coherent
astronaut-on-horse, confetti gone, visually comparable to the Comfy baseline.

**Flagship regression gate (post-deploy, 1024²/8st/cfg1):** Z-Image-Turbo **2.75s** warm (≤3.2 ✓,
44.68 was 2.68s), Krea2-Turbo **4.46s** back-to-back warm (<6.5 ✓, historical 4.44s), both coherent.

**Open / notes:**
- Krea2 OOM'd at VAE decode when Zeta (13 GB pixel-space resident) + Z-Image + Krea2 piled up in one
  process (24 GB); clean after restart. Known model-switch eviction class — Zeta should join the
  sync-probe-reclaim matrix.
- Deploy friction: another agent's in-flight `HunyuanImagePipeline` unused fields (CS0169) break
  Release pack of Diffusion + its 4 dependents → packed those 5 with `-p:TreatWarningsAsErrors=false`.
  A partial 44.70 feed made NuGet float `HartsyInference.Vision` to nuget.org's alpha.45 (NU1605
  downgrade error) — always verify all 15 nupkgs exist for a version before restarting Swarm.
- Sampling recipe (uncond-anchored CFG, dynamic shift exp(mu)) was already correct; artifact was
  purely the decoder head.
- `HARTSY_ZETA_DUMP` instrumentation kept in-tree (env-gated, zero-cost when unset).


## 2026-07-11 — HiDream-i1 activation-memory pass: 1024²-CFG UNBLOCKED — warm **43.8s** (25st/cfg5), peak 23.5 GB

The "~6.9 GB F32 activation spike" blocker is resolved; 1024²/25-step CFG-5 now completes on the 4090
(3 consecutive full MRE runs, zero OOM-retry warnings, coherent astronaut-on-horse visually verified,
seed-42 deterministic across cold/warm). **Warm 43.8s / cold-in-process 61.4s (~1.73 s/step for the
serialized CFG pair, ≈0.87 s/forward), steady-state loop plateau 20.27 GB (dead flat across all 25 steps),
whole-gen peak 23.47 GB** (a single 0.5s-sample spike at the FIRST VAE decode = cuDNN conv algo probing;
the warm decode peaks 21.0 GB). Findings + fixes, all measured with an nvidia-smi 0.5s sampler + a new
env-gated probe (MRE: scratchpad/hdmre):

1. **The denoise loop was never the problem** — with the CFG pair serialized it runs at weights+3.2 GB
   (20.27 GB total) and the stream-ordered pool holds it flat. The real growth was **(a) a 40 MB/forward
   VRAM leak**: `HiDreamTransformer.Forward` never disposed the `x_embedder` output once the first double
   block replaced `curImg` (the step-7 else-branch was unreachable with ≥1 double block) — 2 GB by the end
   of a 25-step CFG gen, marching the plateau into **(b) the VAE-decode peak**, the generation's true
   maximum (the tiled F32 decode runs beside the resident 17 GB fp8 DiT).
2. Fixes shipped (engine, HiDream-scoped only): leak fix in `HiDreamTransformer.cs`; `HiDreamPipeline.cs`
   decode now uses **48-latent tiles** (same 3×3 grid at 128² latent as the 64 default but −44% per-tile
   area → warm-decode peak 23.9→21.0 GB, output bitwise identical after 8-bit quantize) + post-decode
   `TrimMemoryPool()` under KEEP_MODELS (the Flux post-decode pattern) so the decode reservation is handed
   back (23.5→20.24 GB retained watermark — protects cross-model eviction).
3. **F16-activation absmax profile measured** (new `HARTSY_HIDREAM_PROBE=1` first-forward probe in
   `HiDreamTransformer`, the FLite/TE-probe idiom): image stream benign (695→1.7k absmax, rms ~20), but the
   **encoder/joint stream rides at ~99k absmax** from double-block 15 through all 32 single blocks (rms ~280)
   — over F16's 65504 raw, NOT Qwen-class ±10M. Every branch input passes a no-affine LayerNorm, so the
   ChromaF16.ResidualDamp recipe applies cleanly if ever needed (1/32 → ~3.1k absmax); **not needed for
   memory** — skipped because the F32 loop already fits with 3.5 GB spare and the MoE kernels
   (`MoeTopKGate`/`RowGatedAccumulate`) are F32-only today. It remains a pure-perf lever.
4. Not reproduced: the 07-10 "OOMs on step 1" — on a clean GPU (Swarm shut down, only rustdesk's 424 MB
   resident) step 1 fits with ~3 GB to spare even pre-fix; that report likely included leftover-context
   junk on top of the leak. Reliability note: zero fp8 hard-faults across this session's 5 loaded runs.

Scoreboard: first HiDream 1024²/cfg-5 numbers above are MRE-side (engine `alpha.44-local` working tree,
not yet packed); Comfy comparison + official scoreboard row pending the main session's deploy.
→ **Landed 07-11 on `44.71-local`** (see the validation-matrix section below): Swarm-API warm ×3 =
43.99/44.02/44.04s (median **44.0s**, matches the 43.8s MRE number), peak VRAM 20.6 GB (1s sampler around
a warm gen), coherent astronaut-on-horse visually verified. Official row = **44.0s vs Comfy 35.2s (1.25×)**
in PERFORMANCE.md §5 / README / MODEL_STATUS_IMAGE / the benchmark log.

## 2026-07-11 — 44.71-local deployed + full validation matrix: NRE fix VALIDATED, HunyuanImage 74.1s (TE cache HIT), Klein 2.35s (graph ✓), HiDream official row, flagships green

Deploy: the 07-10 CS0169 pack blocker is GONE — the other agent's HunyuanImagePipeline TE-cache fields are
now fully wired (cache hit/miss + `_ditResident` residency staging all read), full solution builds
0 warnings/0 errors with `TreatWarningsAsErrors=true`, all 15 nupkgs packed clean as `44.71-local`
(no suppression), extension pin bumped, extension bin+obj purged, bare relaunch verified
(`[Cuda] perf flags:` line present, Backend #2 live on the 4090).

Validation matrix (standard astronaut prompt, seed 42, Swarm API wall, every PNG viewed):

1. **GGUF model-switch NRE fix VALIDATED** — Flux.2 Dev Q4_K_S (1024²/28st) → Krea2-Turbo → Dev again,
   one process, no restart. All three coherent; **zero** `NullReference`/`OnPromotedHostAccess`/`Finalizer
   GPU-cleanup callback failed` lines. Switch back INTO Dev completed too (121.8s incl. reload — the known
   VRAM-reclaim OOM did not reproduce). Dev warm 73.9s @28st = **2.64 s/step**, identical rate to the
   52.6s/20st first e2e (eager loop; `[Flux2] quantized DiT … step graph is opt-in` logged, no capture-OOM).
   Benign recurring warning: the 1017 MB VAE-decode alloc async-OOMs + sync-retries beside the resident DiT,
   then succeeds (gen unaffected) — candidate for a pre-decode TrimMemoryPool like HiDream's.
2. **HunyuanImage 2.1** 2048²/20st/cfg3.5 ×2: gen 1 78.6s engine-side (≈ the 77.0 baseline), gen 2
   **74.1s** with `[HunyuanImage] prompt-embedding cache HIT — TE phase skipped` logged. Coherent, seed-42
   deterministic, ZERO OOM warnings beside the resident DiT (no KEEP_MODELS=0 fallback needed). Note the
   drop is modest because the 40-forward CFG loop dominates at 2048²; the cache's bigger win is skipping
   the TE⇄DiT VRAM swap. Scoreboard 77.0 → **74.1s**.
3. **Flux.2 Klein 4B** (BF16, 4st/cfg1): warm ×2 engine 2347/2327 ms (**≤2.4s held**), step graph captured
   + replayed (`[Flux2 graph] denoise step captured; replaying via cuGraphLaunch`, capture window
   OUTSTANDING 0), coherent.
4. **HiDream-i1** official row — see the section above (44.0s ×3 flat, peak 20.6 GB).
5. **Flagship gate ✓**: Z-Image-Turbo **2.79s** (≤3.2; 44.70 was 2.75), Krea2-Turbo **4.49s** (<6.5;
   44.70 was 4.46), both coherent.
6. **Zeta-Chroma** VERIFIED-CLEAN status carried into 44.71 (fix shipped in 44.70); MODEL_STATUS_IMAGE
   row promoted ⚠️→✅.

Docs updated: MODEL_STATUS_IMAGE.md (perf table + HiDream/Zeta/Flux.2-Dev/HunyuanImage rows),
PERFORMANCE.md §5, README benchmark table, benchmarks/results/image_comfy-vs-hartsy_2026-07-05.md.
Open: Flux.2 Dev step-graph capture on 24 GB (still eager), HunyuanImage ByT5 glyph branch, the benign
Flux.2 VAE-decode OOM-retry warning.

## ▶▶ NEXT AGENT — HunyuanImage 2.1 bring-up (staged + scoped 2026-07-10, start here)

**All weights staged, no downloads needed:** transformer `Models/Stable-Diffusion/HunyuanImage/HunyuanImage2.1-Q4_K_M.gguf`
(QuantStack, 856 tensors: 20 double + 40 single blocks, Tencent-style keys, no VAE/TE inside);
VAE `Models/VAE/HunyuanImage/hunyuan_image_2.1_vae_fp16.safetensors` (NEW download);
ByT5 `Models/clip/byt5_small_glyphxl_fp16.safetensors` (NEW download);
Qwen2.5-VL-7B TE `Models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors` (already present, + full-fp16 sibling).

**PROGRESS 2026-07-10 pt2 — converter DONE + VALIDATED on the real GGUF:**
`HunyuanImageCheckpointConverter.ConvertTencentToDiffusers` implemented (mirrors diffusers
`convert_hunyuan_image_to_diffusers.py`, fetched + condensed this session): full byt5_in/txt_in-refiner/
double/single/final mapping, fused qkv + single-linear1 [3h|4h] splits quant-block-aligned, final-layer
[shift,scale]→[scale,shift] swap, img_in 1x1-conv weight flattened to a [3584,64] Linear (GGUF rank-4 dims
are ggml-reversed — the rank-2 relabel skips it). `HunyuanImageKeyMapper.MatchesByKeys` now also matches
Tencent GGUFs (img_attn_qkv + byt5_in; note the file declares arch 'hyvid' and previously fell through to
FluxKeyMapper identity — worked, but the explicit match is safer). MRE validation: 856 GGUF tensors →
1264 diffusers keys → `HunyuanImageTransformer(V21).LoadWeights` **OK** (scratchpad `hymre/`).

**PROGRESS 2026-07-10 pt3 — loader WIRED + deployed (`44.50-local`); blocked on the pixel-shuffle VAE:**
`HunyuanImageLoader` written + all 6 dispatch sites wired (compiles, routes; `ModelSupport.cs` refusal gate
lifted — the old "T5-XXL stand-in" note was stale, the Qwen2.5-VL path is real). In Swarm: GGUF transformer
converts + loads ✓, Qwen2.5-VL-7B fp8 TE loads ✓ (auto-download entries `SideModels.Qwen25Vl7BHunyuan` /
`HunyuanImageVae` added with hashes; note EnsureSideModel re-downloaded the VAE to `Models/VAE/` root —
the pre-staged `Models/VAE/HunyuanImage/` copy is a deletable duplicate). `VaeConfig.HunyuanImage` fixed
(64-ch latent, [128,256,512,512,1024,1024], scale 0.75289 per ComfyUI `HunyuanImage21`); `ConvertVaeKey`
gained `reverseUpIndices:false` (this file's `up.0` = deepest, opposite of SD-LDM).
**BLOCKER: the decoder is a PIXEL-SHUFFLE VAE, not standard LDM** — per-level upsamplers are
`conv([4·C_next, C, 3, 3]) → PixelShuffle(2)` carrying the channel transitions (up.0 [4096,1024] →
up.1 1024; up.1 [2048,1024] → 512; …; up.4 [512,256] → 128; up.5 has none), resnets are all
channel-preserving (zero shortcuts). The engine's `VaeDecoder`/UpDecoderBlock2D (resnet transitions +
interpolate-conv upsample) cannot represent this → `KeyNotFoundException conv_shortcut`. NEXT: write a
dedicated HunyuanImage VAE decoder (reference: ComfyUI's hunyuan_image VAE / diffusers
`AutoencoderKLHunyuanImage`), swap into `HunyuanImageLoader`, rerun — everything upstream of the VAE is
proven loading.

**The remaining work (in order):**
1. ~~Converter~~ ✅ DONE (see PROGRESS above).
2. ~~GGUF key-mapper match~~ ✅ DONE (byt5_in + img_attn_qkv heuristic; file declares arch 'hyvid').
3. **Extension loader**: new `HunyuanImageLoader` per the ErnieImageLoader/Lumina2Loader anatomy (6 wiring sites),
   GGUF branch per QwenImageLoader/Flux2Loader (native quant + `RelabelRank2ToPyTorchOrder`), Qwen2.5-VL-7B encode
   via the existing `HunyuanImageQwenTextEncoder` (template already matches ComfyUI), ByT5 second stream
   (`TextEmbedDim2` + `encoder_hidden_states_2` — transformer support exists), VAE is 32×/64-ch (own config).
   Arch detection: check what Swarm classes the GGUF as (`DescribeModel`) — may need `EditModelMetadata`.
   **Findings this session:** `HunyuanImageQwenTextEncoder` is COMPLETE (encode template, hidden_states[-3]
   tap, 34-token prefix drop — needs `LlamaStyleEncoderConfig.Qwen2_5_VL_7B` + `Qwen2Tokenizer.EncodeChat`);
   the PIPELINE has NO byt5 wiring (transformer's `encoder_hidden_states_2` is optional — skip ByT5 for the
   first run, wire the glyph branch later); `VaeConfig.HunyuanImage` preset EXISTS but is marked
   "TODO: confirm exact architecture" — the 32×/64-ch VAE (`Models/VAE/HunyuanImage/hunyuan_image_2.1_vae_fp16.safetensors`)
   has never been loaded and is the likeliest debugging front. Do NOT copy `HunyuanImageGenerationTests`
   wiring — it uses the legacy CLIP-L/T5 ctor and `VaeConfig.Flux` (wrong VAE). Scratchpad `hymre/` has the
   validated GGUF→transformer load sequence to lift into the loader verbatim.
4. Deploy → 1024² gen → visual verify → warm bench + scoreboard row. GPU etiquette + flagship gate as always.

**Traps from the Flux.2 Dev bring-up (2026-07-10, memory `flux2-dev-gguf-bringup`):** verify WHICH file the Swarm
model resolver picks when a canonical name exists in multiple roots (THE Dev noise bug was an fp4 file shadowing
the fp8 TE); `HARTSY_TE_PROBE=1` dumps per-layer TE absmax for conditioning debug; model-switch away from a
GGUF pipeline currently NREs in `GpuTransferHelper.OnPromotedHostAccess` (restart Swarm between model switches);
BF16 GGUF tensors must be host-cast to F16 in the loader (transient path skips BF16 weights → blank).

---


## ▶▶ NEXT AGENT — finish HiDream i1 Dev + OmniGen-2 (start here)

This session verified **8 image models e2e** (Qwen-Image, Anima, Lumina-2, Chroma, Krea-2, ERNIE, Boogu, Kandinsky-5) and landed reusable engine fixes (committed): the GPU-residency DiT-block rewrite pattern, fp8 `.weight_scale`-companion remap, SDPA `[B,1,Sq,Skv]` mask broadcast, GroupNormSilu **and** LayerNorm F32-path F16/BF16-affine casts, BF16→F16 transformer cast for blank-image avoidance, and `ClipTextEncoder` CLIP-L raw-EOS pooled. Two models remain; both download + load but need correctness/perf work.

> ### ⏩ UPDATE (next session, 2026-06-30 pt2) — FFN inner-dim bug fixed in BOTH models
> **Environment gotchas found (all real):** (a) no system CUDA 13 — copied the runtime out of Trash to `~/.local/lib/cuda13`; every GPU run needs `LD_LIBRARY_PATH=~/.local/lib/cuda13:$LD_LIBRARY_PATH`. (b) **The device mapping below is BACKWARDS** — CUDA orders fastest-first, so `CUDA_VISIBLE_DEVICES=0` = **4090** (HiDream), `=1` = **3060** (OmniGen2). (c) tests multi-target `net8.0;net10.0` and only net10.0 has a runtime → **pin `--framework net10.0`**. (d) view BMPs via an existing ComfyUI venv's PIL.
> **OmniGen-2 — FIXED (512-nocfg verified: coherent astronaut-on-horse, no artifacts).** ROOT CAUSE of BOTH the blocky-bottom-third AND the wrong-subject: `OmniGen2Transformer.ComputeFfnInnerDim` used Llama's `8/3·dim` (=6912) but the checkpoint's `feed_forward.linear_1` is `[10240,2520]` (= `round_up(4·dim,256)`). SwiGLU buffers were sized 6912 while `backend.Linear` writes N=10240 from the weight → out-of-bounds GEMM writes → tail image tokens corrupted (bottom third) + adjacent-memory corruption (wrong subject). Fix: base `= 4·HiddenSize`. The embeds were fine. **This also likely fixes the 1024-CFG illegal-address** (same overflow at 4096 tokens) — verifying.
> **Found via the proven numerical-diff loop:** dump per-stage (`OMNIGEN2_DEBUG_DIR`) → localize to noise-refiner block 1 → isolated PyTorch reference from cloned `VectorSpaceLab/OmniGen2` (existing ComfyUI venv torch) fed our exact dumped input → sub-component relL2 showed attn matched (0.009) but MLP-out bottom rows = 1.0 → FFN buffer sizing.
> **HiDream i1 Dev — ✅ FIXED (coherent astronaut-on-horse @1024/8-step).** Two bugs, both found via the numerical-diff harness (dump transformer inputs+stages with `HIDREAM_DEBUG_DIR`, PyTorch reference from naked-fp8 weights `.float()`, first-divergence relL2). **Bug 1 (FFN):** `SwiGluForward` sized SwiGLU buffers from computed `ffDim=4·hidden=10240` vs the 6912/3584 weights → overflow; now derives inner dim from `w1.Shape[0]`. **Bug 2 (DOMINANT — the brown cloud):** `HiDreamTransformer.LoadWeights` set `numCaptionProjections = CaptionChannels.Length` (=2), loading only 2 of the **49** caption projections → every Llama layer projected through `caption_projection[0]`, T5 through `[1]`. Diffusers uses 49 (= num_layers+num_single_layers+1): Llama layer `i` → `caption_projection[i]`, T5 → `[-1]`=[48]. Fix: load all 49, project per-block. (`t5_proj` relL2 was 17.86; matched proj[1] not [48].) **HiDream FULLY VERIFIED:** Dev nocfg coherent at full **25-step** (~49s/step, sharp astronaut-on-horse) AND the **1024-CFG path is functional** (4-step smoke test, 2 sequential B=1 forwards, no illegal-address/OOM — the handoff's "1024-CFG illegal address" was a symptom of the same FFN/caption bugs). **GPU-residency perf rewrite KEPT (52s→~29s/step, 44% faster):** block CPU glue → backend ops (SliceLastDim / AffineBroadcastLastDim / GatedResidualLastDim / LayerNormNoAffine). An earlier run OOM'd and was WRONGLY blamed on the rewrite — the real cause was ~550MB of leftover Python CUDA contexts (my own reference-diff scripts) + rustdesk on the 4090 eating the 47.5MB-margin headroom; on a clean 4090 it fits fine. (Rewritten image differs 9/255 from the CPU version — diffusion sensitivity to GPU-vs-CPU LayerNorm float ordering, not a bug.) **Reliability caveat:** intermittent hard-fault crash (~5-10%/forward, no OOM warning) during the fp8 denoise loop under memory pressure — retry-able (25-step crashed once then completed on retry); `pkill -9 -f testhost` + kill dotnet holding GPU VRAM between runs. Memory: `hidream-caption-projection-per-block`.
> **(superseded note) earlier HiDream FFN-only attempt:** Code used `ffDim = 4·hidden = 10240` but HiDream's checkpoint FFN is `6912` (routed experts / text FFN) and `3584` (shared expert) — the Llama `8/3` width. `SwiGluForward` allocated with the computed dim → overflow. Fix: `SwiGluForward` now derives the inner dim from `w1.Shape[0]` (the weight = the GEMM's N), robust for all three FFN paths. **General lesson: never size a SwiGLU/proj buffer from a computed dim — read it from the weight, since `backend.Linear` takes N from the weight.** But 8-step output is STILL a brown cloud — necessary, not sufficient. QK-norm verified correct vs diffusers (RMSNorm over inner_dim, before head reshape). NEXT: HiDream needs the same per-stage-dump numerical diff (vs diffusers `HiDreamImageTransformer2DModel` / `comfy/ldm/hidream`). Top suspects: final `norm_out` scale/shift order, image-only RoPE, 12/6-way AdaLN chunk order, fp8 dequant. See memory `hidream-status-still-incoherent`.

**HiDream i1 Dev (run on the 4090, `CUDA_VISIBLE_DEVICES=1`):** the encoder crash is FIXED (CLIP-L pooled, SD3/SDXL-safe) and `HiDreamBlock.cs` is reverted in the working tree to its original correct-but-CPU-slow version (cpu=27; the half-rewrite committed at `591e6ad` was BROKEN — crashed the forward — do not use it). START by running `HiDream_I1_Dev_Gpu_1024_NoCfg` (drop to ~8 steps for a fast read) to confirm the original block makes a COHERENT image. The load occasionally HANGS during the 17GB-fp8 transformer load (retry; `pkill -9 -f testhost` first to free VRAM). The OOM fix is in (naked fp8 + `CacheWeightCasts=false`). If coherent → redo the `HiDreamBlock` GPU-residency rewrite PROPERLY (cpu→0, copy `BooguImageDoubleBlock.cs`, verify pixels unchanged); if not → numerical-diff vs ComfyUI `comfy/ldm/hidream`. NOTE 1024-CFG may hit the same illegal-address as OmniGen2.

**OmniGen-2 (run on the 3060, `CUDA_VISIBLE_DEVICES=0` — its 8GB F16 fits):** RUNS at 512-nocfg (68s) but has 4 bugs — fix in order: (1) **conditioning first** — it renders the WRONG subject, so verify the precomputed embeds (`OmniGen2/TestEmbeddings/{prompt,negative}.bin`, generated via ComfyUI `clip.encode_from_tokens`) actually match OmniGen2's instruction template + hidden-layer tap vs diffusers `pipeline_omnigen2.py`; (2) the **blocky bottom-third artifact** (full Decode, so it's in the LATENT → transformer corrupts bottom image tokens — check RoPE positions for later tokens, the unverified `OmniGen2Block` rewrite, and whether the ref-image/edit token path is wrongly active in t2i); (3) **1024-CFG `CUDA_ERROR_ILLEGAL_ADDRESS`** (B=2 out-of-bounds); (4) host-bound at 1024 (once-per-forward transformer glue scales with tokens). Use the numerical-diff loop (fixed input → per-component dump ours vs reference → first divergence). **DO NOT run silent poll-loops or `CUDA_LAUNCH_BLOCKING=1` timing runs — that's how the agents stalled this session; bound every step and report findings even if partial.**

---

## ▶ RESUME HERE — state as of 2026-06-30 (fresh-session handoff)

**Nothing is committed** (user does all git manually). All changes are in the working tree. Run recipe: `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`(4090)`/=0`(3060) + cuBLAS-12 `LD_LIBRARY_PATH` (cu12 libs, see below) + `-f net10.0 --no-build` + `-- xUnit.ParallelizeTestCollections=false`.

**Verified ✅ this session:** Qwen-Image, Anima, Lumina-2, **Chroma**, **Krea-2 Turbo**, **ERNIE-Image**, **Boogu-Image Base**, **Kandinsky-5 Lite** (8 new). Chroma + ERNIE transformers also numerically verified vs diffusers.
**General engine fix (Kandinsky):** `CudaBackend.LayerNorm` F32-input path now casts F16/BF16 affine weight/bias UP to F32 (was unguarded → near-zero output → noise; same class as GroupNormSilu). Audit every norm op's F32-input affine read when casting a model to F16/BF16.
**OmniGen-2 — 4 distinct bugs (NOT close):** (1) 512-nocfg runs (3060, 68s) but renders WRONG subject (man, not the astronaut-horse the embeds were made for) → conditioning/embeds likely wrong (embeds generated via ComfyUI `clip.encode_from_tokens` for Qwen2.5-VL — may not match OmniGen2's instruction template/tap). (2) BLOCKY bottom-third artifact (full `Decode`, not tiled → it's in the latent → transformer corrupts bottom image tokens; suspect RoPE-positions / unverified `OmniGen2Block` rewrite / ref-token path active in t2i). (3) host-bound at 1024 (9s/step, GPU 4% — once-per-forward transformer glue scales with tokens). (4) **1024-CFG crashes `CUDA_ERROR_ILLEGAL_ADDRESS`** at ~step 5 (B=2 bounds bug). Needs a dedicated multi-bug debug campaign. 3B TE + embeds staged.
**HiDream — encoder FIXED, now crashes in the block:** the `ConcatPooled` NRE is FIXED — `ClipTextEncoder.ExtractPooledOutput`/`EncodePenultimate` now return the raw EOS pooled when there's no `_textProjectionWeight` (CLIP-L), guarded so SD3/SDXL (which load projections) are unchanged. HiDream now gets past encoding + into the denoising loop ("HiDream t2i: 1024x1024" prints). BUT it then CRASHES in the transformer forward (exit 1 @155s, no Step output) — the half-rewritten `HiDreamBlock` (cpu=16, broken intermediate) dies on the first forward. **NEXT: finish the HiDreamBlock GPU-residency rewrite (cpu 16→0) OR revert it to the committed original, then re-run.** Also host-bound until the block is done.

**Reusable ENGINE fixes landed (working tree, uncommitted):**
- `Tensor.SetKeepAlive` + `SafeTensorsLoader`/`GgufLoader` root the mmap handle (fixes GC-unmap AccessViolation). `InternalsVisibleTo ModelHandler` added.
- `GgufModelLoader.RelabelRank2ToPyTorchOrder` (GGUF ggml [in,out]→[out,in]); applied in the Qwen GGUF path.
- `CudaBackend.DequantizeToF32` (test/tooling GGUF dequant) + GGUF dequant verified vs gguf-python (4.6e-4).
- **Qwen final-layer scale/shift swap fix** (`QwenImageTransformer.ApplyFinalLayer`: AdaLayerNormContinuous is `(scale,shift)` scale-first) — was THE Qwen grid bug.
- **Qwen conditioning fix** (test: `Qwen2Tokenizer.EncodeChat` template + `promptDropIndex:34`, not raw-512).
- **`QwenImageBlock` GPU-residency rewrite** (30min→13min; mirrors ChromaDoubleStreamBlock + fused `QwenImageRope.ApplyJoint`).
- **ERNIE BF16-BN-cast fix** (`ErnieImagePipeline` ctor casts bn stats to F32).
- **General SDPA mask-broadcast fix** (`AttentionKernels` already handles full-Sq; `ErnieImageTransformer.BuildAttentionMask` now emits `[B,1,Sq,Skv]`; new CUDA `B·Sq·Skv` broadcast-over-heads branch in `CudaBackend.ScaledDotProductAttention`).
- **Krea-2 fp8 scale-companion fix** (`Krea2CheckpointConverter.RemapTransformerKey`: when renaming fp8 keys, the `.weight_scale`/`.scale_weight` companion MUST be remapped through the same logic as its `.weight` or `ApplyFp8ScaledDequant` can't pair them → scale dropped → weights ~250-900× too big → noise). ⚠ **Check every fp8-key-renaming converter for this** (BooguImageCheckpointConverter next).
- **ERNIE VAE banding fix** (`ErnieImagePipeline.cs:188` DecodeTiled→full `Decode`; tiled per-tile GroupNorm + bad overlap blend caused horizontal bands).
- **GroupNormSilu F32-path BF16-affine cast fix** (`CudaBackend.GroupNormSilu` ~line 1465: the F32-activation path passed BF16/F16 weight+bias straight to the kernel WITHOUT casting → wrong affine. Was THE ERNIE washout. Affects any F32-activations + BF16-affine GroupNorm; F16/BF16 paths already cast). + ERNIE pipeline frees TE weights right after encode (3060 OOM fix).
- Worktree (UNMERGED, pending validation): `agent-a9925eeaa59218004` — Flux2/HiDream/Krea2/Lumina2 block GPU-residency rewrites.

**Downloaded + key-verified (in /tmp, symlinked into Models/ — /tmp won't survive reboot!):**
- Qwen-Image-Edit Q5_K_M GGUF (`QwenImageEdit/`), Krea-2 Base+Turbo fp8 + Qwen3-VL-4B TE (`Krea2/{Base,Turbo}/`), Boogu Base/Turbo/Edit fp8 + Qwen3-VL-8B TE (`Boogu/{Base,Turbo,Edit}/`). All keys verified loadable. TestPaths blocks added (`Krea2`, `Boogu`, `QwenImageEdit`).
- **Kandinsky 5.0 Lite** (BF16 12GB, `Kandinsky5/...Diffusers/{transformer,vae}/`) — ✅ MATCH, precomputed embeds present (`TestEmbeddings/prompt_qwen.bin`+`prompt_clip.bin`) → **READY to test, no fix needed**.
- **HiDream i1 Dev fp8** (17GB naked-fp8, `HiDream/{hidream_i1,vae,clip_l,clip_g,t5xxl,llama_3_1_8b}.safetensors`) — ✅ MATCH, quad-encoder all staged, no scale companions → **READY to test**.
- **OmniGen-2** (F16 8GB, `OmniGen2/{omnigen2,vae}.safetensors`) — staged but BLOCKED: (a) engine key-remap — file ships `time_caption_embed.timestep_embedder.linear_1/2`, `caption_embedder.0/1`, `norm_out.linear_1/linear_2` but `OmniGen2Transformer.LoadWeights` wants `time_proj.0/2`/top-level `caption_embedder`/`norm_out.linear`+`norm` (silent-null via TryGetValue → wrong forward); fix `OmniGen2CheckpointConverter`/`LoadWeights`. (b) needs Qwen2.5-VL-**3B** TE (`qwen_2.5_vl_fp16` 7.5GB, NOT the local 7B) + missing precomputed `TestEmbeddings/prompt.bin`.

**IN FLIGHT / NEXT:**
1. **Krea-2 Turbo: ✅ DONE** (verified `krea2_turbo_1024_20260630_095817.bmp`, std 66.5/grid 0.042, fp8 scale-companion fix above). Remaining: Base/CFG path (shares converter+transformer; CFG anchoring `Krea2Pipeline.cs:100` validation-pending) + GPU-residency (28-step CFG host-bound, item 7).
2. **Boogu Base: ✅ DONE** (`boogu_base_1024_20260630_122132.bmp`, std 97.8/grid 0.038, ~6min @1024-28cfg). Fixes: VAE bare-ldm key remap (`ConvertVaeKey`) + double-block GPU-residency rewrite (`BooguImageDoubleBlock`). Test `BooguImageGenerationTests.cs` (+`512_Fast`). **Turbo** = same path, needs a run; **Edit** needs the Qwen3-VL vision tower. ⚠ Boogu converter is pass-through so NO fp8 scale-companion bug.
3b. **STATUS UPDATE (agents stalled — driving directly):** Kandinsky-5: OOM+perf FIXED, runs 512/25-step in 52s on 4090, but the BF16+CacheWeightCasts=false path gave a BLANK image → switched to **F16 cast** (std 5.6→70.2, weights now applied) → but now NOISE = a real transformer CORRECTNESS bug (needs numerical diff vs ComfyUI; same class as Qwen final-layer/Chroma). HiDream: block HALF-rewritten (27→16 sites, broken intermediate — finish or revert before running); OOM handled (fp8+CacheWeightCasts=false, naked fp8). OmniGen2: block rewritten (32→0) but needs the 3B TE + embeds. **Lesson: the remaining models each need a correctness diff, not just download+run.**
3. **Kandinsky-5 / HiDream / OmniGen-2 — ALL IN PROGRESS.** Kandinsky-5 (4090): fixing the OOM (12GB BF16 transformer cast to 24GB F32 → keep BF16 + CacheWeightCasts=false) + perf/correctness. HiDream Dev (4090-when-free): same fp8-F32-cast OOM fix (17GB→34GB) + verify. OmniGen-2 (3060, 8GB fits): key-remap fix (file ships `time_caption_embed.timestep_embedder`/`caption_embedder.0/1`/`norm_out.linear_1/2`; LoadWeights wants old names) + Qwen2.5-VL-**3B** TE download + embeds. **RECURRING PERF/VRAM PATTERN for fp8/BF16 models: tests do `CastWeightsToF32` on the big transformer → 2× VRAM → OOM; fix = pass quantized weights to LoadWeights + `CacheWeightCasts=false` (transient per-GEMM cast).** Operational: `pkill -9 -f testhost` between runs (orphans hold VRAM → false OOM).
4b. **Qwen-Image-Edit / Boogu-Edit** — TODO: build the EDIT pipeline (VAE-encode ref image + Qwen-VL multimodal conditioning); transformers load but no edit pipeline/TestPaths exist.
4. **ERNIE — ✅ DONE** (verified `ernie_image_v1_512_cfg_20260630_103703.bmp`, std 60.9/grid 0.069). 4 bugs fixed (BN-cast, SDPA mask, VAE banding, GroupNormSilu F32-path BF16-affine cast — the washout). Remaining: GPU-residency (CPU-bound blocks, item 7).
5. **Chroma checkpoint**: only `/tmp/chroma_dl/Chroma1-HD-fp8mixed-final.safetensors` remains (the `do_not_use/…exp` variants + alternates were deleted to reclaim 137GB; the symlink points to the good one). **Disk policy (user-approved): verified-✅ models' /tmp downloads may be deleted when low on space** — keep only the file the Models/ symlink uses.
6. **HunyuanImage 2.1: BLOCKED** on 24GB (35GB bf16; fp8/GGUF repacks use incompatible original-Tencent keys). Deleted from /tmp.
7. **GPU-residency block rewrite (PERF — ~27s/forward + GPU ~7% util = host-bound CPU glue).** Done: Qwen, Chroma, **Boogu** (double-block rewrite → 7%→72% util, ~6min/1024; NOTE the per-block bottleneck was only the DOUBLE blocks — Boogu's single blocks were already GPU-resident, and the transformer's remaining ~27 CPU sites are once-per-forward glue, not per-block). **STILL PENDING: Krea-2, ERNIE, Flux2, HiDream, Lumina2** (apply the same — check whether the CPU glue is in the double/single block class or once-per-forward before rewriting).
**GLUE SCAN (run this FIRST per model — `grep -cE "for \(int|float\*|DataPointer" <file>` vs Backend ops; BLOCK files = per-block hot path that MUST be GPU-resident, TRANSFORMER files = mostly once-per-forward [minor], ROPE files = intentionally CPU):**
  - NEED block rewrite: `ErnieImageBlock` cpu=36, `Lumina2Block` cpu=38, `Flux2DoubleBlock` cpu=33 / `Flux2SingleBlock` cpu=29, `OmniGen2Block` cpu=32, `HiDreamBlock` cpu=27, `Krea2Block` cpu=15.
  - ALREADY GPU-resident (skip): `Kandinsky5Block` cpu=6/gpu=50, `BooguImageSingleBlock` (done), `QwenImageBlock`/`ChromaDoubleStreamBlock` (done).
  - Transformer once-per-forward glue (don't bother unless still slow after block fix): Lumina2Transformer 54, ErnieImageTransformer 46, Kandinsky5Transformer 40, HiDreamTransformer 38, OmniGen2Transformer 36, Krea2Transformer 25. Pattern: CPU for-loops over batch×seq×hidden reading DataPointer → `IBackend` GPU ops (RmsNorm/LayerNormNoAffine/Concat/Permute0213/SliceRows/AffineBroadcastLastDim/GatedResidualLastDim); copy `QwenImageBlock.cs`/`ChromaDoubleStreamBlock.cs`/`BooguImageDoubleBlock.cs`. **Verify the rewrite preserves pixels (perf-only).**

---

Goal: systematically take every **unverified** image model (🔬/🔧) to visual e2e + Python parity on this
box (RTX 4090 24 GB = CUDA index 1, RTX 3060 12 GB = CUDA index 0). Per-model loop: download official
weights (ungated Comfy-Org/community mirror if gated) → dump safetensors keys → check C# logic vs the
official reference → run the gated generation test → **inspect the output image myself** → if wrong, dump
a Python layer reference and diff until the first `avg_err>1e-3` layer → fix C# → re-run → document the
bug. Final bug entries go to `PARITY_VERIFICATION.md` §Bugs; status flips in `MODEL_STATUS_IMAGE.md`.

GPU selection: tests hard-code `deviceOrdinal:0`, so pick the card with `CUDA_VISIBLE_DEVICES`
(`=1` → 4090, `=0` → 3060). Big/dual-transformer models → 4090; ≤8 GB models → 3060 (frees 4090 for a
parallel run).

## Environment ready
- dotnet 10.0.109; diffusion test project builds clean (Release). Solution multi-targets `net8.0;net10.0`
  but only net10 SDK is installed → **always run tests with `-f net10.0`** (net8 testhost errors out).
- **CUDA cuBLAS version is the dominant env issue.** Driver is 13.0, but `CudaLibraryResolver` resolves
  `cublas` by trying `.so.13` → **`.so.12`** → `.so.11` (engine's validated target is **cuBLAS 12**, per the
  `CublasApi` doc). Forcing cuBLAS 13 (cu13 libs on the path) loads but throws **`CUBLAS_STATUS_NOT_SUPPORTED`**
  on some GEMM configs (hit on Flux Dev) and is the suspected cause of the bad AuraFlow/Chroma images.
  **Use cuBLAS 12** by putting ONLY the cu12 libs on the path (so `.so.13` isn't found → resolver falls back to 12):
  `export LD_LIBRARY_PATH="/home/hartsy/Desktop/Swarm/SwarmUI.not too old/dlbackend/ComfyUI/venv/lib/python3.12/site-packages/nvidia/cublas/lib:/home/hartsy/Desktop/Swarm/SwarmUI.not too old/dlbackend/ComfyUI/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/usr/lib/x86_64-linux-gnu"`
  (cuBLAS 12 + cudart 12 run fine on the CUDA-13 driver via forward-compat.) `libcuda.so` (driver API) is system.
  **NOTE: all earlier results in this doc used cuBLAS 13 and are suspect — re-run on cuBLAS 12.**
- GPU select: **must set `CUDA_DEVICE_ORDER=PCI_BUS_ID`** (CUDA's default order ≠ nvidia-smi; without it
  `CUDA_VISIBLE_DEVICES=1` wrongly picked the 3060). With it: `CUDA_VISIBLE_DEVICES=1` → 4090, `=0` → 3060.
  Tests hard-code `deviceOrdinal:0`. AuraFlow fp8 (+T5) OOMs the 3060's 12 GB → use the 4090.
- HF CLI venv: `/home/hartsy/hfvenv/bin/hf`. No HF token (gated → Comfy-Org/community mirror).
- `Models/` wired via symlinks to local SwarmUI weights (see below) + Qwen3 tokenizer downloaded.

## Per-model status (this session)

| # | Model | Tier | Weights | e2e | Parity | Notes |
|---|---|---|---|---|---|---|
| 1 | AuraFlow v0.3 | ✅ | ✅ `calcuis/aura` fp8 (self-contained) | **✅ on-prompt** | ✅ visual | **DONE @1024 → clean photoreal horse + mounted rider, on-prompt ("an astronaut riding a horse").** Fixed by TWO changes: (a) T5 attention scale 1.0 (was 1/√64), (b) Pile-T5-XL tokenizer `pile_t5xl_spiece.model` (was wrongly fed `t5_xxl_spiece.model`, different vocab). Denoiser/VAE/scheduler were always correct. Image: `Output/auraflow_v03_1024_cfg_20260629_105400.bmp`. NOTE: transient path @1024 took 26 min — slow; optimize later. Rider reads soldier-not-astronaut (AuraFlow v0.3 weakness + crop), composition correct. |
| 2 | Chroma | 🔧 | ✅ fp8 (`silveroxides/Chroma1-HD-fp8-scaled`, downloaded mxfp8mixed + fp8_scaled variants) | pending | — | Logic-checked: 2 fixes needed before run — (a) CFG should be **uncond-anchored** (`ChromaPipeline.cs:272` uses cond-anchored), (b) T5-XXL `AttentionScale=1.0` (`Xxl` preset unset → 0.125). **Trap: `Xxl` is shared with ✅ Flux/SD3 — give Chroma its OWN XXL config, don't edit the shared one.** mod-table/approximator wiring verified correct. |
| - | Anima (Cosmos-Predict2 2B) | ✅ | local fp32 | **✅ on-prompt @512 on 3060** | visual | Clean coherent "anime girl with blue hair in a garden" (`anima_t2i_512_nocfg_20260630_000534.bmp`), 93.9s/25-step on the 3060. Precomputed Qwen3-0.6B embeds. Verified e2e this session. |
| - | Qwen-Image | ✅ | ✅ Q4_K_M GGUF (`QuantStack/Qwen-Image-GGUF`, 13 GB) | **✅ clean photoreal on-prompt astronaut-on-horse @1024** (`qwen_image_v1_1024_gguf_20260630_020418.bmp`) | ✅ visual | **GGUF downloaded + pre-flight verified** (arch `qwen_image`, bare diffusers keys, no `model.diffusion_model.` prefix, no fused QKV; Q4_K/Q5_K/Q6_K all have GPU dequant kernels). 4 earlier fixes (scheduler maxSeqLen 8192/maxShift 0.9, `shift_terminal=0.02`, RoPE `scale_rope` centering, lazy GGUF load). **THREE NEW BLOCKER BUGS FIXED THIS SESSION (each crashed the run in sequence):** (1) **mmap dangle** — `Tensor` borrowing an mmap pointer didn't root the `MmapHandle`, so a GC during the 1933-tensor GGUF convert finalized it and unmapped the fp8 text-encoder mid-load → AccessViolation in `ApplyFp8ScaledDequant`. Fixed engine-wide: `Tensor.SetKeepAlive` + `SafeTensorsLoader`/`GgufLoader` root the handle. (2) **GGUF shape transposed** — `GgufLoader` emits ggml `[in,out]` dims; the image GGUF path never relabeled to PyTorch `[out,in]` (the LLM path does), so the first Linear (timestep embedder) got K=3072>input256 → M=0 → zero-grid kernel launch `CUDA_ERROR_INVALID_VALUE`. Fixed: shared `GgufModelLoader.RelabelRank2ToPyTorchOrder` applied in the image GGUF branch. (3) **VRAM OOM** — test never set `CacheWeightCasts=false`, so it cached an F16 dequant of every weight of a ~20B model (~40 GB). Fixed: transient per-GEMM dequant on the GGUF path → stable 15.3 GB resident on the 4090. **PERF BUG FOUND + FIXED:** `QwenImageBlock.Forward` ran its entire glue (LayerNorm/AdaLN-modulation/QK-norm/reshape/concat/split/gated-residual) on the CPU via `DiTUtils`/`QkNorm`/`AdaLNModulation` — each forced a GPU→host→GPU round-trip of every ~50 MB intermediate (~16 ops × 60 blocks × CFG = thousands of PCIe syncs/step → 10+ min, GPU util ~15%). Rewrote it GPU-resident mirroring the verified `ChromaDoubleStreamBlock` (interleaved RoPE fused into one joint CPU pass `QwenImageRope.ApplyJoint`). Result: **30+ min host-stalled → ~13 min denoise + 37s VAE, GPU util 15%→72%**. **BUT the image is garbage** (periodic woven grid, not the prompt) — a *pre-existing* Qwen correctness bug (first-ever completed run; rope-positions/unpatchify/VAE/fp8-conditioning never validated — block rewrite is faithful per-op to the old CPU path, so not the cause). **Next: Qwen parity loop** (diffusers reference, per-component diff) + further perf (per-GEMM Q4_K dequant repeated every step is the remaining ~13 min; needs fused dequant-GEMM or bounded F16 cache).

**UPDATE (2026-06-30) — perf FIXED, correctness still open, bug localized to the transformer forward:**
- **Perf root cause = `QwenImageBlock.Forward` ran all glue on CPU** (DiTUtils/QkNorm/AdaLNModulation reading `DataPointer` → D2H per ~50MB intermediate, ~16×60blocks×CFG). Rewrote GPU-resident mirroring verified `ChromaDoubleStreamBlock` + fused interleaved RoPE into one joint CPU pass (`QwenImageRope.ApplyJoint`, **bit-exact** via `QwenImageRopeFusionTests`). 30+min host-stalled → **~13min, GPU util 15%→72%** (remaining cost = per-GEMM Q4_K dequant, HBM-bound).
- **Conditioning bug FOUND + FIXED (real, needed):** test fed the RAW prompt **padded to 512** through `Qwen3Tokenizer`, no template, no prefix-drop. diffusers `_get_qwen_prompt_embeds` requires the ChatML encode-template + real length + drop 34. Fixed: `Qwen2Tokenizer.EncodeChat(prompt, qwenImageSystem, addGenerationPrompt:true)` + `promptDropIndex:34` → condHidden seqLen **512→12**. (The codebase already had this pattern in `HunyuanImageQwenTextEncoder`.)
- **Image STILL a woven grid at 28 steps with correct conditioning** → conditioning was NOT the grid cause. **Systematically ruled out:** Q4_K GPU dequant (new `DequantizeToF32` + `QwenGgufDequantTests`: matches gguf-python to 4.6e-4), block math (faithful to Chroma), VAE (`QwenImageVaeSmokeTests`: constant→smooth, gradient→smooth gradient, only faint banding), scheduler (standard `x+=v·dt`), pack/unpack (exact inverse, diffusers `(c,py,px)` order), rope GPU/host coherence (activation-cache evict-on-read works). **Remaining: the transformer produces a too-weak/wrong velocity (step-0 std 0.57) → latent stays noise → grid.** Needs a per-component reference dump of the DiT forward (ComfyUI same-Q4_K-weights or diffusers) to localize (rope position VALUES vs diffusers QwenEmbedRope, img_in/txt_in/time_embed, or final AdaLN-continuous layer are the un-diffed suspects).
- VAE has a minor faint-banding artifact (follow-up, not the dominant bug).
- **ROOT CAUSE FOUND + FIXED (2026-06-30, via ComfyUI `comfy/ldm/qwen_image/model.py` component diff):** the final layer `ApplyFinalLayer` (norm_out / AdaLayerNormContinuous) read `shift=firstHalf, scale=secondHalf` — but diffusers/ComfyUI `LastLayer` does `scale, shift = chunk(emb, 2)` (**scale first**). The per-block AdaLayerNormZero IS [shift,scale,gate] (ours was right there), but the final continuous norm is [scale,shift] — they differ. Swapped → velocity std collapsed to 0.57, latent stayed noise → woven grid. **Fix → noisePred std 0.57→1.29, finalLatent std 1.6→0.65, grid→coherent image** (6-step shows clear subject+sky). ComfyUI diff also CONFIRMED matching: rope position ids (img `row/col - len//2`, txt `max(h//2,w//2)+s`), patchify `(c,py,px)`, attention `[txt,img]` concat, block modulation order, SiLU+Linear mod, timestep sinusoid (cos-first, scale 1000). 28-step clean-image confirmation running. See [[adaln-continuous-scale-shift-order]]. |
| - | Flux.2 Dev (32B) | 🔧 | local Q4 gguf | blocked | — | no gen test + 32B > 24 GB |
| - | Ideogram 4 | 🔬 | local nvfp4 single-file | blocked | 🔬 1e-7 | test wants diffusers folder layout + ≥22 GB (4090 only); local copy is single-file nvfp4 — download Comfy-Org folder layout |
| - | Kandinsky 5.0 Lite | 🔧 | ⬇ transformer 12 GB + VAE (`kandinskylab/...T2I-Lite-sft-Diffusers`) | **embeds ready, dl-ing** | — | Embeds generated from on-disk Qwen2.5-VL-7B fp8 (dim 3584, `hidden_states[-1][:,41:]`) + CLIP-L pooled (768) — `dump_kandinsky5_embeddings.py`. Transformer+VAE downloading (text encoders skipped — embeds precomputed). Denoiser-only run after dl. |
| - | OmniGen2 | 🔧 | local fp16 (7.9 GB) | **10-fix plan ready** | — | `NotImplementedException` was STALE — Forward is structurally wired. Real blockers (ref `comfy/ldm/omnigen/omnigen2.py`): **(1) 4 wrong weight-key names** in `LoadWeights` (`caption_embedder`→`time_caption_embed.caption_embedder.1`, time_proj→timestep_embedder.linear_1/2, norm_out.linear→linear_1, proj_out→norm_out.linear_2, delete fabricated norm_out.norm) → null→NRE; (2) missing caption pre-RMSNorm (caption_embedder.0); (3) FFN inner-dim 10240 not 6912; (4) final norm = affine-free LayerNorm eps 1e-6 not RMSNorm; (5) timestep sinusoid width 256 not 2520; (6) timestep double-scaled (pass t=1-sigma); (7) verify velocity sign; (8) CFG needs neg embeds (test passes null→ArgumentException). Mostly S/M. Defer to focused session. |
| - | Anima | 🔧 | local (Cosmos-Predict2 2B) | **RUNNING on 3060 @1024** | — | Unblocked: precomputed Qwen3-0.6B embeds + T5 token-id `.bin` files generated (`tests/python-reference/encode_anima_prompt.py`, last_hidden_state dim 1024). Fits 3060 (MinReq 8 GB). Inspect on completion. |
| - | ERNIE-Image | 🔧 | ⬇ FP8 (`rootlocalghost/ERNIE-Image-FP8` transformer + `Comfy-Org/ERNIE-Image` ministral-3-3b TE, ~15.75 GB) | **downloading, fixes applied** | — | Real e2e (bundled Ministral-3B encoder). Runnable on 4090 w/ FP8. **2 test fixes applied:** (a) `CacheWeightCasts=false` (was `PreloadWeights` → OOM), (b) wired VAE BatchNorm un-norm (`bn.running_mean/var`, eps 1e-4 — was silently skipped → wrong latent scale). MUST use `model.`-prefixed Comfy TE (baidu TE has `language_model.model.` prefix → KeyNotFound). Compiles. Run when dl done + GPU free. |
| - | HiDream i1 | 🔧 | partial (CLIP-L+VAE+llama-tok staged) | **DEFERRED — needs A100/H100** | — | Encoders unconditionally upcast to F32 → ~54 GB peak (T5 19 + Llama 32 + CLIP 3) + 30 GB test gate. Won't fit 24 GB without keeping encoders fp8-on-GPU (larger code change). Arch verified correct (real top-2 MoE, not single-expert fallback). 30 GB transformer dl NOT pulled. Latent bugs noted: T5/CLIP `LoadWeights` skip `TextEncoderQuantNormalizer.Normalize` (fp8_scaled → wrong); worked around w/ non-scaled T5 + fp16 CLIP. |
| - | Lumina2 | 🔧 | local transformer + Flux.1 VAE (symlinked) | **ready to run** | — | Gemma-2 2B embeds generated (`hidden_states[-2]`, dim 2304, `encode_lumina2_prompt.py`). VAE unblocked: symlinked the Flux.1 16-ch `ae.safetensors` (bare keys verified, `VaeConfig.Flux`). 2B → fits 3060. Run after Anima frees the 3060. |

## ⚠️ BASELINE VALIDATED — resolution was the red herring
- **SDXL @ 1024×1024 produces a clean, on-prompt lion-in-sunflowers on the 4090.** Engine, both GPUs,
  cuBLAS 12 AND 13 are all sound. (cuBLAS 12-vs-13 made NO difference for SDXL — identical output; the
  Flux `NOT_SUPPORTED` is an fp8-transient-path-specific issue, separate.)
- **Root cause of the earlier "everything is broken" panic: testing 1024-native models at 256²/512².**
  SDXL@256 = blocky primary-color garbage; SDXL@1024 = flawless. These DiT/UNet models are 1024-trained and
  degenerate badly at low res. **RULE: always test image models at 1024×1024** (512 only for quick smoke,
  never 256). AuraFlow@512 (green banding) and Chroma@512 (noise) must be re-judged at 1024.
- **Validated run recipe:** `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1` (4090) +
  `LD_LIBRARY_PATH=<cu13>/lib:/usr/lib/x86_64-linux-gnu` + `-f net10.0`, gen at 1024².

## Engine change this session (broadly useful)
- **`CudaBackend.CacheWeightCasts`** (new, default true) — set `false` for low-VRAM: forces transient per-GEMM
  fp8/quant→F16 weight cast instead of a resident cache. Cuts large-fp8-DiT weight VRAM ~3×. This is the
  lever that lets AuraFlow (and other big fp8 models) fit the 4090. `CudaBackend.cs` ~line 57 + the
  `LinearImpl` cache branch (~line 332).

## Local weight symlinks created under Models/
SD15, SDXL (sd_xl_base_1.0), SD3.5-Large fp8, Flux-dev fp8, Z-Image-Turbo fp8, Qwen-Image fp8,
Ideogram4 nvfp4, OmniGen2 fp16, Anima, Lumina2, Flux2-dev Q4 gguf; text encoders (qwen3-4b, qwen3-0.6b,
qwen2.5-vl-7b, mistral-fp8, umt5); VAEs (flux2-vae, qwen_image_vae).

## ⚡ DiT PERF: GPU-residency rewrite (root cause of ~40-min runs)
**Definitive diagnosis** (profiler `HARTSY_PROFILE=1` in `NvtxRange` + `CUDA_LAUNCH_BLOCKING=1`): a 2-step Chroma@1024 run = 277s, of which **GPU kernels are only ~35s** (Linear 19.5s, SDPA 12s). The other **~240s is host-side**: the DiT blocks do reshape/concat/split/QK-norm/RoPE on the **CPU** via `DiTUtils.*` (all read `(float*)tensor.DataPointer` → D2H sync of the full Q/K/V tensor + nested CPU loops), `QkNorm.Forward` (CPU), and `FluxRope.Forward` (CPU). Per double-block that's ~12 full-tensor `DiTUtils` ops + 4 QK-norms + rope, ×19 double+38 single blocks × forwards.
**Red herrings ruled out:** RoPE alone (`HARTSY_SKIP_ROPE=1` saved only 5s); weight-cast cache (F16 cache size = param-count, not GGUF size → OOMs Chroma on 24GB regardless of quant); tensor alloc (lazy — never zeroes host mem for GPU-resident tensors). See [[dit-inference-host-overhead]].
**Template:** Ideogram4 (the one fast DiT) does everything via GPU ops — `backend.RmsNorm`/`Permute0213`/`SliceLastDim`/`AffineBroadcastLastDim`/`GatedResidualLastDim`/`ApplyRope`, never touches `DataPointer`.
**Rewrite spec (per DiT block, existing GPU ops unless noted):**
1. QK-norm `_normQ.Forward` → `backend.RmsNorm(out, q_viewed_[*,headDim], _normQ.Weight, _normQ.Eps)`.
2. `DiTUtils.ReshapeToMultiHead([B,S,H,D]→[B,H,S,D])` → `backend.Permute0213(out, in, S, H, D)`.
3. Joint concat/split: concat txt+img in **[S, H·D] layout BEFORE the permute** (contiguous row-concat via `ScatterRowsAfter`/copy-with-offset), permute after; split = `SliceRows`. Avoids interleaved-by-head concat.
4. `FluxRope.Forward` (CPU) → `backend.ApplyRope` (GPU): needs cos/sin as GPU tensors + apply on `[B,L,H,D]` pre-permute (like Ideogram4).
5. Modulation/residual already could use `AffineBroadcastLastDim`/`GatedResidualLastDim`.
**Risk:** `DiTUtils`/`FluxRope` are shared by verified Flux + Flux2/HiDream/Krea2/Lumina2 — but the rewrite only edits `ChromaDoubleStreamBlock`/`ChromaSingleStreamBlock` (used ONLY by `ChromaTransformer`), so it's **Chroma-only** and can't regress other models. Keep changes bit-exact; replicate per-model afterward with a CPU-vs-GPU parity test. Profiler kept (`NvtxRange`), RmsNorm F16 GPU-path fix landed. Temp `HARTSY_SKIP_ROPE` gate in FluxRope to remove.
**→ Full actionable TODO with op-by-op conversion table + gotchas + validation steps: [`TODO_CHROMA_GPU_RESIDENCY.md`](TODO_CHROMA_GPU_RESIDENCY.md).**

## Bugs found (promote to PARITY_VERIFICATION.md once confirmed)
- **AuraFlow / Pile-T5-XL — attention scale `1/√64` instead of `1.0` (FIXED).** `T5TextEncoderConfig.PileT5Xl`
  didn't set `AttentionScale`, so `T5Block` fell back to `1/√head_dim`. All T5/UMT5/Pile-T5 use scale=1.0
  (sibling `T5Base` already does). Wrong scale across all 24 encoder layers → corrupted conditioning. Caught
  by official-source review (diffusers/ComfyUI). Same latent bug confirmed in the shared `Xxl` preset (affects
  Chroma; must be fixed via a Chroma-specific config to avoid touching verified Flux/SD3).
- **AuraFlow — final `norm_out` "missing LayerNorm": FALSE POSITIVE.** A research pass flagged it, but the
  official `AuraFlowPreFinalBlock` genuinely applies no LayerNorm before `x*(1+scale)+shift`. C# was correct.
  (Lesson: verify agent findings against the real upstream source before editing.)
- **AuraFlow — QK-norm uses RMSNorm; official is `qk_norm="fp32_layer_norm"` (non-affine LayerNorm). NOT the
  bug — image came out clean+on-prompt with the existing RMSNorm path once tokenizer+scale were fixed.** The
  "horizontal banding" that triggered this suspicion was a low-res artifact (512 on a 1024-native model).
  Leave as-is unless a future layer-diff shows drift; checkpoint ships no `norm_q/k` weights either way.
- **AuraFlow / Pile-T5-XL — WRONG TOKENIZER (FIXED, was the conditioning bug).** Test fed `t5_xxl_spiece.model`
  (the UMT5/T5-XXL SentencePiece) to a Pile-T5-XL encoder, which uses a **LLaMA-derived 32000-vocab SP model
  with different token IDs** (`pile_t5xl_spiece.model`). Same prompt tokenized to entirely different ids
  (astronaut/horse → `[385,29132,20546,364,4821,263,10435]` pile vs `[46,30059,7494,3,9,4952]` xxl), so the
  model rendered a clean-but-off-prompt image. Fix: `TestPaths.PileT5XlSpiece` now defaults to the real pile
  model file. Confirmed byte-identical (md5 `eeec4125…`) to EleutherAI/pile-t5-xl's `spiece.model`. **Combined
  with the T5-scale-1.0 fix, AuraFlow v0.3 now produces correct on-prompt output → promoted to ✅.**
