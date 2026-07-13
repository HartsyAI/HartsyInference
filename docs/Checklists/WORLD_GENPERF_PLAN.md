# World / Interactive gen-perf — audit + optimization plan (2026-07-12)

Applying the image/video/audio gen-perf playbook (`VIDEO_GENPERF_PLAN.md`, memories `radiance-perf-pass`,
`cuda-graph-step-capture-recipe`, `vae-host-loops-hidden-20s`, `image-genperf-host-glue-wins`) to the
**interactive world models**. Same principles, but the metric is **step latency / FPS** (real-time AR
loops), not seconds-per-clip. Target for interactive models is 25–40 FPS (Phase 10 gate:
MG2 25 FPS @540p / MG3 ≥10 FPS @720p on a 4090).

## The playbook, distilled (order it wins)

1. **Wall-clock phase probes FIRST** — `[world-phase]` logs (VAE-encode seed / per-step DiT / ActionModule /
   VAE-decode / history-roll+FOV / integrate+Plücker). No world model has probes yet.
2. **GPU-residency ports** — replace every host `DataPointer` loop in DiT blocks / ActionModule / VAE with
   existing device ops (`AffineBroadcastLastDim`, `GatedResidualLastDim`, `SliceRows`, `WanRmsNormChannel`,
   `Modulate`, `LayerNormNoAffine`). Bit-identical, zero numeric risk, biggest wins.
3. **cuDNN fused SDPA** (`HARTSY_SDPA_CUDNN=1`, `allowF16`) — mask-null only, D∈{64,128}. Spatiotemporal
   axial attn + ActionModule mouse-self / keyboard-cross attn are the sites.
4. **F16 activations** (`HARTSY_DIT_F16`) — halve bandwidth on norms/elementwise; per-arch opt-in.
5. **CUDA-graph step capture** (`HARTSY_DIT_GRAPH`) — AR loops are FIXED-shape, single-forward (distilled
   1–4 step), resident → **the best graph-capture regime in the whole engine**. Needs a drain-free step
   (device sampler, fixed boundary buffers).
6. **Caching + residency** — `HARTSY_KEEP_MODELS`, RoPE-table memo, per-frame timestep-embed cache,
   image-cond/CLIP cache across segments, FOV-selection GPU kernel (reference: single biggest FPS knob).
7. **Async hygiene** — no per-step `TrimMemoryPool`, no pageable per-step H2D, no mid-forward host reads.

## Inventory — what we support

| Model | Params | Status | Fits 4090 (24 GB)? | Weights | Metric regime |
|---|---|---|---|---|---|
| **DIAMOND** (Atari) | ~tiny CNN U-Net | ✅ verified (CUDA+CPU, bit-exact) | yes (CPU too) | ungated `eloialonso/diamond` | pixel-space EDM, 3-step; launch-bound → **graph** |
| **Oasis-500m** | 500M DiT-S/2 | ✅ verified (CUDA/3060, corr 1.0) | yes | ungated mirror `camenduru/oasis-500m` + `vit-l-20` VAE | DDIM v-pred 10-step AR; small → launch-bound |
| **Matrix-Game 2.0** | 1.8B Wan2.1 | 🔧 DiT fwd parity only (ActionModule pending) | yes | Wan2.1 DiT + 16ch VAE + CLIP-ViT-H (DL + `.pth` conv) | DMD 3-step AR; glue+compute |
| **Matrix-Game 3.0** | 5B Wan2.2-TI2V | 🔧 canned-action built, numerics pending | yes (24 GB min) | Wan2.2 DiT + VAE + umT5 (DL) | FlowUniPC/DMD; ActionModule + FOV memory |
| **Hunyuan-GameCraft 1.0** | 12.5B HYVideo | 🔧 structural, numerics pending | **NO** (40 GB+) | user `.pt` | PCM+CFG 8-step; big-GPU only |
| **Cosmos-Predict1 V2W** | 13B AR | ❌ not started | NO | — | FSQ tokenizer + AR transformer |

## Host-glue surface (audit, `DataPointer` density = the Krea2 disease)

| File | DataPointer reads | Note |
|---|---|---|
| `MatrixGame3ActionModule.cs` | 23 | mouse self-attn / keyboard cross-attn — **novel surface, not shared with video** |
| `OasisSpatioTemporalBlock.cs` | 12 | axial spatial + causal temporal, each with adaLN-zero |
| `OasisDit.cs` | 12 | per-frame `c[t]=TimestepEmbed+Linear(action)` |
| `MatrixGame3Pipeline.cs` | 13 | segment loop, history roll, integrate/Plücker |
| `MatrixGame2Pipeline.cs` | 12 | AR 3-frame blocks, DMD renoise |
| `OasisPipeline.cs` | 10 | AR frame loop, Diffusion Forcing |
| `MatrixGame3Transformer.cs` | 8 | memory-augmented sequence, readout |
| `Diamond/*.cs` | ~17 total | tiny; U-Net + EDM precond + sampler |

## Results log

> ### ✅ Round 1 (2026-07-12) — Diamond + Oasis-500m real perf runs on the 4090 (verified, parity preserved)
> Both ungated verified models downloaded, references re-dumped, parity reconfirmed on CUDA (Diamond 3/3, Oasis
> 4/4), then residency-ported with **existing device ops only** (zero backend/kernel edits), each guarded by a
> parity assert in a new gated FPS harness (`DiamondGenPerfTests`, `OasisGenPerfTests`). Metric = the AR per-frame
> compute, 4090.
>
> **DIAMOND (Atari, tiny CNN U-Net + EDM, 3-step):** AR-rollout **223 → 36 ms/frame (6.2×), 4.5 → 27.8 FPS**
> — past the 25-FPS interactive bar. Bit-exact (coherence mean/std unchanged; parity 3/3 held). Wins, in order:
> `DiamondOps.AdaGroupNorm` host loop → `GroupNorm(x, γ=1+scale, β=shift)` (44 loops/forward, 223→80.6 ms, 2.77×);
> `ConcatChannels`/input `cat` host memcpy → `backend.Concat` (12/forward, 80.6→36 ms, 2.19×). Residency floor
> reached — remaining ~12 ms/step is pure launch overhead (many small kernel launches ×3 steps + the pixel-quantize
> D2H syncs). **Remaining lever (not built):** CUDA-graph the fixed-shape 3-step denoise; blocked only by the
> lack of a device pixel-quantize op (clamp→256-level truncation), which forces the one host readback/step. A tiny
> quantize PTX kernel unlocks a drain-free step → graph capture (the launch-bound regime where it wins wall).
>
> **Oasis-500m (DiT-S/2, 16 blocks, 576 tokens, dim 1024):** per-DiT-forward **2342 → 1327 ms (1.77×)**, parity
> corr 0.99999929 unchanged (block0/blockLast taps corr ~1.0). Ported the per-frame adaLN glue in
> `OasisSpatioTemporalBlock` bit-exactly: `Clone` → `backend.CopyInto`; host `DiTUtils.LayerNormNoAffine` →
> `backend.LayerNormNoAffine`; `ModulatePerFrame` → `SliceLastDim`+`AddScalar`+`AffineBroadcastLastDim`;
> `GatedAddPerFrame` → `SliceLastDim`+`GatedResidualLastDim` (working tensors reshaped to rank-3 `[T,sp,dim]` so
> the broadcast ops index the frame axis). **Dominant remaining lever (not built):** the attention island —
> `SplitHeads`×3 / `MergeHeads` / `RotateBhsd`×2 per attention (192 host round-trips/forward). It's harder than the
> broadcast ports: Oasis uses **interleaved (2i,2i+1) + PARTIAL axial** RoPE on a `[b,h,s,d]` layout, which the
> stock `backend.ApplyRope` (rotate-half) and `ApplyRopeInterleaved` (full-dim, `[b,s,h,d]`) do NOT match →
> needs a dedicated device axial-rope apply + `Permute0213` head split/merge (or a small kernel). Expected once
> device-resident: the Krea2/Chroma-class 4–20× (the host attention loops are the floor). FinalLayer modulation +
> patchify/unpatchify + `BuildCondition` host add are smaller follow-ups.
>
> Harness recipe (both): env-gated (`DIAMOND_PERF=1`/`OASIS_PERF=1` + `*_WEIGHTS`/`*_DIT`+`*_REF` **absolute**
> paths + `PARITY_BACKEND=cuda`); warmup then timed window with `backend.Sync()`; parity assert vs reference each
> run. Weights: Diamond `eloialonso/diamond` Breakout.pt (54 MB), Oasis `camenduru/oasis-500m` (`.pt`→safetensors).

## Per-model perf-run plan

### DIAMOND — RUN NOW (tiny, verified, ungated) — the graph-capture proof case
- **Fits, cheap.** Real-time metric. 3-step Karras+Euler, fixed 64×64, 4-frame history → ideal fixed-shape
  single-forward loop.
- Plan: (0) download weights + build reference dump → confirm parity still ✅; (1) `[diamond-phase]` wall
  probes on an AR rollout (per-step U-Net vs sampler vs history); (2) residency ports of the ~17 host loops;
  (3) **CUDA-graph the 3-step denoise** (fixed shape, resident) — expect launch-bound win; (4) FPS before/after.

### Oasis-500m — RUN NOW (small, verified, ungated)
- **Fits.** 10-step DDIM v-pred AR with Diffusion Forcing, 32-frame sliding window, 360×640.
- Plan: (0) download `oasis500m.safetensors` + `vit-l-20.safetensors`, reference dump → reconfirm corr 1.0;
  (1) `[oasis-phase]` probes on a real RGB-in/RGB-out rollout; (2) residency ports (SpatioTemporalBlock 12 +
  OasisDit 12 + pipeline 10); (3) cuDNN SDPA on the axial attns (mask-null? check); (4) graph the 10-step
  per-frame denoise; (5) FPS before/after, byte-exact-vs-baseline gate.

### Matrix-Game 2.0 — STRUCTURAL (fits, but ActionModule numerics unverified — correctness before perf)
- Blocked on: (a) ActionModule parity, (b) weight download + `.pth`→safetensors (Wan2.1 VAE, CLIP-ViT-H).
- Do the **bit-identical residency ports** now (safe regardless of numerics) + probe scaffold; defer wall
  bench until ActionModule parity lands. Inherits every Wan `WanVideoBlock`/`WanDitOps` port already shipped.

### Matrix-Game 3.0 — STRUCTURAL (flagship; fits 24 GB but numerics pending + big download)
- Mirror the LTX-2.3/Wan-14B video directive: residency ports + graph scaffold with clean eager fallback so
  big-GPU users benefit; verify parity on the CPU tiny-config; defer wall bench to real-weight validation.
- Reuses shipped Wan2.2 ports (RoPE memo, device final-layer, temb drain fix). New surface = ActionModule
  (23 host loops) + FOV-memory selection (CPU port today → GPU kernel = biggest FPS knob per reference).

### Hunyuan-GameCraft 1.0 — STRUCTURAL ONLY (12.5B won't fit 4090)
- Same directive: implement residency ports + graph path structurally, verify the capture path falls back
  to eager cleanly, defer wall proof to a 40 GB+ user. Inherits HunyuanVideo ports (already partly done).

### Cosmos-Predict1 V2W — not started; out of scope for this pass.

## Method rules (non-negotiable, from the war)
- Phase probes before any op-level work; wall ≠ op-profile means an un-instrumented host phase.
- **Correctness before perf** — do NOT optimize MG2/MG3/GameCraft's numerically-unverified paths beyond
  bit-identical residency ports; a fast wrong AR loop compounds error every frame.
- GPU-shared box: hard-gate every GPU run on `nvidia-smi` idle wait-loop; prefer 3060 for fits, take turns.
- Bit-identical ports first (residency/caching), numeric-risk (F16/graph) behind flags with per-stage relL2.
- Interactive metric = **step latency / FPS**, and coherence of the rendered rollout (not just finite floats).
