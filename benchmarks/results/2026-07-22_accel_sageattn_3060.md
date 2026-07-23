# SageAttention INT8 kernel (B1/H4) — build + iteration log, RTX 3060

**Item:** INFERENCE_ACCEL_GRIND §H4 — K-smoothed per-row INT8 flash attention, custom PTX.
Correctness oracle: `SageAttentionReferenceTests` (CPU int8 math) + `SageAttnKernelTests` (GPU vs CPU-F32
parity, channel-consistent K outliers, budget 1e-2). Bench: `SageSdpaMicroBench` (warm, 5 trials,
Welch's t) vs the default F32 no-mask dispatch (materialized cuBLAS-TF32 pipeline) on the 3060.

## Iteration log (each step parity-gated before timing; all default-off behind HARTSY_SAGE_ATTN=1)

| Version | Change | [1,24,12288,128] | vs default | Notes |
|---|---|---|---|---|
| v0 | wmma m16n16k16 s8, SMEM O/softmax (FA2-template clone) | 2306 ms | **0.17×** | wmma's undefined fragment layouts force SMEM round-trips + serial per-thread D-loops; Sq%32 gate |
| v1 | raw `mma.sync` m16n8k32-s8 QK^T + m16n8k16-f16 PV, register-resident O + register softmax (no-shuffle S→P repack), BR=64/BC=64 | 329 ms | 1.26× (t=7.7) | 168 regs, 0 spills, 3 blocks/SM; any-Sq (full row guard) |
| +swizzle | 16B XOR swizzle both tiles | 446 ms | 0.92× | conflicts −80% but regs 168→221 → lost a block (occ 24→16%) |
| +bounds(3) | reg cap 170 | 333 ms | 1.24× | cap-induced spills ate the conflict win |
| +BC=32 | halve K/V tile | 305 ms | 1.36× (t=9.9) | 168 regs, 0 spills, 3 blocks; ncu: 69% stalls = SMEM short-scoreboard |
| **v1.3** | **one-shot V pre-transpose to F16 (`sage_v_f16t`, padded rows) + cp.async.cg 16B double-buffered staging** | **74.0 ms** | **5.55× (t=31.5)** | 96 regs (!), 0 spills; the per-block V re-transpose was Sq/64-redundant |

Image shape [1,24,4096,128]: 9.62 ms vs 43.9 ms = **4.57×** (t=9.0).

Parity (all versions shipped): maxErr ≤ 3.3e-4 vs CPU F32 with outlier-injected K (30× under the 1e-2
Sage budget), clean-input control passes, Skv-tail (250) correct, bit-exact run-to-run.
Fixed along the way: cp.async 16B source-alignment fault on odd-Skv per-head kscale bases
(CUDA_ERROR_MISALIGNED_ADDRESS → ks tile staged with scalar loads).

## Roofline sanity

At [1,24,12288,128]: QK^T ≈ 0.93 TFLOP INT8 (~9 ms at the 3060's ~101 TOPS) + PV ≈ 0.93 TFLOP
F16→F32 (~36 ms at ~26 TFLOPS) ⇒ ~45 ms mixed floor; measured 74 ms ≈ **60% of the mixed-precision
ceiling**. Remaining gap: PV dominates — candidates are F16 PV with F16 accumulate (paper-faithful,
needs an accuracy gate) and deeper staging overlap. The default path's 411 ms is consistent with its
TF32 GEMM + 14.5 GB score-matrix traffic.

## Formal BDN run (H4 gate 2 — `SdpaGpuBenchmarks`, BenchmarkDotNet v0.14, 3060, 27 benchmarks)

| Shape (B,H,Sq,Skv,D) | F32 default | F16 (cuDNN fused) | Sage INT8 | Sage vs F32 | Sage vs cuDNN-F16 |
|---|---|---|---|---|---|
| 3: SD3.5M (1,24,1357,1357,64) | 3.63 ms | 0.738 ms | **0.703 ms** | 5.2× | **1.05× (wins)** |
| 4: Flux (1,24,1280,1280,128) | 3.74 ms | 0.999 ms | 1.233 ms | 3.0× | 0.81× |
| 5: Z-Image (1,30,1088,1088,128) | 3.83 ms | 0.955 ms | 1.177 ms | 3.3× | 0.81× |
| 8: video DiT (1,24,16384,16384,128) | 915.9 ms | 130.6 ms | **130.2 ms** | **7.0×** | **1.00× (tie)** |
| 0,1,2,6,7 (D∉{64,128} or tiny Skv) | — | — | ≡ F32 path | fallthrough ✓ | — |

Reading: the INT8 path (F32 in/out — no activation-precision compromise) ties the house's best F16
fused path at the video shape and beats it at D=64; the D=128 small-seq deficit is quant-prologue
overhead (amortizes with Sq·Skv). Non-eligible shapes bit-track the F32 path — the gate leaks nothing.
Follow-up: add a min-`Skv` dispatch gate (tiny-KV cross-attn gains nothing and pays the prologue).

## H4 gate 3 — e2e Wan with HARTSY_SAGE_ATTN=1 (4090): NEGATIVE, root-caused

Wan 5B e2e frames with the knob armed are **bit-identical** to baseline — Sage never dispatched.
Root cause: `WanVideoBlock` calls SDPA with `allowF16: true` (QK are RMS-normed → F16-safe), which the
dispatch chain routes to the **cuDNN F16-cast fused branch before** the F32/no-mask block where Sage
sits. So Wan's attention incumbent is already cuDNN-fused F16 — exactly the path Sage TIES at video
shapes in the BDN table above. Verdict: no e2e win is available on Wan by dispatch order alone; the
e2e-relevant targets for Sage today are models/paths that actually hit the F32 materialized/tiled
branch (no `allowF16`, or masked-fallthrough cases). Future levers, in value order: (1) an F16-ingest
Sage variant competing head-on with cuDNN (needs the INT8-quant-from-F16 prologue + an accuracy gate);
(2) dispatch-preference for proven-win shapes (D=64: Sage BEATS cuDNN 1.05× in the BDN table).

## E1 — F16-accumulate PV (2026-07-22, after gate 3): the cuDNN-beating variant

GeForce Ampere runs f16f16**f32** mma at HALF the f16f16**f16** rate — PV was our floor-dominant term.
E1 ships `sage_attn_int8_v1_*_f16acc_*` entries (knob `HARTSY_SAGE_PV=f16acc`) with two hardening steps,
each forced by a failing gate:
1. **Overflow**: unnormalized flash-O reaches l·|v| → P packed pre-scaled 1/16 (pure exponent shift,
   exact for p∈[0,1]; ×16 folded into the epilogue's 1/l) — headroom to ~349k keys.
2. **Swamping** (caught by the new uniform-attention/constant-V gate at Skv=24576: finite but −11%
   mass): whole-row F16 accumulation loses increments once Σ ≫ ulp. Fix: **per-K-step F16 tile
   accumulation drained into an F32 running O** — f16 mma rate, f32 fidelity. Post-fix the adversarial
   gate reads **maxErr = 0.000 (exact)**; full parity 5/5 at ≤4.2e-4.

Formal BDN rerun (f16acc armed):

| Shape | F32 default | cuDNN F16 fused | Sage INT8 (f16acc) | vs F32 | vs cuDNN |
|---|---|---|---|---|---|
| 8: video (16384², D=128) | 915.1 ms | 130.4 ms ±0.36 | **110.6 ms ±0.95** | **8.3×** | **1.18× (WINS)** |
| 3: SD3.5M (D=64) | 3.62 ms | 744 µs | 738 µs | 4.9× | 1.01× |
| 4/5: Flux/Z-Image (D=128, ~1.1–1.3k seq) | 3.7/3.8 ms | 979/933 µs | 1053/1007 µs | 3.5/3.8× | 0.93× (prologue-bound) |

**This is the kernel-vs-kernel claim that matters: from F32 inputs, with F32-fidelity accumulation,
the INT8 path now beats the production incumbent (cuDNN fused flash) by 1.18× at video-DiT scale.**
Micro A/B trail: f32acc 74.0 ms → naive f16acc 63.9 ms (failed swamping gate) → drained f16acc 64.7 ms
(all gates green) at [1,24,12288,128]; drain cost ≈ 0.8 ms of the 9.3 ms win.

## Caveats (honest frame)

- Baseline is the **materialized cuBLAS-TF32** F32/no-mask path — the incumbent for Wan-class F32
  activations. The cuDNN F16 fused path (Hunyuan/Kandinsky class) is NOT yet comparable: Sage takes
  F32 in/out today; an F16-ingest variant is future work.
- 3060 (SM 8.6) only so far; 4090 (Ada) numbers pending the next GPU window.
- e2e gate (H4 gate 3: Wan step time + SSIM with HARTSY_SAGE_ATTN=1) still pending.

## Dispatch preference + Wan e2e quality gate (2026-07-22, 4090)

Sage (armed) now claims no-mask F32 calls BEFORE the cuDNN-F16-cast branch, gated `Skv ≥ 2048`
(below: prologue-bound, measured 0.93×; Wan's 512-token cross-attn correctly stays on cuDNN).
exp2 log2-domain softmax folded into the quant scale (13/13 parity green, perf-neutral — kept as
strictly fewer ops).

Wan-5B 480p/33f/50st e2e with `HARTSY_SAGE_ATTN=1 HARTSY_SAGE_PV=f16acc`:
- **Dispatch confirmed** (frames byte-differ from the unarmed baseline), deterministic ×3.
- **Quality: mean pixel drift 0.5% (1.2–1.3/255), same clip identity, eyeball-identical** — the
  INT8 path survives 50 steps × 30 blocks × 2 attentions with no visible cost.
- **Wall: neutral (68.8 vs 68.0 s)** — attention ≈6% of the 5B forward at 480p; 1.18× on 6% is
  <1% e2e. Honest e2e frame: Sage's end-to-end value on Wan-class models needs 14B/720p+ seqlens
  (attention share 25%+); its decisive wins remain the F32-materialized-path shapes (3–8×).

## F16-ingest + 14B e2e (2026-07-22, session 3 block)

**Wan-14B fp8 e2e (4090, 480p/33f/12-step probe): 6.55 s/step vs 6.85 baseline = 1.047× — the first
visible e2e win** (≈15 s saved per 50-step gen; matches 1.18× on the ~25% attention share).

**F16-ingest shipped** (`sage_quant_{q,k}_int8_f16h`, `sage_k_mean_f16h`, `sage_v_f16t_h`, OUT16
epilogue → `sage_attn_int8_v1_*_f16io`; branch-1 dispatch behind `HARTSY_SAGE_F16_MIN_SKV`,
default 8192). Parity: F16-ingest vs CPU-F32-on-identical-values 2.3–3.1e-4; full sage suite 7/7.
Crossover vs the CAST-FREE native-F16 cuDNN incumbent (3060, trimmed steady-state): **1.15× @12288,
1.11× @8192, ~parity @4096** — Qwen-Image-class 4k seqs need the ldmatrix floor-work to flip.

**Anomaly logged**: Skv=8192 (power-of-2) is disproportionately slow for BOTH paths (12288 does 2.25×
the work in 1.03× the time) — suspected L2 set-aliasing on strided V-transpose reads; fix candidate:
offset `skvPad` off powers of 2. ncu the prologue next session.

F32-caller crossover (vs cast-paying cuDNN branch, trimmed): 1.15–1.25× everywhere ≥2048 → the
`Skv ≥ 2048` F32 gate stands. Image-fleet map: F32 callers = Sage wins now; F16-native (Qwen/Hunyuan/
Kandinsky) = wins ≥8k, parity at 4k pending ldmatrix; Ideogram4 D=256 = separate instantiation;
Chroma = masked (needs mask support).

## ldmatrix experiment + anti-aliasing pad + first Ada numbers (2026-07-22, session 3)

- **ldmatrix.x4 PV fragment loads: NEGATIVE (reverted).** Parity green after fixing a double-transpose
  (Vt is pre-transposed → NON-trans distribution is the B-fragment layout), but 1.5% slower than the
  manual u32 loads: post-cp.async the loads are latency-hidden behind the mma chains, and the per-lane
  swizzled-address ALU outweighed the instruction savings. Documented in-kernel.
- **skvPad anti-aliasing**: exact-power-of-2 pads ≥2048 now bump +8 (pure-pow2 Vt row stride aliases
  every d-row into one L2 set group — the suspected Skv=8192 pathology). Parity 7/7. Ampere
  verification PENDING (3060 was taken by another agent mid-bench — contaminated run discarded);
  no pathology on Ada to begin with (72 MB L2).
- **First 4090/Ada crossover (F16-native incumbent, trimmed steady-state):** 12288: Sage 10.3 ms vs
  cuDNN 12.3 = **1.19×**; 8192: parity (5.14 vs 5.21); 4096: cuDNN ahead (1.38 vs 1.52). The
  HARTSY_SAGE_F16_MIN_SKV=8192 default holds on Ada. Ada absolute times ≈ 6–12× the 3060's (as
  expected of the tier gap). NOTE for all micro A/Bs: quote TRIMMED means — cuDNN's first post-init
  call carries a plan-search outlier (2–8× the steady state).

**CORRECTION (clean 3060 rerun):** the Skv=8192 "pathology" was CONTENTION — another agent's job began
mid-bench (cuDNN recovered identically, 69→33 ms, without touching the padded buffer). The anti-aliasing
pad stays as principled hygiene but fixed nothing measurable. Clean Ampere F16-native crossover
(trimmed): 1.03× @4096, 1.09× @8192, 1.15× @12288 — monotonic; the 8192 gate default stands.

## Kandinsky5-Lite T2V e2e (4090, F16-native path, forced dispatch MIN_SKV=1024)

Dispatch ✓ (frames byte-differ), quality ✓ (mean pixel drift 0.15%, eyeball-pristine), speed
**785 vs 795 ms/step (~1.3%)** — parity-zone outcome at its ~4k seq, exactly as the Ada crossover
predicted. The crossover model is now e2e-validated across three models: Wan-5B neutral (6% attn
share), **Wan-14B +4.7%/step** (25% share), K5-Lite +1.3% (parity seq). Conclusion: the shipped
gates (F32: Skv≥2048; F16: Skv≥8192) are correctly placed; F16-native e2e wins arrive with
Hunyuan/Wan at 720p+ seqlens.

## HunyuanVideo 13B @ 720p e2e (4090) — the F16-ingest arc's capstone

1280×720, 17f (~18k tokens), 10-step probe, bf16 DiT (Comfy-Org repack, freshly staged after an
F-Lite prune — user-approved):
- **baseline 6.64 s/step → armed 6.03 s/step = 1.10× end-to-end** (92.1 vs 98.3 s total; ≈30 s saved
  on a 50-step 720p generation)
- dispatch ✓ (frames differ), **quality ✓ (0.3% mean pixel drift, eyeball-pristine)**

e2e validation table, final (all predictions from the micro crossover curves):

| Model / config | seq / attn share | predicted | measured |
|---|---|---|---|
| Wan-5B 480p | 3.5k / ~6% | neutral | neutral ✓ |
| K5-Lite T2V | ~4k (parity zone) | ~neutral | +1.3% ✓ |
| Krea2-Turbo 1024² | 4.4k, F16 gate | no dispatch, no regress | bit-identical ✓ |
| Wan-14B 480p | 3.5k / ~25% | +4–5% | **+4.7%/step** ✓ |
| **HunyuanVideo 720p** | **~18k, deep win zone** | ~+10% | **+10.2%/step** ✓ |

The crossover model predicted every e2e outcome. Shipped gates final for this round:
F32 ingest Skv≥2048; F16 ingest Skv≥8192 (both arches).

## Kernel-floor probes (session 4): BC=64 NEGATIVE

Floor profile of the shipped f16acc kernel: tensor 46%, IPC 2.34, occupancy at its 3-block cap —
issue-bound on per-K-step overhead. BC=64 experiment (halve K-steps; SMEM 49.7 KB → 96 KB opt-in,
2 blocks/SM): parity 7/7 but 69.0 vs 64.7 ms @12288 and 9.6 vs 8.3 @4096 — the occupancy loss beat
the overhead amortization (f16io variant also hit 243 regs). REVERTED; BC=32 is the validated
sweet spot. Remaining floor levers are structural (warp specialization / deeper software pipeline) —
diminishing returns territory at ~60% of mixed roofline; deprioritized behind coverage work.

## Ideogram4 restored + w13 fusion gate (2026-07-22): NEGATIVE both ways — guarded off

Full stack re-staged (2 transformers + Qwen3VL-8B TE + Flux2 VAE; F-Lite + LTX-2.3 pruned with
user approval). Measured (4090, 1024², 20 steps, seed 42):
- **Baseline (current shipping speed): 1.237 s/step** rock-steady, 59.7 s cold incl. ~19 GB load.
- Fused (`HARTSY_FUSED_FFN=1`): 1.290 s/step (−4%) AND **F16 output degenerate** (repeating-texture);
  F32 fused is **bit-identical** to baseline — math correct, defect is F16-path-specific.
- Op-level bisect (new `FusedFfnF16BisectTests`): the exact op sequence (fp8+scale, F16 acts, damp,
  slice, undamp) passes on BOTH GPUs at synthetic shapes → the delta is this checkpoint's
  **comfy_quant/weight_scale companion format**, whose per-key descriptors the fused tensor drops.
- Shipped guard: `FuseSwiGluPairs` now SKIPS pairs carrying comfy_quant/weight_scale companions —
  fusion cannot mis-fire on this format again. Also perf verdict independent of the bug: at 4k-token
  image workloads the GPU is already full — fusion's launch-count win doesn't apply (it was measured
  for tiny decode GEMVs), and the slice copies cost real bandwidth. **Ideogram4 keeps the split FFN;
  the fusion utils remain correct and valuable for the LLM decode case they were re-scoped to.**

**Ideogram4 through SwarmUI (production engine, 4090):** staged into the Swarm tree via symlinks;
gen succeeded first try. **Warm 19.2 s** (cold 51.7 s incl. load) at 1024²/20 steps — matches the
19.5 s production record. Output eyeball-verified (on-prompt astronaut-on-horse photograph).

### ⚠️ CORRECTION (2026-07-22, same day, later session): the F16-fused "defect" was a HARNESS bug

The standalone test family (`Ideogram4GenerationTests` and every A/B built on it) fed the RAW
`Qwen3Tokenizer.EncodeChat` output — **padded to 2048 tokens with BOS(151643) and carrying a
`<think></think>` block** — while the engine recipe (`Ideogram4RecipePipeline`) trims the pad and
disables the think block. Ideogram attends UNMASKED, so ~2020 pad-token TE rows drowned the ~18 real
prompt tokens: **the standalone BASELINE was itself degenerate** (deterministic texture fields —
face-tiles in F16, gold pebbles in F32; the two differ because an unconditioned trajectory is
precision-chaotic, NOT because F16 is broken). Nobody had eyeballed the standalone baseline — only
fused-vs-baseline diffs and the (always-correct) Swarm image. Consequences:
- The "comfy_quant/weight_scale companion" causal claim for the fused-F16 degenerate output is
  **unsubstantiated** — fused F16 was compared against an equally-degenerate baseline. The
  `FuseSwiGluPairs` guard stays (harmless; fusion is opt-in and re-scoped to LLM decode), but the
  mechanism story is retracted.
- The fused **−4% perf** number was measured at the padded 6144-token sequence, not the production
  ~4120 — treat as unmeasured (moot: Ideogram keeps the split FFN regardless).
- The standalone step time 1.237 s/step was the PADDED shape; the fixed harness matches the
  Swarm/production shape (~0.96 s/step). All model files sha256-verified against
  Comfy-Org/Ideogram-4 (TE + both DiTs + VAE checked; first three match, VAE is the shared
  `VAE/Flux/flux2-vae.safetensors` used by the verified Flux.2 pipelines).
- Both test harnesses now mirror the engine tokenization (`includeThinkBlock: false` + right-pad
  trim); fixed-harness probe produces an on-prompt astronaut image at warm ~20 s. Same disease as
  the Wan standalone-test pad-row bug (2026-07-22) — **house rule: standalone harnesses must copy
  the engine's conditioning path exactly, and baselines get eyeballed BEFORE any A/B is trusted.**


## Occupancy experiment: `__launch_bounds__(128,4)` coerced-spill A/B (2026-07-23, 3060)

The independent H0 ncu capture ([2026-07-23_h0_ncu_top5_3060.md](2026-07-23_h0_ncu_top5_3060.md))
diagnosed the v1 kernel as latency/occupancy-bound: 24.9% occupancy, register-capped (3 blocks/SM),
DRAM 3–5%, tensor-pipe 37–47%. Cheapest test: force a 4th resident block via launch bounds and let
the compiler spill. Raw walls looked like a 5–10% win across 2k–6k shapes — **but the in-suite cuDNN
control moved by the SAME margin in the same direction** (run A rode boost clocks from a just-finished
heavy job on the same card). Normalized to the control, 3-block vs 4-block is equal within 1–2%.
**Verdict: NEUTRAL — reverted; the occupancy hypothesis is not confirmed by coerced spills.** True
register reduction (spill-free) remains the open lever; it is structural surgery (the softmax/O state
is the register budget) and stays deprioritized at ~60% mixed roofline. Method note for the fleet:
never A/B kernel builds in separate runs without an in-run control or locked clocks — the paired
cuDNN baseline is what caught this.


## Production (Swarm) Wan-14B on/off spot check (2026-07-23, 4090, default-on shipped)

Wan-2.2 T2V high-noise 14B via `/API/GenerateText2Image`, 704×480/25f/20st, seed 42, one arm per
Swarm process (the dispatch switch is process-static): DiT loop **86.1 s (Sage on) vs 87.5 s (off)**
≈ 1.6% — single trials, direction consistent with the H4 record (+4.7%/step at 832×480, where the
attention share is larger). The 832×480 request OOMs mid-VAE-decode beside the resident 14B expert
pair on 24 GB (decode-headroom estimate undershoots; pre-existing tightness, noted for the serving
backlog). Images remain byte-identical under the flip; flagships hold (4.44 s / 2.74 s).

## HunyuanVideo production path opened (2026-07-23, 4090) — the +10.2% arch now Swarm-drivable

The biggest recorded Sage win (HunyuanVideo, ~18k-token seqlens, +10.2%/step at 720p) had no
production path: SwarmUI refused the architecture because the extension's `ModelSupport` table had
no `hunyuan-video` compat-class entry — the engine recipe and `VideoRecipeRegistry` registration
were already live, so the fix was one mapping line (committed by the user in extension `f90a512`
together with a `2.0.0-alpha.2` pin bump). Verified e2e via `/API/GenerateText2Image` at
512×320/25f/20st: prompt-matched coherent motion across decoded frames 1/13/25; warm sampling
~1.6 s/step wall (Sage on, default) vs the 2.15 s/step pre-Sage record at this geometry. Ops notes:
first attempt died disk-full (root at 100%, 7.7 GB free < 9.1 GB LLaVA download) — resolved by
symlinking the existing engine-side LLaVA into Swarm's `Models/text_encoders/`; 720p Swarm runs
(the deep-win zone) remain to be exercised. Post-redeploy flagship gates re-measured and holding:
Krea2-Turbo 4.41 s, Z-Image-Turbo 2.69–2.90 s. Method note: the flagship gate figures are
ENGINE-INTERNAL `txt2img complete` times from the Swarm log, NOT API wall — API wall runs 1.5–2.5 s
higher under concurrent-agent CPU load (loadavg ~5 during this check) and will false-fail the
Z-Image ≤3.2 s gate if misread that way.
