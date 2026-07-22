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
