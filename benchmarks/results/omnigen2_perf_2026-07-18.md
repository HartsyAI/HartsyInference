# OmniGen 2 perf pass — 2026-07-18 (RTX 4090, 1024², 20 steps, cfg 4, warm median via SwarmUI API)

Prompt: "a highly detailed photograph of an astronaut riding a horse across a martian desert at golden
hour, dramatic lighting, sharp focus". 1 warmup + 3 timed gens, random seed each. Both backends on the 4090.

## Scoreboard

| Build | Hartsy warm median | GPU util | vs prev | vs ComfyUI 13.0s |
|---|---:|---:|---|---|
| Baseline (re-benched, `alpha.53`) | **132.6s** | 29% | — | 10.2× slower |
| + Device RoPE (`alpha.54`) | **32.66s** | 77% | 4.06× | 2.51× slower |
| + cuDNN F16 SDPA (`alpha.55`) | 32.48s | 77% | ~neutral | 2.50× |
| + F16 activations (`alpha.56`) | ~32.4s | 77% | ~neutral* | — |
| + Drain-free loop, F32 (`alpha.58`) | **22.96s** | 84% | 1.42× | 1.77× slower |
| + Drain-free loop, F16 (`alpha.59`) | 20.53s | 84% | 1.12× | 1.58× slower |
| + Device Concat/Split (`alpha.60`, SHIPPED) | **~19.1s** | ~90% | 1.07× | **1.47× slower** |

Baseline walls: 132.61 / 135.47 / 113.49. Final (alpha.59) walls: 20.44 / 20.64 / 20.53, peak 19.5 GB.
ComfyUI reference = **13.0s** (measured 07-10 on the same 4090 / model / params; ComfyUI perf is stable —
not re-run this session to avoid the backend-toggle trap, but re-confirmable on request).

\* F16 was *neutral* until the loop went drain-free — the wall was host-glue-bound, so faster GPU compute just
widened the idle gaps. F16 only pays off once the host syncs are gone.

## What changed (levers, in order)

1. **Device RoPE (THE win, 4×).** `OmniGen2Block` was calling a host `rope.Apply` loop that drained Q and K to
   the CPU, rotated, and re-uploaded — **~72 D2H round-trips per forward × 40 forwards ≈ 2900/gen** (GPU util
   29%). Wired the existing device `WanRopeInterleaved` on the pre-permute `[B,S,H,D]` layout with cached
   `[S,headDim]` tables (`OmniGen2Rope.GetOrBuild{Text,Image,Joint}Tables`). Bit-equivalent. util 29 → 77%.
2. **cuDNN F16 SDPA** (`allowF16: true`, D=120 gate) + **F16 activations** (`DitDtype.Act`, Lumina2/Krea2
   pattern — stream tensors F16, modulation/norm params F32). Coherent, but neutral while host-bound.
3. **Drain-free loop.** Patchify the seeded noise to token space ONCE (`DiTUtils.PatchifyNCHW`), keep the latent
   GPU-resident across all 40 forwards via `OmniGen2Transformer.ForwardPacked` (skips the per-forward host
   patchify/unpatchify), and do CFG-combine + flow-match Euler in one in-place device op (`IBackend.CfgEulerStep`)
   with a device unpatchify at the end. Removed the per-step D2H syncs (util 77 → 84%) AND the
   `cpu-glue-async-race` crash. F32: 22.96s.
4. **Concat/Split F16 OOB fix (cross-model bug).** `DiTUtils.ConcatAlongSeqDim`/`SplitAlongSeqDim` were
   **F32-hardcoded host loops** (`float*` + `sizeof(float)`); fed F16 tensors they over-read the half-size
   buffers by 2× → heap corruption → the intermittent F16 segfault. Made them **dtype-aware** (byte copies +
   output inherits input dtype; F32 behaviour byte-identical). This unblocked stable F16 → 20.53s. Benefits any
   model that feeds F16 activations through these shared helpers.

## Remaining path to beat ComfyUI 13.0s — a project-scale effort, BLOCKED on architecture

`ForwardPacked` is now fully device-resident (device Concat via `backend.Concat`, device split via `SliceRows`).
The forward IS therefore CUDA-graph-capturable — but capture is **blocked by OmniGen2's CFG**:

- **cond and uncond have DIFFERENT text lengths.** The loader encodes each prompt to its true token count
  (`Qwen3Tokenizer`, ComfyUI omnigen2.py parity — cond/uncond as separate unpadded forwards). So the two
  per-step forwards have different joint sequence lengths → different graph shapes. The `StepGraph` API is
  single-slot (one `StepGraphOwner`), so it cannot hold both. Same blocker defeats **batched CFG**.
- **Unblocking it needs a text-padding + attention-masking rewrite** (pad cond/uncond to a common L, mask the
  pad in the joint SDPA, parity-verify masked==unpadded) — correctness-risky — *then* the CUDA-graph fixed-buffer
  plumbing (fixed latent/temb/caption buffers, capture-warmup, owner management; capture also risks OOM beside
  the ~19.5 GB resident DiT — cf. the Krea2 step-graph 901/OOM notes).

Ceiling analysis: util ~90% → even at 100% ≈ 17s, and F16 GEMMs barely moved the needle, so the wall is the
**many small memory-bound kernels** (norm/permute/gate/rope/SDPA) × 36 blocks × 40 forwards + launch overhead.
CUDA graph alone → est. ~15–16s (still > 13s). Beating 13s ALSO needs **op fusion** (fused QKV: 3 attn Linears→1;
fused FFN linear_1+linear_3: 2→1 — no text-length issue, moderate effort each, ~5–10% apiece). fp8 won't help
(not GEMM-bound). Net: beating 13s is a multi-lever project (fusion + padding/mask + graph), not a single lever.
