# Quantization and Low-Precision Inference for Diffusion / Transformer Backbones

Research notes for HartsyInference (pure C#/.NET 10, custom PTX kernels, cuBLAS/cuBLASLt GEMM via P/Invoke, no PyTorch at runtime).

Companion to `QUANTIZATION_DIFFUSION.md`, which covers component-level sensitivity, GGUF on-disk formats, and reference engines (sd.cpp, ComfyUI-GGUF). This doc covers the algorithmic and kernel side: PTQ method families, FP8/cuBLASLt mechanics, SVDQuant/NVFP4/MXFP4, INT8/INT4 GEMM paths, low-precision attention, and an implementability ranking for the custom C# + PTX + cuBLASLt engine.

Target hardware: RTX 3060 (SM 8.6, Ampere, native INT8 tensor cores, NO native FP8) as the primary local card, plus cloud Ada (SM 8.9) / Hopper (SM 9.0), with Blackwell (SM 10.x/12.0) noted for the future.

Engine starting point (already built): a native FP8 cuBLASLt GEMM path (per-tensor scale folded into alpha) for Ada/Hopper, and an FP8 -> F16 cast path on Ampere.

---

## Executive summary / what to build first

1. The single most important hardware fact: on the RTX 3060 (Ampere SM 8.6), INT8 tensor cores (IMMA) are native and run at roughly 2x FP16 rate, while FP8 tensor-core MMA does not exist until Ada (SM 8.9). So on the 3060, INT8 is the only true low-precision tensor-core GEMM. FP8 weights can only be a storage/cast format there (upcast to F16 for compute).

2. Weight-only formats (GGUF K-quants, NF4) are training-free and weight-loadable, but on a single-batch diffusion forward pass they are a memory/VRAM win, not a throughput win. ComfyUI-GGUF and bitsandbytes both dequantize a tile to FP16 then run a normal GEMM. There is no quantized matmul in the diffusion GGUF ecosystem.

3. The biggest real speedups come from W8A8 / W4A4 (both operands low precision) using the INT8/INT4 tensor cores, plus quantized attention (SageAttention). These need calibration and custom fused kernels, not just a dequant kernel.

4. SVDQuant's 4-bit path is implementable on Ampere TODAY via its INT4 (W4A4) branch; only the NVFP4 variant needs Blackwell.

Recommended build order is in the final section.

---

## Family 1: Post-Training Quantization (PTQ) for diffusion

All methods here are training-free PTQ (no backprop at runtime); none ship a drop-in quantized checkpoint in a standard container. The published repos are PyTorch calibration scripts that emit scales/zero-points you would export yourself.

### Why naive PTQ fails on diffusion (the three root problems)

- Time-step-varying activation distributions: the same layer sees very different activation ranges at t=999 vs t=0 because the input is iteratively denoised. A single static per-tensor scale calibrated on one slice of timesteps is badly wrong on others. A documented example: a layer where >99% of values sit in [-0.6, 1.7] but the full range is [-10, 34], and that range shifts per timestep.
- Activation/channel outliers: a few channels run 10x-50x larger than the rest, stretching the per-tensor quant range so the non-outlier channels lose almost all their bits. Motivates per-channel handling and smoothing.
- Accumulated error over denoising steps: quant error at step t feeds the input of step t-1, so errors compound across 20-1000 sampling steps. A small per-step bias turns into large FID drift. Motivates bias/variance correction.
- Secondary: bimodal "shortcut"/skip-connection activations in U-Nets (concatenation of two distributions) need split quantization.

### Calibration strategies (cross-cutting)

- Sample count is small: PTQ4DM uses 1024 calibration samples; Q-Diffusion-family methods use a few thousand intermediate inputs collected across timesteps.
- Do NOT sample t uniformly. PTQ4DM's NDTC samples t from a skew/normal distribution N(mu, T/2) with mu <= T/2, biasing toward the timesteps that matter most for final fidelity. ADP-DM partitions timesteps by importance weight. TFMQ-DM uses Finite Set Calibration (the temporal embedding depends only on the finite set of t values, so calibrate over exactly that set).
- Practical rule: collect calibration activations by running the FP sampler and logging inputs at many timesteps (importance-weighted toward mid/low t), not random tensors.

### Methods

| Method | Bits | Quality (vs FP) | Models | Training-free | Kernel need |
|--------|------|-----------------|--------|---------------|-------------|
| PTQ4DM | W8A8 | CIFAR-10 FID 7.10 vs 7.14 (lossless); degrades hard below 8-bit | DDPM/DDIM U-Net | Yes | Plain INT8 GEMM, per-tensor |
| Q-Diffusion | W4 weights, A8 | <=+2.34 FID at 4-bit vs >100 for naive PTQ | LDM, Stable Diffusion U-Net | Yes | INT4/INT8 GEMM, per-channel wts, split-shortcut |
| PTQD (the "PTQ-D") | W8A8, W4A8, per-step mixed | +0.06 FID vs FP on LDM-4 ImageNet at ~19.9x BitOps cut | LDM-4, LSUN | Yes | INT8/INT4 GEMM + cheap elementwise correction + variance-schedule edit |
| ADP-DM | per-timestep group-wise | beats prior PTQ (exact numbers unverified) | LDM | Yes (data-free) | Group-wise INT8 GEMM |
| TFMQ-DM | W4A8, W8A8 | SD MS-COCO FID 13.15 -> 13.36 (W4A8) / 13.09 (W8A8) | LDM + SD U-Net | Yes | Standard INT8/INT4 GEMM; novelty is which blocks get reconstructed |
| ViDiT-Q | W8A8, W4A8 + MP | PixArt-alpha COCO FID 73.34 -> 75.61 (W8A8) / 74.33 (W4A8), near-lossless | PixArt, OpenSora, Latte (DiT) | Yes | Per-token (per-row) activation scales in epilogue + smoothing + mixed precision; CUTLASS kernels (validated RTX 4080) |
| Q-DiT | W6A8, W4A8 | DiT-XL/2 ImageNet W6A8 reduces FID by 1.09 vs baseline | DiT-XL/2 | Yes | Group-wise weight scales + per-sample dynamic activation scales |

Notes:
- PTQ4DM (arXiv 2211.15736, CVPR 2023) is the simplest port and a good first milestone (one INT8xINT8 -> INT32 GEMM + FP32 dequant epilogue).
- Q-Diffusion (arXiv 2302.04304, ICCV 2023) adds timestep-aware calibration + split shortcut quantization.
- PTQD (arXiv 2305.10657, NeurIPS 2023) is the request's "PTQ-D": decompose quant noise into a correlated part (channel-wise correction) and an uncorrelated Gaussian part (subtract mean bias, then recalibrate the denoising variance schedule), plus step-aware mixed precision. The corrections are elementwise math outside the GEMM, so it is mostly a sampler-loop change. Very attractive for a custom engine.
- ViDiT-Q (arXiv 2406.02540, ICLR 2025) is the most relevant for DiT/video: per-token dynamic activation quant + channel-balancing smoothing + metric-decoupled mixed precision. Reports 2.0x-2.67x memory, 1.4x-1.7x latency, 4x-7.11x BitOps. W4A8-MP keeps FFN (~15% of layers) at W8A8 and the first quarter of timesteps at W8A8.
- Q-DiT (arXiv 2406.17343) targets the per-input-channel variance specific to DiTs with automatic granularity allocation.
- Mixed-precision per-layer is the single biggest lever (used by PTQD, ViDiT-Q-MP, Q-DiT, MixDQ, MPQ-DM). Allocation is two-dimensional in diffusion: per-layer AND per-timestep. Note the disagreement on which timesteps need more bits (PTQD raises late high-SNR steps; ViDiT-Q raises the first quarter), so the right policy is model-dependent and must be calibrated. Engine implication: the scheduler must support per-layer + per-step bit selection (keep both INT8 and INT4 weight copies or fast-repack, dispatch the matching GEMM per layer per step).

### Engine takeaways for PTQ

- Start with PTQ4DM-style W8A8 (per-tensor, NDTC calibration), validate FID within ~2 of an FP reference.
- For W4A8 you need INT4 storage + dequant-to-INT8/FP16 or a mixed-input W4A8 GEMM, with configurable per-channel and per-group scale granularity.
- For DiTs, the winning pattern is ViDiT-Q: per-row activation scales in the epilogue + online min/max scale (cheap PTX reduction per token) + smoothing + mixed precision.
- Always add PTQD per-step correction (bias + variance-schedule recalibration); it is cheap elementwise math in the sampler loop and combats the accumulation problem that dominates at low bits.

---

## Family 2: Weight-only quant formats (GGUF / NF4) used in the ecosystem

All weight-only, all training-free PTQ. The runtime everyone ships is: unpack a quantized block to FP16 -> normal cuBLAS GEMM. The win is VRAM footprint, not throughput (at batch=1 diffusion GEMMs are partly compute-bound and the extra dequant work often makes GGUF the same speed or slightly slower than FP16). cuBLASLt epilogues operate on the GEMM output (bias/GELU/scale), not on decompressing an input operand, so you cannot dequantize "in the epilogue"; you either run a separate dequant kernel into FP16 scratch then GEMM (simplest, what everyone does) or write a fused dequant-in-prologue W4A16 kernel (Marlin-style, far more work, the only path that actually speeds up).

### GGUF K-quant block layout (port these byte-for-byte from llama.cpp)

K-quants use a super-block of QK_K = 256 weights, subdivided into small blocks, with a two-level scale: each small block has its own quantized scale (and for type-1, a min), and the super-block carries an FP16 d (and for type-1, an FP16 dmin).

- type-1 (Q2_K, Q4_K, Q5_K): blocks of 32, asymmetric, store scale AND min. Dequant: `w = d*scale*q - dmin*min`.
- type-0 (Q3_K, Q6_K): blocks of 16, symmetric, store scale only, fixed bias. Dequant: `w = d*scale*(q - bias)`.

| Format | Bytes / 256-w superblock | Bits/weight | Dequant |
|--------|--------------------------|-------------|---------|
| Q8_0 (legacy, per-32) | 34 B / 32 w | 8.5 | `w = d*q` |
| Q4_0 (legacy, per-32) | 18 B / 32 w | 4.5 | `w = d*(q-8)` |
| Q2_K | 84 | 2.625 | `w = d*sc*q - dmin*m` (4-bit sc/min) |
| Q3_K | 110 | 3.4375 | `w = d*sc*(q-4)` (6-bit sc) |
| Q4_K | 144 | 4.5 | `w = d*sc*q - dmin*m` (6-bit sc/min) |
| Q5_K | 176 | 5.5 | as Q4_K, q = low4(qs)+bit(qh) |
| Q6_K | 210 | 6.5625 | `w = d*sc*(q-32)`, q = low4(ql)+2bit(qh) |

The fiddly part: Q4_K/Q5_K pack 8 sub-block scales + 8 mins (16 x 6 bits = 12 bytes) in an interleaved layout. You must port llama.cpp's `get_scale_min_k4` exactly. Q6_K's scales are plain int8 (simpler).

The `_S`/`_M`/`_L` suffixes are NOT block formats; they are quantize recipes that assign different block formats per tensor (e.g. Q4_K_M bumps attention wv/wo and FFN down to Q6_K, so its effective bpw is ~4.8, not 4.5). The loader must read per-tensor `ggml_type` from GGUF metadata and dispatch the matching unpack kernel; you cannot assume one format per file.

Effective bpw + KL-div vs FP16 (LLM reference, Mistral-7B): Q2_K 3.00 bpw / 0.0588, Q3_K_M 3.89 / 0.0171, Q4_K_S 4.57 / 0.0083, Q4_K_M 4.83 / 0.0075, Q5_K_M 5.67 / 0.0043, Q6_K 6.57 / 0.0032. Flux community verdict: Q8_0 indistinguishable from FP16; Q6_K / Q5_K_S the best quality/size sweet spot; Q4_K_S minimal degradation; Q3_K_S "slightly rough"; Q2_K "awful." (The Flux quality ranking is community consensus, not a measured FID study.)

Memory vs FP16: Q8_0 ~2x, Q6_K ~2.4x, Q5_K_M ~2.8x, Q4_K_M ~3.3x, NF4 ~3.6-3.9x. Flux-dev ~24 GB FP16 drops to ~12 GB (Q8_0) or ~6.8 GB (Q4_K_M), the difference between fitting a 3060 and not.

### NF4 (4-bit NormalFloat, bitsandbytes / QLoRA, arXiv 2305.14314)

Non-uniform 4-bit using a quantile (NormalFloat) LUT optimized for near-Gaussian weights. The 16 LUT values (hardcode these): -1.0, -0.6961928, -0.5250731, -0.3949175, -0.2844414, -0.1847734, -0.0910500, 0.0, 0.0795803, 0.1609302, 0.2461123, 0.3379152, 0.4407098, 0.5626170, 0.7229568, 1.0. Block size 64, one FP16/BF16 absmax scale per block. Dequant: `w = nf4_lut[code] * absmax_block`. Double Quantization quantizes the per-64 absmax scales themselves in blocks of 256 to 8-bit with a second FP32 scale, saving ~0.37 bits/param (effective ~4.127 bpw). Runtime: bitsandbytes Linear4bit dequantizes to compute dtype then standard GEMM. Flux NF4 quality ~ Q4_K_M / Q5 territory, visibly better than naive uniform INT4.

### Per-tensor vs per-channel vs per-block scales

- Per-tensor: 1 scale/tensor, worst (outliers wreck range), ~0 overhead.
- Per-channel (per output row): good for linear, standard for W8A8, tiny overhead.
- Per-block/group (GGUF, NF4): best at low bits, but the scale memory IS the bpw overhead (a per-32 FP16 scale alone is 0.5 bpw, which is exactly why K-quants quantize scales to 6-bit and NF4 uses Double Quantization). Per-block group=32/64 is the right default for a from-scratch low-bit engine.

### What it takes to implement (two-stage path)

1. GGUF reader (C#): parse header/metadata, read per-tensor ggml_type, mmap tensor data.
2. One PTX unpack kernel per format. Minimum viable for diffusion: Q8_0, Q4_K, Q5_K, Q6_K (covers ComfyUI-GGUF Flux/SD3.5) + NF4 (bitsandbytes Flux). Each kernel: one CUDA block/warp per super-block, reconstruct 6-bit scales (get_scale_min_k4 for Q4_K/Q5_K), apply the dequant formula, write FP16 into a reused scratch buffer. Reference: llama.cpp `ggml-cuda/dequantize.cuh`, `dequantize_block_q4_K`, `dequantize_block_q6_K`.
3. cuBLASLt FP16 GEMM on the dequantized weight. Bias/activation in the LtMatmul epilogue.
4. Memory strategy: dequantize per-layer into a reused scratch buffer; never dequantize the whole model up front or you pay FP16 memory and lose the entire point.

Speed reality check: stage-1 path is same-or-slightly-slower than FP16 at batch=1; the value is fitting the model. A real speedup needs the fused W4A16 prologue kernel (separate, larger project, pays off only when weight-bandwidth-bound).

---

## Family 3: FP8 specifics

### Formats (FP8 Formats for Deep Learning, arXiv 2209.05433)

| Format | Exp | Mantissa | Max normal | Special values |
|--------|-----|----------|------------|----------------|
| E4M3 | 4 | 3 | 448 | no inf; single NaN pattern (range-extended) |
| E5M2 | 5 | 2 | 57344 | IEEE-style inf + NaN |

E4M3 (the `e4m3fn` finite variant) is the inference format for weights and activations (precision over range). E5M2 is for gradients (training); essentially irrelevant for forward-only inference.

### cuBLASLt FP8 GEMM

- Compute type CUBLAS_COMPUTE_32F (FP32 accumulate). Sample: A,B = CUDA_R_8F_E4M3; C = CUDA_R_16BF; D = E4M3 or a higher-precision type.
- Scaling equation: `D = scale_D * Epilogue(alpha*scale_A*scale_B*op(A)op(B) + beta*scale_C*C)`, with amax_D written out for the next iteration.
- Descriptor scale pointers (device FP32 scalars, per-tensor): A_SCALE_POINTER, B_SCALE_POINTER, C_SCALE_POINTER, D_SCALE_POINTER, AMAX_D_POINTER. Folding scale_A*scale_B into alpha is mathematically equivalent in per-tensor mode (engine's current approach is valid). The dedicated pointers become required only when you move to per-row/per-block scaling (those scales are vectors/tiles, not scalars).
- TN layout requirement (load-bearing): on Ada/Hopper, FP8 tensor-core matmul requires one input transposed (TN, K contiguous / K fastest-varying). The sample sets TRANSA = CUBLAS_OP_T. Blackwell (sm_100 data-center) lifts this. Plan layouts so K is contiguous; a non-TN FP8 call fails or falls back.

### Finer-grained FP8 scaling (which CUDA versions / hardware)

- cuBLAS 12.9 (CUDA 12.9, ~May 2025) added on Hopper (sm_90): per-row/per-column (outer-vector) scaling, and 1x128 + 128x128 block scaling (docs 3.1.4.5). DeepSeek-style recipe: activations 1x128, weights 128x128, all E4M3.
- Blackwell (sm_100): native MXFP8 micro-scaling at 1x32 with E8M0 (CUDA_R_UE8M0) scale.
- On Ada / RTX 4090, cuBLASLt FP8 is effectively per-tensor only (matches the engine's fold-into-alpha design).

### ComfyUI fp8_scaled vs naive cast

- Naive fp8_e4m3fn: tensor-wise cast, no stored scale; values outside +/-448 saturate. Lowest quality.
- fp8_scaled: stores the FP8 base tensor PLUS a per-tensor FP16 scale_weight (chosen so weight amax maps near 448); ComfyUI patches Linear to do `fp8_value * scale`. Closes most of the gap to bf16. To match ComfyUI quality, consume the scale_weight tensor as the per-tensor B-scale (or fold into alpha). A naive cast without the stored scale visibly underperforms.
- Block-wise fp8 scaled (newer, ComfyUI issue #10491) is finer granularity for higher quality.
- Community quality order: fp16/full >= bf16 >= fp8_scaled >= fp8_e4m3fn. Community FID-equivalent ranking for Flux: GGUF Q8 closest to BF16, fp8_scaled below Q8 but well above naive fp8.

### Dynamic vs static activation scaling (Transformer Engine)

- Delayed scaling (static-ish): scale from an amax history (default len 1024). Smooths outliers, adds staleness/sync.
- Current/JIT scaling (dynamic): scale from the current tensor's amax; one extra amax reduction per tensor.
- For inference: weights use a static per-tensor scale at load (exactly what fp8_scaled bakes in, zero runtime cost). Activations either static (offline amax calibration on representative prompts) or dynamic (runtime amax per step). Dynamic is simpler and robust for diffusion's per-timestep variation, at the cost of an amax reduction kernel per tensor.

### Accuracy: Ampere cast-path vs native

- Native FP8 (Ada/Hopper): A,B in E4M3, multiplied in the tensor core, FP32 accumulate. Error only from FP8 quantization of inputs.
- Ampere cast (RTX 3060): no FP8 tensor cores; store FP8, upcast to F16, run an ordinary F16 GEMM. Accuracy equals "FP8-quantized weights through an F16 GEMM" and is generally at least as accurate as native FP8 (F16 multiply is more precise than FP8xFP8). You save only VRAM and load bandwidth, NOT FLOPs (you expand to F16 in SMEM/registers). The 3060 win is the ~2x model-size reduction to fit larger models, not throughput.

### FP8 hardware matrix

| GPU | Arch | Native FP8 TC | Notes |
|-----|------|---------------|-------|
| RTX 3060 / A100 | Ampere | No | INT8 is the lowest HW format; FP8 must cast to F16 |
| RTX 4090 / L40S | Ada | Yes (compute) | per-tensor scale applied in software (fold into alpha) |
| H100 / H200 | Hopper | Yes (full + TE) | adds per-row/per-block in cuBLAS 12.9 |
| B200 / RTX 5090 | Blackwell | Yes (5th-gen + MXFP8 1x32 E8M0) | TN restriction lifted |

Speedup: cuBLAS reports ~4.8x FP8 on H100 over BF16 on A100; same-gen FP8-over-BF16 on Hopper ~1.7x. Community diffusion: FP8 ~1.5-2x BF16 on native-FP8 GPUs, mostly at higher resolution/batch.

---

## Family 4: SVDQuant / NVFP4 / MXFP4 (4-bit)

### SVDQuant (arXiv 2411.05007, ICLR 2025 Spotlight; Nunchaku engine)

The algorithm. At 4-bit you must quantize both weights and activations (W4A4) to get speedup, but DiT activations have large outliers. Three stages:

1. Smoothing (outlier migration activation -> weight): per-channel factor s, `X_hat = X*diag(s)`, `W_hat = diag(1/s)*W`, product unchanged. Moves the hard outliers into the static weights (SmoothQuant trick).
2. Low-rank branch absorbs the weight outliers (the novel part): SVD-split `W_hat ~= L1 L2 + R`, with L1 (in, rank), L2 (rank, out) kept in 16-bit. The top singular components carry the outlier energy (Eckart-Young-Mirsky), so the residual R is smooth and quantizes cleanly to 4-bit INT4.
3. Inference: `Y = X_hat (dequant(qweight) + L1 L2)` = a 4-bit residual GEMM + a 16-bit rank-r GEMM.

Rank: typical fast config is 32 (Nunchaku exposes 32 fastest / 128 balanced / 256 best, INT4 only).

Kernel fusion (load-bearing): a standalone rank-32 GEMM is memory-bound and would re-read the activation, doubling traffic. SVDQuant fuses the down-projection (L1) into the quantization kernel and the up-projection (L2) into the 4-bit compute kernel, sharing activations in SMEM/registers. The low-rank branch then adds only ~5-10% latency (DeepWiki: ~57% overhead reduction from fusion). Budget for a fused INT4 GEMM that consumes the rank-32 16-bit factors inline; a separate cuBLASLt call erases the speedup.

Reported numbers (RTX 4090): 3.6x model-size reduction (12B FLUX.1), 3.5x memory vs 16-bit, 3.0x speedup vs NF4 W4A16, up to 8.7x-10.1x total latency on a 16GB laptop 4090 (these last bundle avoiding CPU offload; pure same-resident compute speedup is ~3x). 3.1x vs W4A16 on RTX 5090. Quality vs FP16 (4-bit rank-32 INT4): FLUX.1-dev LPIPS 0.223 / ImageReward 0.935 (beats NF4 0.272 / 0.910); PixArt-Sigma LPIPS 0.323 (vs ViDiT-Q INT4 0.854); SDXL FID on par with FP16. 4-bit SVDQuant matches or exceeds 8-bit methods.

### NVFP4 (NVIDIA, Blackwell only)

Two-level scaling: elements E2M1 (range ~+/-6, magnitudes {0,0.5,1,1.5,2,3,4,6}); per-block scale in FP8 E4M3 (true float, not power-of-two), block size 16; plus a per-tensor FP32 second-level scale. ~4.50 bits/element. Native FP4 matmul on Blackwell 5th-gen tensor cores only (B200/B300, RTX 50). NVIDIA LLM numbers (not diffusion): 3.5x memory vs FP16, <1% accuracy degradation vs FP8, 2x FP8 throughput.

### MXFP4 / MX formats (OCP MX v1.0; arXiv 2310.10537)

MX block: block size k=32, k element codewords, one shared scale in E8M0 (8-bit, power-of-two only), NO per-tensor second-level scale. Defined formats: MXFP8 (E5M2/E4M3), MXFP6, MXFP4 (E2M1), MXINT8. MXFP4 ~4.25 bits/element. Coarser than NVFP4 (power-of-two vs E4M3 float scale), but open/cross-vendor (also AMD MI300X/MI355). Native matmul needs Blackwell (NVIDIA) or CDNA3+ (AMD).

### What needs Blackwell vs runs on Ampere

| Path | Native HW | RTX 3060 (Ampere)? |
|------|-----------|--------------------|
| SVDQuant W4A4 INT4 | INT4 tensor cores (Turing+) | Yes, custom integer GEMM kernels |
| NVFP4 native matmul | Blackwell 5th-gen (sm_120) | No native; dequant-to-FP16 only |
| MXFP4 native matmul | Blackwell / CDNA3+ | No native; dequant-to-FP16 only |

Crucial: Nunchaku ships two data paths and auto-selects via compute capability. get_precision() returns "int4" for cc 7.5-8.9 (Turing -> Ada, integer kernels gemm_w4a4 / gemv_awq) and "fp4" for cc 10.x/12.x (Blackwell native FP4). SVDQuant's published 3x/8.7x 4090 numbers are the INT4 path. So implement the W4A4 INT4 SVDQuant front-end (smoothing + rank-32 16-bit low-rank + fused INT4 residual GEMM) now for the 3060 and cloud Ada/Hopper; add an NVFP4 residual-GEMM variant later behind an sm_120 check. They share the front-end; only the residual datatype/kernel differs. NVFP4/MXFP4 can still be stored on Ampere and dequantized to FP16 (memory win only, no compute speedup) which is the regime SVDQuant's INT4 path was built to beat.

---

## Family 5: INT8 / INT4 GEMM paths and low-precision attention

### cuBLASLt INT8 GEMM (IMMA / int8 tensor cores) -- the key Ampere path

Ampere SM 8.6 has native INT8 tensor cores (IMMA); int8 is NOT emulated. fp8 tensor-core MMA does not exist before Ada SM 8.9, so on a 3060 int8 is the only true low-precision tensor-core GEMM.

- Inputs A,B = CUDA_R_8I; accumulate/output C = CUDA_R_32I (int32); compute type CUBLAS_COMPUTE_32I. int32 accumulate prevents overflow summing many int8xint8.
- IMMA TN layout (the gotcha): the tensor-core path needs special interleaved orderings, not row/col-major. A: OP_N, order CUBLASLT_ORDER_COL32. B: OP_T, order COL4_4R2_8C (Turing/Ampere) or COL32_2R_4R4 (Ampere-tuned). C: OP_N, COL32. These are opaque; convert into/out of them with cublasLtMatrixTransform. Pre-convert weights to the target layout ONCE at load; only transform activations per step. The legacy non-tensor int8 path (cublasGemmEx) takes plain layouts but runs on int32 CUDA cores (much slower). Reference sample: cuBLASLt/LtIgemmTensor.
- Known issue: some CUDA versions report "not supported" for int8 + COMPUTE_32I on Hopper; probe with cublasLtMatmulAlgoGetHeuristic at init and keep a fallback.
- Numbers: RTX 3060 (GA106) ~101.9 INT8 TOPS dense (~203.8 with 2:4 sparsity). Architecturally int8 ~2x fp16, int4 ~4x. Real measured (RTX 3090 Ti, same SM 8.6, third-party): cuBLASLt int8 ~118 TOPS vs ~75 for a naive fp16 matmul. The bigger real-world win is usually 2x weight memory + bandwidth, not peak TOPS.

### SmoothQuant (W8A8, arXiv 2211.10438)

Activations have systematic large-magnitude outliers in a few fixed channels; weights are smooth. SmoothQuant migrates difficulty offline via a per-channel factor s: divide activation channel j by s_j, multiply the weight row by s_j (product unchanged), so both become int8-friendly. Migration strength alpha ~0.5. This is the technique that pairs with the cuBLASLt int8 path: you need BOTH operands int8 to use IMMA. AWQ/GPTQ are weight-only and do NOT feed the int8 tensor cores on their own.

### AWQ (W4A16, arXiv 2306.00978)

Protect ~1% of salient weight channels (selected by activation magnitude, not weight magnitude) by scaling them up (equivalent transform), so the whole tensor goes to 3/4-bit uniformly. Weight-only (activations FP16). Reorder-free tensor-core kernels with online dequant: ~1.45x over GPTQ, ~1.85x over cuBLAS FP16. W4A16 is memory-bandwidth-bound dequant, not an IMMA path; on a 3060 the win is VRAM + weight-load bandwidth.

### GPTQ (W3-4A16, arXiv 2210.17323)

One-shot layer-wise PTQ using approximate second-order (Hessian) info; quantizes column-by-column, updating remaining weights to compensate. 3-4 bits with negligible degradation; quantized 175B in ~4 GPU-hours. Weight-only (memory/bandwidth play).

### Applicability to diffusion vs LLM

LLM single-calibration PTQ is invalid for diffusion (per-timestep distributions + bimodal shortcuts). What has been ported: PTQ4DM (first int8 diffusion U-Net, traced the drop to per-timestep distribution discrepancy), Q-Diffusion (timestep-aware calibration + split-shortcut), ViDiT-Q (SmoothQuant-style smoothing for DiTs; W8A8 matches FP16, and only smoothing prevented W4A8 collapse; 2-2.5x memory, 1.4-1.7x latency, custom kernels). Takeaway: the SmoothQuant family (specifically ViDiT-Q) is the directly applicable W8A8 route on the int8 tensor cores. AWQ/GPTQ ideas (salient-channel protection, Hessian-aware) transfer conceptually to weight-only diffusion compression but you must add timestep-aware calibration.

### Low-precision attention

cuBLASLt cannot do attention (no softmax, no online-softmax rescaling, no fused QK->softmax->PV). You must write a custom flash-attention-style fused PTX kernel regardless; cuBLASLt can at most do the QK^T/PV GEMM tiles in an unfused (memory-heavy) version, which you should not at long sequence length.

- FP8 attention (FlashAttention-3, arXiv 2407.08608): Hopper-only fp8 path (warp specialization, TMA, async). FP16 FA-3 ~740 TFLOPS on H100 (~75% util, 1.5-2x over FA-2); FP8 ~1.2 PFLOPS/s, 2.6x lower numerical error via block quantization + incoherent processing. Falls back to FA-2 on Ampere/Ada. Hopper-tier reference, not an Ampere option.
- INT8 attention (SageAttention v1, arXiv 2410.02367, ICLR 2025): quantize Q,K to INT8 for QK^T (smooth K by subtracting its channel-wise mean to absorb outliers), keep PV in FP16 with FP16 accumulator. ~2.1x over FA-2, ~2.7x over xformers; "almost no end-to-end metric loss." Ampere supported (v1 Triton targets Ampere; benchmarked on RTX 3090, same SM 8.6 as the 3060). This is the realistic 3060 low-precision attention to implement in PTX.
- SageAttention2 (arXiv 2411.10958, ICML 2025): per-thread INT4 Q/K + FP8 PV (FP32 two-level accumulation), smooth Q and V. ~3x over FA-2 (RTX 4090). The INT4/FP8 path leans on Ada+; on Ampere use the INT8 (v1 or SageAttn2-8b) path. SageAttention3 (FP4) needs Blackwell + CUDA >= 12.8.

---

## Family 6: Quantized KV / attention for long-sequence video backbones

Framing correction: video diffusion backbones (HunyuanVideo, Wan, LTX, CogVideoX) do bidirectional full self-attention over the whole sequence every denoising step. There is no growing autoregressive KV cache. So "quantized KV" here means quantizing the Q/K/V tensors and the QK^T / PV matmuls (recomputed fresh each step), i.e. the SageAttention/FlashAttention-quantization family. LLM KV-cache quant (KVQuant, KIVI) targets a storage problem you mostly do not have.

Sequence lengths are enormous: HunyuanVideo at 768x1280x129f is ~122.9k tokens, so the QK^T score matrix is ~1.5e10 entries per head and is never materialized. Flash-style streaming (O(N) memory, tiled, running softmax) is mandatory; quantization then cuts the bytes moved for Q/K/V tiles AND the tensor-core matmul cost. The two compound.

### SageAttention on video (the main lever)

- v1 INT8 (arXiv 2410.02367): on CogVideoX the adaptive strategy sped attention by 11.7% with no metric loss. The safe, highest-compatibility recipe for Ampere.
- SageAttention2 (arXiv 2411.10958) verified end-to-end video (SageAttn2-4b): CogVideoX 1.5-5B (RTX 4090) 1.8x end-to-end, but VQA-t dropped 70.928 -> 52.989 (a real quality hit at 4-bit on this model). HunyuanVideo (L20) 1.55x, near-lossless (VQA-a 81.478 vs 82.516). Mochi (L20) 1.96x, but VQA-a 35.955 vs 45.549 (meaningful drop). So "near-lossless" is model-dependent: HunyuanVideo tolerates INT4 well; CogVideoX/Mochi need INT8 or the 8-bit variant.
- API names worth mirroring: sageattn_qk_int8_pv_fp16_cuda (the safe Ampere path), sageattn_qk_int8_pv_fp8_cuda (Ada+).

### FP8 + sparsity co-design (video)

- FPSAttention (arXiv 2506.04648): training-aware FP8 + sparsity with 3D tile-wise granularity and a denoising-step-aware schedule. Up to 4.96x "without quality loss"; 2.8-17x over FA-2, 1.6-10x over FA-3. Needs fine-tuning (not plug-and-play).
- Sparse attention (orthogonal, compounds): Sparse VideoGen (arXiv 2502.01776, 2.28-2.33x), Sparse VideoGen2 (arXiv 2505.18875), Radial Attention (arXiv 2506.19852, O(n log n)), PAROAttention (arXiv 2506.16054, makes attention both sparse and quantization-friendly).

### LLM KV-quant transfer (limited)

- KVQuant (arXiv 2401.18079): per-channel Key, pre-RoPE Key quant, dense-and-sparse outliers; <0.1 ppl at 3-bit; 1M context on one A100. KIVI (arXiv 2402.02750): asymmetric 2-bit, Key per-channel + Value per-token (Key has persistent channel outliers); 2.6x memory, 2.35-3.47x throughput.
- Transfer assessment (analysis): the per-channel-Key / pre-RoPE insight transfers (it is the same root finding SageAttention exploits by smoothing K/Q channel-wise; if the DiT uses RoPE like Wan/HunyuanVideo, prefer quantizing K before RoPE or smoothing channels). The 2-bit/1-bit storage ambitions do NOT transfer: video attention recomputes Q/K/V each step (no long-lived cache) and bidirectional attention is more sensitive to score error than causal decoding; sub-INT4 on video attention is unproven and shows quality cliffs.

### Kernel-support per arch

- RTX 3060 (Ampere): INT8 QK^T via mma.sync s8 (IMMA) + FP16 PV with FP16/FP32 accumulate (SageAttention v1 / qk_int8_pv_fp16). No native FP8.
- Ada: INT8 or FP8(E4M3) QK + FP8 PV (SageAttention2 qk_int8_pv_fp8).
- Hopper: async TMA + warp specialization (FA-3-style FP8).
- Blackwell (future cloud): NVFP4 microscaling (SageAttention3), CUDA >= 12.8.

---

## Ranked summary table (implementability on this engine)

Rank = ease/payoff on Ampere first, then Ada/Hopper/Blackwell. "Speedup" is over FP16 unless noted.

| # | Technique | Train-free | Weight-loadable | Kernel need | Memory | Speedup | Quality cost | Ampere (3060) | Ada/Hopper |
|---|-----------|------------|-----------------|-------------|--------|---------|--------------|---------------|------------|
| 1 | GGUF weight-only (Q8_0/Q6_K/Q5_K/Q4_K) | Yes | Yes (city96 checkpoints) | Custom PTX unpack -> FP16 -> cuBLAS GEMM (no fused matmul) | 2-3.3x | ~1x (memory win only) | Q8 ~none; Q4_K minimal on DiT | Easy, native int unpack | Same |
| 2 | NF4 weight-only (bitsandbytes Flux) | Yes | Yes | LUT unpack PTX -> FP16 -> GEMM | ~3.6-3.9x | ~1x | ~Q4_K/Q5 | Easy | Same |
| 3 | FP8 weight storage, F16 compute (Ampere cast) | Yes | Yes (fp8_scaled) | Already built; consume scale_weight | ~2x | ~1x on 3060 | fp8_scaled ~bf16 | Done | n/a |
| 4 | FP8 native GEMM (per-tensor) | Yes | Yes (fp8_scaled) | cuBLASLt FP8, TN layout (already built) | ~2x | ~1.5-1.7x | fp8_scaled close to bf16 | No native | Ada/Hopper now |
| 5 | INT8 attention (SageAttention v1) | Yes | n/a (runtime) | Custom flash PTX: INT8 QK^T (mma s8) + FP16 PV, smooth K | attn-mem down | ~2.1x attn (1.1-1.5x e2e) | near-lossless (v1) | Yes (SM 8.6) | Yes |
| 6 | W8A8 PTQ (PTQ4DM / ViDiT-Q / SmoothQuant) | Yes | export scales | cuBLASLt INT8 IMMA (COL32 TN) + per-row act scale epilogue + calibration | 2-2.5x | ~1.4-1.7x e2e | near-FP at W8A8 | Yes (native int8 TC) | Yes |
| 7 | PTQD per-step correction | Yes | export scales | elementwise + variance-schedule edit in sampler (no GEMM) | (rides W8A8/W4A8) | n/a | recovers FID at low bits | Yes | Yes |
| 8 | SVDQuant W4A4 INT4 | Yes | Yes (Nunchaku/DeepCompressor) | Fused INT4 residual GEMM + rank-32 FP16 low-rank inline | ~3.5x | ~3x | matches/beats 8-bit | Yes (int4 TC, big kernel effort) | Yes (int4 path) |
| 9 | W4A8 PTQ (ViDiT-Q-MP / Q-DiT / TFMQ) | Yes | export scales | INT4 weight unpack -> mixed-input GEMM, per-group scales, mixed precision | 2x+ | ~1.5x | near-FP with MP | Yes (more kernel work) | Yes |
| 10 | INT4/FP8 attention (SageAttention2) | Yes | n/a | INT4 Q/K + FP8 PV flash kernel | attn-mem down | ~3x attn | model-dependent (cliffs on some) | INT8 subset only | Yes (full) |
| 11 | FP8 attention (FlashAttention-3) | Yes | n/a | Hopper async/TMA/warp-spec fused kernel | attn-mem down | ~2x attn | low error (block-quant) | No (falls back to FA-2) | Hopper only |
| 12 | NVFP4 / MXFP4 native 4-bit GEMM | Yes | Yes (storage) | Blackwell FP4 tensor cores | ~3.5x | ~2x over FP8 | <1% (LLM) | No native (dequant only) | Blackwell only |

---

## Top recommendations for HartsyInference on RTX 3060 (Ampere) + cloud Ada/Hopper

Phase A (memory, low risk, do first):
1. GGUF weight-only path: PTX unpack kernels for Q8_0, Q4_K, Q5_K, Q6_K (+ NF4 LUT) -> FP16 scratch -> cuBLAS FP16 GEMM, dequantizing per-layer into a reused buffer. Loads city96 Flux/SD3.5 GGUF and bitsandbytes NF4 directly. This is the "fit Flux on a 12GB 3060" win. Pure memory benefit, no FID risk at Q8/Q6/Q5.
2. fp8_scaled support: consume the stored scale_weight; on the 3060 use the existing FP8->F16 cast path; on cloud Ada/Hopper use the existing native FP8 cuBLASLt GEMM (TN layout). Matches ComfyUI quality, ~2x model-size cut.

Phase B (real speedup, native Ampere int8):
3. SageAttention v1 INT8 attention as a custom fused PTX flash kernel (INT8 QK^T via mma.sync s8, FP16 PV, channel-mean smooth K). Native on SM 8.6, ~2x attention speedup, near-lossless. The highest-leverage compute win on the 3060 and essential for long-sequence video DiTs where attention dominates.
4. W8A8 PTQ via the cuBLASLt INT8 IMMA path (COL32/COL4_4R2_8C TN layout, weights pre-transformed at load) + per-row activation scales in the epilogue, calibrated ViDiT-Q/SmoothQuant style with timestep-aware (NDTC) calibration. Add PTQD per-step bias + variance-schedule correction (free, in the sampler loop). ~1.4-1.7x end-to-end, near-FP quality, 2-2.5x memory.

Phase C (aggressive 4-bit, larger kernel effort):
5. SVDQuant W4A4 INT4 (Ampere int4 path): smoothing + rank-32 16-bit low-rank branch FUSED into an INT4 residual GEMM (do not run the low-rank branch as a separate cuBLASLt call). Loads Nunchaku/DeepCompressor checkpoints. ~3.5x memory, ~3x speedup, matches 8-bit quality. The big-effort, big-payoff item.
6. W4A8 PTQ with per-layer + per-timestep mixed precision (ViDiT-Q-MP / Q-DiT / TFMQ-DM) as a fallback for models where W4A4 over-degrades.

Defer to cloud / future hardware:
- FlashAttention-3 FP8 attention: Hopper-only, falls back to FA-2 on Ampere.
- SageAttention2 INT4/FP8 full path: Ada+ for the FP8 PV; on Ampere only the INT8 subset.
- NVFP4 / MXFP4 native matmul: Blackwell only. Share the SVDQuant front-end and add an NVFP4 residual-GEMM variant behind an sm_120 capability check. On Ampere/Ada these are storage-only (dequant to FP16, memory win, no compute speedup).

Cross-cutting design notes:
- cuBLASLt epilogues cannot dequantize input operands; weight-only formats need a separate unpack kernel (or a fused W4A16 prologue for actual speed).
- Make GEMM scale granularity configurable from day one (per-tensor, per-channel/row, per-group), because everything past W8A8-per-tensor needs it.
- Attention always needs a custom fused kernel; cuBLASLt has no softmax.
- All PTQ here needs a sampler-driven, importance-weighted-timestep calibration pass (~1k samples); no method gives a drop-in quantized checkpoint with calibrated activation scales.

---

## Caveats / unverified claims

- Flux GGUF quality ranking (Q8 > Q6_K > ... > Q2_K) is community consensus, not a measured FID/CLIP study. KL-div/perplexity numbers in the bpw table are LLM (Mistral/Llama) references.
- FP8 "fp16 >= bf16 >= fp8_scaled >= fp8_e4m3fn" ordering is community-reported (ComfyUI issues/blogs), not a controlled paper benchmark. cuBLAS 12.9 date (~May 2025) is approximate.
- SVDQuant rank=32 is from Nunchaku implementation docs, not pinned in the arXiv abstract. The 8.7x/10.1x speedups bundle avoiding CPU offload on a 16GB laptop 4090; same-resident compute speedup is ~3x. NVFP4 throughput/accuracy numbers are NVIDIA LLM benchmarks, not diffusion.
- INT8 "118 vs 75 TOPS" is a single third-party RTX 3090 Ti benchmark; the "int8 = 2x fp16, int4 = 4x" ratios are architectural peaks, not delivered cuBLASLt numbers. RTX 3060 is not separately benchmarked in SageAttention tables (inferred from shared SM 8.6 with the tested RTX 3090).
- SageAttention2 "near-lossless" is model-dependent (CogVideoX VQA-t and Mochi VQA-a show real drops under INT4; HunyuanVideo is the success case). SageAttention2 per-model speedups were measured on different GPUs (RTX 4090 / L20), not directly comparable, none on a 3060.
- ADP-DM (arXiv 2305.18723) exact W#A#/FID not verified (abstract only); mechanism confirmed. PTQD per-config FID confirmed via TFMQ-DM's comparison table, not the PTQD PDF directly.
- KVQuant -> video-attention transfer (Family 6) is analysis, not a measured result; LLM KV-cache machinery is largely irrelevant to the bidirectional, cache-free video hot path.

## Sources

PTQ for diffusion:
- Q-Diffusion: https://arxiv.org/abs/2302.04304 , https://github.com/Xiuyu-Li/q-diffusion
- PTQ4DM: https://arxiv.org/abs/2211.15736 , https://github.com/42Shawn/PTQ4DM
- PTQD: https://arxiv.org/abs/2305.10657 , https://github.com/ziplab/PTQD
- ADP-DM: https://arxiv.org/abs/2305.18723
- TFMQ-DM: https://arxiv.org/abs/2311.16503 , https://github.com/ModelTC/TFMQ-DM
- ViDiT-Q: https://arxiv.org/abs/2406.02540 , https://github.com/thu-nics/ViDiT-Q
- Q-DiT: https://arxiv.org/abs/2406.17343 , https://github.com/Juanerx/Q-DiT

Weight-only formats:
- ComfyUI-GGUF: https://github.com/city96/ComfyUI-GGUF , dequant.py https://github.com/city96/ComfyUI-GGUF/blob/main/dequant.py
- llama.cpp k-quants PR #1684: https://github.com/ggml-org/llama.cpp/pull/1684 ; ggml-common.h block structs: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h
- bpw + KL-div (Artefact2): https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9
- QLoRA / NF4: https://arxiv.org/abs/2305.14314 ; bitsandbytes Linear4bit: https://huggingface.co/docs/bitsandbytes/en/reference/nn/linear4bit
- city96 FLUX K-quant comparison: https://huggingface.co/city96/FLUX.1-dev-gguf/discussions/15

FP8:
- FP8 Formats for Deep Learning: https://arxiv.org/abs/2209.05433
- NVIDIA Transformer Engine FP8 primer: https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html
- cuBLAS docs: https://docs.nvidia.com/cuda/cublas/
- cuBLASLt LtFp8Matmul sample: https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuBLASLt/LtFp8Matmul/sample_cublasLt_LtFp8Matmul.cu
- cuBLAS 12.0 features: https://developer.nvidia.com/blog/new-cublas-12-0-features-and-matrix-multiplication-performance-on-nvidia-hopper-gpus/
- cuBLAS 12.9 (per-row, 1x128/128x128): https://developer.nvidia.com/blog/boosting-matrix-multiplication-speed-and-flexibility-with-nvidia-cublas-12-9/
- ComfyUI fp8 block scaling issue #10491: https://github.com/Comfy-Org/ComfyUI/issues/10491
- ComfyUI SD3.5 (t5xxl_fp8_e4m3fn_scaled): https://blog.comfy.org/p/sd3-5-comfyui

SVDQuant / NVFP4 / MXFP4:
- SVDQuant: https://arxiv.org/abs/2411.05007 ; MIT blog https://hanlab.mit.edu/blog/svdquant ; Nunchaku https://github.com/nunchaku-ai/nunchaku
- Nunchaku hardware/precision (DeepWiki): https://deepwiki.com/mit-han-lab/nunchaku/1.2-hardware-compatibility-and-precision-selection
- NVFP4: https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference
- OCP MX v1.0 spec: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
- Microscaling Data Formats: https://arxiv.org/abs/2310.10537

INT8/INT4 GEMM + attention:
- cuBLASLt int8 layouts (corsix): https://www.corsix.org/content/cublaslt-notes
- cuBLASLt LtIgemmTensor sample: https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuBLASLt/LtIgemmTensor/sample_cublasLt_LtIgemmTensor.cu
- Ampere SM86 int8 / no native fp8: https://amohan.dev/blog/2026/fp8-as-storage-imma-ampere/
- SmoothQuant: https://arxiv.org/abs/2211.10438 ; AWQ: https://arxiv.org/abs/2306.00978 ; GPTQ: https://arxiv.org/abs/2210.17323
- FlashAttention-3: https://arxiv.org/abs/2407.08608
- SageAttention: https://arxiv.org/abs/2410.02367 ; SageAttention2: https://arxiv.org/abs/2411.10958 ; SageAttention3: https://arxiv.org/abs/2505.11594 ; repo https://github.com/thu-ml/SageAttention

Video / long-sequence attention:
- FPSAttention: https://arxiv.org/abs/2506.04648
- Sparse VideoGen: https://arxiv.org/abs/2502.01776 ; Sparse VideoGen2: https://arxiv.org/abs/2505.18875
- Radial Attention: https://arxiv.org/abs/2506.19852 ; PAROAttention: https://arxiv.org/abs/2506.16054
- KVQuant: https://arxiv.org/abs/2401.18079 ; KIVI: https://arxiv.org/abs/2402.02750
