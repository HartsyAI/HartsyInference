# GroupNorm Math — Research Notes

---

## Summary

GroupNorm (Wu & He, 2018) divides C channels into G groups and normalizes each group independently over spatial dimensions, making it batch-size-independent — the key property that made it the default normalization in diffusion UNets. Every ResNetBlock in SD 1.5, SDXL, and FLUX uses GroupNorm (G=32) followed by SiLU activation. Because GroupNorm reduces over potentially thousands of elements per group, FP16 accumulation is numerically unsafe; PyTorch and NVIDIA Apex both accumulate mean/variance in FP32 even when inputs are FP16. Fusing GroupNorm with SiLU into a single kernel eliminates one full memory round-trip and yields ~50% kernel speedup, with ~0.5-1% end-to-end improvement on SD 1.5 (and up to ~35% end-to-end when combined with NHWC convolution).

---

## Detailed Findings

### 1. What GroupNorm Is

GroupNorm belongs to a family of normalization techniques that all follow the same general formula but differ in **which set of elements** they normalize over. The general normalization formula from Wu & He (2018) is:

```
y_i = (x_i - mu_i) / sqrt(sigma_i^2 + eps)  *  gamma  +  beta
```

where `i = (iN, iC, iH, iW)` indexes a specific element in a 4D feature map (batch, channel, height, width), and `mu_i` and `sigma_i^2` are the mean and variance computed over a specific set `S_i` of pixel indices.

The mean and variance are computed as:

```
mu_i    = (1 / |S_i|) * SUM_{k in S_i} x_k
sigma_i^2 = (1 / |S_i|) * SUM_{k in S_i} (x_k - mu_i)^2
```

The learnable parameters `gamma` (scale) and `beta` (shift) are per-channel vectors of length C, applied element-wise after normalization. When `affine=False`, gamma=1 and beta=0 (identity).

### 2. How the Normalization Variants Differ (Set S_i)

All four normalization methods use the same formula above but differ in the definition of `S_i`:

| Method | Set S_i (what is normalized together) | Depends on batch? |
|--------|---------------------------------------|-------------------|
| **BatchNorm** | All spatial positions across all samples in the batch for the same channel: `S_i = {k : k_C = i_C}` | Yes |
| **LayerNorm** | All channels and spatial positions for a single sample: `S_i = {k : k_N = i_N}` | No |
| **InstanceNorm** | All spatial positions for a single channel of a single sample: `S_i = {k : k_N = i_N, k_C = i_C}` | No |
| **GroupNorm** | All spatial positions for a group of channels of a single sample: `S_i = {k : k_N = i_N, floor(k_C / (C/G)) = floor(i_C / (C/G))}` | No |

Special cases:
- **GroupNorm with G=1** is equivalent to **LayerNorm** (one group = all channels).
- **GroupNorm with G=C** is equivalent to **InstanceNorm** (each channel is its own group).
- BatchNorm is the only variant that normalizes across the batch dimension, which makes it batch-size-dependent and problematic for small batches.

### 3. GroupNorm Channel Grouping — Detailed

Given input tensor of shape `(N, C, H, W)` and G groups:
- Each group contains `C_g = C / G` channels (C must be divisible by G).
- Conceptually reshape to `(N, G, C_g, H, W)`.
- Compute mean and variance over dimensions `(C_g, H, W)` for each `(n, g)` pair.
- The reduction size per group is `C_g * H * W`.

For SD 1.5 / SDXL with G=32 and typical channel counts:

| Channel count C | Channels per group C_g | Spatial H*W (64x64 latent) | Elements per reduction |
|-----------------|----------------------|---------------------------|----------------------|
| 320 | 10 | 4096 | 40,960 |
| 640 | 20 | 1024 | 20,480 |
| 1280 | 40 | 256 | 10,240 |

### 4. FP16 Accumulation Stability

**FP16 (float16) has only 10 bits of mantissa precision and a max value of 65,504.** Summing thousands of values in FP16 leads to:

- **Swamping**: When the running sum becomes large, small addends are rounded to zero. Summing 40,960 values of magnitude ~1.0 produces a sum of ~40,960, and adding a value of 0.001 to it loses all significance in FP16.
- **Overflow**: Accumulating squared values can exceed 65,504.
- **Catastrophic cancellation**: The naive variance formula `E[x^2] - (E[x])^2` subtracts two similar large numbers, losing most significant digits.

**PyTorch's solution**: The CUDA `group_norm_kernel.cu` uses `acc_type<T, true>` which maps `half` -> `float` for all accumulation. The Welford online algorithm is used (not the naive two-pass), which is numerically stable even in FP32 because it avoids the subtraction of large similar values.

**NVIDIA Apex's solution**: Similarly accumulates in FP32 and supports mixed dtype combinations: `(input=float16, params=float32)`, `(input=float16, params=float16)`, etc.

**Recommendation for SharpInference**: Always accumulate mean and variance in FP32, even when inputs are FP16 or BF16. The FP32 accumulation cost is negligible compared to the memory bandwidth cost of reading the input tensor — the reduction is compute-light and memory-bound.

### 5. FP32 Accumulation Performance Cost

GroupNorm is **memory-bandwidth-bound**, not compute-bound. The dominant cost is reading `N * C * H * W` elements from memory, not the arithmetic. Widening the accumulator from FP16 to FP32 adds:
- One `half->float` conversion per element (essentially free on modern hardware, 1 cycle).
- Two extra FP32 registers per thread for running mean/variance.
- One `float->half` conversion per output element.

In practice, FP32 accumulation adds **<1% overhead** to the kernel wall-clock time on both CPU (SIMD) and GPU. This is confirmed by PyTorch using FP32 accumulation unconditionally for all normalization layers.

### 6. GroupNorm + SiLU Fusion

In diffusion UNets, the pattern is always:
```
x = GroupNorm(x)
x = SiLU(x)          # SiLU(x) = x * sigmoid(x)
```

**Unfused**: Two separate kernels, each reading and writing the full tensor — 2 reads + 2 writes.

**Fused**: One kernel that normalizes and applies SiLU in-place — 1 read + 1 write.

This cuts memory traffic in half for the normalization+activation step.

**Performance numbers** (from channels-last-groupnorm project, RTX 3060):
- Fused GroupNorm+SiLU kernel: ~50% faster than separate GroupNorm kernel alone
- End-to-end SD 1.5 (512x512, batch 1): 0.5-1% speedup from fusion alone
- Combined with NHWC convolution layout: ~35% end-to-end speedup

**NVIDIA Apex** implements fused GroupNorm+SiLU in `apex.contrib.group_norm` with NHWC layout, supporting the `act="silu"` parameter.

**Fusion with preceding Conv2D output** is also possible (write Conv2D output, then read it back for GroupNorm becomes: apply GroupNorm+SiLU as Conv2D writes its output). This is a deeper fusion that requires the Conv2D kernel to call GroupNorm inline, which is complex but done in some production frameworks.

### 7. Welford's Online Algorithm

PyTorch uses Welford's algorithm (1962) for numerically stable single-pass mean and variance computation. The algorithm maintains running mean and sum-of-squared-deviations:

```
Initialize: mean = 0, M2 = 0, count = 0

For each value x:
    count += 1
    delta = x - mean
    mean += delta / count
    delta2 = x - mean        # note: uses updated mean
    M2 += delta * delta2

Finalize:
    variance = M2 / count
    inv_std = 1 / sqrt(variance + eps)
```

**Why Welford over naive two-pass?**
- Single pass: reads data once (important since GroupNorm is memory-bound).
- Numerically stable: no subtraction of large similar numbers.
- ~8% slower than naive one-pass on CPU benchmarks (PyTorch issue #69525), but the stability gain is worth it.

**Why Welford over naive one-pass (`E[x^2] - E[x]^2`)?**
- The naive formula `E[x^2] - E[x]^2` can produce negative variance due to floating-point cancellation.
- Welford always produces non-negative variance.

### 8. NCHW vs NHWC Layout

GroupNorm has different memory access patterns depending on layout:

- **NCHW**: Channels of the same group are contiguous in memory, but spatial elements within a channel are also contiguous. Reduction over `(C_g, H, W)` accesses contiguous memory blocks. This is the default PyTorch layout.
- **NHWC**: At each spatial position, all channels are contiguous. This is preferred by cuDNN/Tensor Cores for convolution. For GroupNorm, each group's elements are strided in memory, requiring gather operations.

NVIDIA Apex provides dedicated NHWC GroupNorm kernels (`nhwc_fprop`, `nhwc_bprop`) that handle the strided access pattern efficiently, choosing between one-pass and two-pass algorithms based on spatial size:
- `H*W <= 256` (or 1024 on SM80+): one-pass kernel
- `H*W > threshold`: two-pass kernel (first pass computes mean/var, second pass normalizes)

**For SharpInference CPU (NCHW)**: The contiguous layout makes SIMD vectorization straightforward — process 8 floats at a time with AVX2 `Vector256<float>`.

---

## Key Numbers / Constants

| Constant | Value | Source |
|----------|-------|--------|
| **epsilon (eps)** | **1e-5** (SD 1.5 UNet), **1e-6** (SD 1.5 Transformer blocks) | HuggingFace diffusers, PyTorch default |
| **num_groups (G)** | **32** | Wu & He (2018) paper default, used by all SD variants |
| **affine** | **True** | All diffusion models use learnable gamma/beta |
| **FP16 max value** | 65,504 | IEEE 754 half-precision spec |
| **FP16 mantissa bits** | 10 (+ 1 implicit) | IEEE 754 |
| **FP32 mantissa bits** | 23 (+ 1 implicit) | IEEE 754 |
| **Typical reduction size** | 10,240 - 40,960 elements | SD 1.5 with 64x64 latent |
| **Welford overhead vs naive** | ~8% on CPU | PyTorch issue #69525 |

---

## Data Layouts / Formats

### Input Tensor
```
Shape: (N, C, H, W) — batch, channels, height, width
dtype: float32 or float16
Layout: NCHW (default) or NHWC (channels-last, preferred for GPU convolution)
```

### Learnable Parameters
```
gamma (weight): shape (C,), dtype float32 (even when input is float16)
beta  (bias):   shape (C,), dtype float32 (even when input is float16)
```

### Internal Intermediates (per sample, per group)
```
mean:  shape (N, G), dtype float32 (always)
var:   shape (N, G), dtype float32 (always)
rstd:  shape (N, G), dtype float32 — reciprocal std = 1/sqrt(var + eps)
```

### Output Tensor
```
Shape: (N, C, H, W) — same as input
dtype: same as input (float32 or float16)
```

---

## Algorithm Steps

### GroupNorm Forward Pass (NCHW, with optional SiLU fusion)

```
Input:  x[N, C, H, W], gamma[C], beta[C], G, eps
Output: y[N, C, H, W]

C_g = C / G                           // channels per group

For each sample n in [0, N):
  For each group g in [0, G):
    // --- Pass 1: Compute mean and variance (Welford) ---
    mean = 0.0f
    M2   = 0.0f
    count = 0

    For c in [g * C_g, (g+1) * C_g):      // channels in this group
      For h in [0, H):
        For w in [0, W):
          val = (float)x[n, c, h, w]       // cast to FP32
          count += 1
          delta  = val - mean
          mean  += delta / count
          delta2 = val - mean
          M2    += delta * delta2

    var  = M2 / count
    rstd = 1.0f / sqrt(var + eps)

    // --- Pass 2: Normalize, scale, shift, [activate] ---
    For c in [g * C_g, (g+1) * C_g):
      For h in [0, H):
        For w in [0, W):
          val = (float)x[n, c, h, w]
          norm = (val - mean) * rstd
          out  = norm * gamma[c] + beta[c]

          // Optional SiLU fusion:
          // out = out * sigmoid(out)
          // out = out * (1.0f / (1.0f + exp(-out)))

          y[n, c, h, w] = (input_dtype)out   // cast back
```

### SIMD-Optimized CPU Path (AVX2)

For the inner loops over spatial dimensions:
```csharp
// Process 8 floats at a time using Vector256<float>
var vMean = Vector256.Create(mean);
var vRstd = Vector256.Create(rstd);
var vGamma = Vector256.Create(gamma[c]);
var vBeta  = Vector256.Create(beta[c]);

for (int i = 0; i < H * W; i += 8)
{
    var vx = Vector256.LoadUnsafe(ref xPtr, i);  // load 8 floats
    var vNorm = (vx - vMean) * vRstd;
    var vOut  = vNorm * vGamma + vBeta;
    // SiLU fusion: vOut = vOut * Sigmoid(vOut)
    Vector256.StoreUnsafe(vOut, ref yPtr, i);
}
```

For the Welford mean/variance pass, use `TensorPrimitives` or a manual SIMD Welford with horizontal reduction at the end.

---

## Reference Implementations

### PyTorch Native (C++/CUDA)

- **Source**: `aten/src/ATen/native/cuda/group_norm_kernel.cu`
- **Algorithm**: Welford online algorithm via `WelfordData<T_ACC, int64_t>` and `WelfordOps`
- **Accumulation**: `acc_type<T, true>` maps `half -> float`, `bfloat16 -> float`
- **Dispatch**: `AT_DISPATCH_FLOATING_TYPES_AND_HALF` covers float32, float64, float16, bfloat16
- **Affine**: Fused into `ComputeFusedParamsCUDAKernel` which precomputes `a = gamma * rstd` and `b = beta - mean * a` so the normalize+scale+shift is a single FMA: `y = a * x + b`
- **URL**: [PyTorch group_norm_kernel.cu](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/group_norm_kernel.cu)

### NVIDIA Apex (CUDA, NHWC-optimized)

- **Source**: `apex/contrib/group_norm/group_norm.py` + custom CUDA kernels
- **Layout**: NHWC only (assertion enforced)
- **Fusion**: SiLU/Swish fused via `act="silu"` parameter
- **Algorithm selection**: One-pass for small spatial (`H*W <= 256`), two-pass for large spatial
- **Mixed precision**: Supports `(input=fp16, params=fp32)` and other combinations
- **URL**: [NVIDIA Apex GroupNorm](https://github.com/NVIDIA/apex/blob/master/apex/contrib/group_norm/group_norm.py)

### channels-last-groupnorm (CUDA, NHWC)

- **Source**: Drop-in replacement for `torch.nn.GroupNorm`
- **Fused activations**: identity, ReLU, SiLU, GeLU, GeLU-tanh
- **Performance**: ~50% faster than Triton GroupNorm, ~35% e2e with NHWC conv
- **URL**: [channels-last-groupnorm](https://github.com/latentCall145/channels-last-groupnorm)

### HuggingFace diffusers

- **Source**: `diffusers/models/normalization.py`
- **AdaGroupNorm**: GroupNorm modified to incorporate timestep embeddings (scale/shift from time embedding)
- **Defaults**: `num_groups=32`, `eps=1e-5`, `affine=True`
- **URL**: [diffusers normalization.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/normalization.py)

---

## Differences Between Implementations

| Aspect | PyTorch native | NVIDIA Apex | channels-last-groupnorm |
|--------|---------------|-------------|------------------------|
| **Layout** | NCHW (default) | NHWC only | NHWC only |
| **Algorithm** | Welford (single-pass) | One-pass or two-pass (adaptive) | Not documented |
| **SiLU fusion** | No | Yes (`act="silu"`) | Yes (multiple activations) |
| **FP16 accum** | FP32 via `acc_type` | FP32 (mixed dtype support) | Assumed FP32 |
| **Affine fusion** | Yes (`a*x+b` precompute) | Yes | Yes |
| **Two-pass threshold** | N/A (always Welford) | H*W > 256 (SM<80) or 1024 (SM>=80) | Not documented |
| **Conv2D fusion** | No | No | No (but benefits from NHWC conv) |

---

## Open Questions

- [ ] **Conv2D -> GroupNorm -> SiLU triple fusion**: Is it worth fusing the Conv2D write with GroupNorm+SiLU? Requires Conv2D kernel modification. Likely only for CUDA path.
- [ ] **TensorPrimitives for Welford**: Does .NET 10 `TensorPrimitives` expose a Welford-based mean/variance, or do we need to hand-roll with SIMD intrinsics?
- [ ] **AVX-512 GroupNorm**: With `Vector512<float>` processing 16 floats at a time, is there a meaningful speedup over AVX2 for the memory-bound GroupNorm kernel?
- [ ] **Adaptive algorithm selection on CPU**: Should SharpInference use one-pass Welford for small spatial sizes and two-pass for large, similar to Apex's strategy?

---

## Implementation Notes

### For SharpInference.Cpu

1. **Always accumulate in FP32.** Even if the tensor storage is FP16, cast each element to `float` before accumulating mean/variance. Cast back to input dtype after applying gamma/beta.

2. **Use Welford's algorithm** for the mean/variance pass. The ~8% overhead vs naive is acceptable for the numerical stability guarantee. Avoid the naive `E[x^2] - E[x]^2` formula — it will produce incorrect results for FP16 inputs.

3. **Fuse GroupNorm + SiLU** into a single method. The normalize pass (pass 2) should optionally apply `SiLU(x) = x / (1 + exp(-x))` before writing the output. Use a boolean or enum parameter, not a delegate, to avoid per-element virtual dispatch.

4. **Precompute fused affine parameters** as PyTorch does:
   ```
   a[c] = gamma[c] * rstd[g]
   b[c] = beta[c] - mean[g] * a[c]
   ```
   Then the normalize+scale+shift reduces to `y = a * x + b` (single FMA).

5. **SIMD strategy**: For the normalize pass, use `Vector256<float>` (AVX2) to process 8 elements at a time. The mean/variance pass can also be vectorized using a SIMD Welford (accumulate 8 parallel Welford streams, then horizontally reduce at the end).

6. **Memory layout**: Start with NCHW (natural for C# row-major arrays). Channels within a group are contiguous, so the reduction loops access memory sequentially — good for prefetchers.

7. **Thread parallelism**: Each `(n, g)` pair is independent. Parallelize over `N * G` work items using `Parallel.For` or similar. For SD 1.5 with batch=1, G=32, this gives 32 independent work items.

8. **Epsilon**: Default to `1e-5f`. Allow override via constructor parameter for models that use `1e-6`.

9. **SiLU approximation**: For CPU, `exp(-x)` is expensive. Consider a fast polynomial approximation of sigmoid for the fused SiLU path, with an accuracy-vs-speed tradeoff flag.

### For SharpInference.Cuda (future)

1. Follow the NHWC layout to match cuDNN convolution output format, avoiding transpose overhead.
2. Implement fused GroupNorm+SiLU as a single PTX kernel.
3. Use Apex's adaptive one-pass/two-pass strategy based on spatial size.
4. Consider `__half2` CUDA intrinsics for reading FP16 data 2 elements at a time, then widening to FP32 for accumulation.

---

## Sources

- [Group Normalization (Wu & He, 2018) — arXiv](https://arxiv.org/abs/1803.08494)
- [Group Normalization — ECCV 2018 PDF](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.pdf)
- [PyTorch GroupNorm documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html)
- [PyTorch group_norm_kernel.cu source](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/group_norm_kernel.cu)
- [PyTorch Numerical Accuracy notes](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
- [NVIDIA Apex GroupNorm (NHWC + SiLU fusion)](https://github.com/NVIDIA/apex/blob/master/apex/contrib/group_norm/group_norm.py)
- [channels-last-groupnorm (fused NHWC kernels)](https://github.com/latentCall145/channels-last-groupnorm)
- [HuggingFace diffusers normalization API](https://huggingface.co/docs/diffusers/api/normalization)
- [Welford's algorithm — Wikipedia](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance)
- [NVIDIA Mixed Precision Training guide](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)
- [PyTorch GroupNorm FP16 issue #17216](https://github.com/pytorch/pytorch/issues/17216)
- [PyTorch Welford for GroupNorm issue #69525](https://github.com/pytorch/pytorch/issues/69525)
