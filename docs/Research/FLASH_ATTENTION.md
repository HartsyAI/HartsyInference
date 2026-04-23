# Flash Attention — Research Notes

---

## Summary

Flash Attention (Dao et al., 2022) is an IO-aware exact attention algorithm that computes scaled dot-product attention in **O(N) memory** instead of the naive O(N^2), by tiling the computation into SRAM-sized blocks and never materializing the full N x N attention matrix. The key algorithmic insight is the **online softmax trick**: by maintaining running statistics (a per-row maximum `m` and a per-row sum-of-exponentials `l`), the softmax numerator and denominator can be computed incrementally across tiles, with prior partial results rescaled when a new maximum is encountered. The result is mathematically identical to standard attention.

For diffusion model cross-attention, Q comes from spatial features (up to 4096 tokens at 512x512, or 16384+ at 1024x1024) while K and V come from text embeddings (77 tokens for CLIP, 256 for T5). This asymmetry means the KV sequence is short enough to fit entirely in SRAM in many cases, and the outer loop over Q blocks dominates. Flash Attention handles self-attention and cross-attention with the same kernel -- the only difference is the source of K/V and the sequence lengths.

Flash Attention 2 (Dao, 2023) swaps the loop order (outer over Q blocks, inner over KV blocks), reduces non-matmul FLOPs, and improves warp-level parallelism, reaching 50-73% of theoretical A100 FLOPs (vs 25-40% for FA1). Flash Attention 3 (Dao et al., 2024) targets Hopper GPUs with asynchronous TMA, warp specialization, and FP8 support, reaching 840 TFLOPs/s in BF16 on H100.

---

## Detailed Findings

### 1. Why Standard Attention Is Memory-Bound

Standard scaled dot-product attention computes:

```
S = Q @ K^T / sqrt(d)       # N_q x N_kv
P = softmax(S, dim=-1)      # N_q x N_kv
O = P @ V                   # N_q x d
```

The intermediate matrices S and P are each N_q x N_kv. For self-attention at 64x64 latent resolution (4096 tokens), S alone is 4096 x 4096 = 16M elements = 64 MB in FP32. At 128x128 latent (1024x1024 image), that becomes 16384 x 16384 = 1 GB. These must be written to and read from HBM, and HBM bandwidth (1.5-2.0 TB/s on A100) becomes the bottleneck -- GPU compute units sit idle waiting for memory.

### 2. The IO-Awareness Insight

The GPU memory hierarchy has two relevant tiers:

| Memory | Capacity | Bandwidth |
|--------|----------|-----------|
| HBM (global) | 40-80 GB (A100) | 1.5-2.0 TB/s |
| SRAM (shared memory) | 192 KB per SM, ~20 MB total (A100, 108 SMs) | ~19 TB/s |

SRAM is ~10x faster than HBM but ~4000x smaller. Flash Attention restructures the computation so that all intermediate results (S_ij, P_ij tiles) live entirely in SRAM, and only the final output O and the small statistics vectors (m, l) are written to HBM.

### 3. The Online Softmax Trick

The standard softmax for a row x of length N is:

```
m(x) = max_j(x_j)
l(x) = sum_j exp(x_j - m(x))
softmax(x)_i = exp(x_i - m(x)) / l(x)
```

The subtraction of m(x) is the "safe softmax" trick for numerical stability. The **online softmax** extends this to process x in chunks. Given two chunks processed so far, with statistics (m_old, l_old) for the first and (m_cur, l_cur) for the second:

```
m_new = max(m_old, m_cur)
l_new = l_old * exp(m_old - m_new) + l_cur * exp(m_cur - m_new)
```

The critical rescaling of the running output accumulator is:

```
O_new = diag(l_old * exp(m_old - m_new)) * O_old
      + diag(exp(m_cur - m_new)) * P_cur @ V_cur

O_new = O_new / diag(l_new)
```

Or equivalently per-row:

```
alpha = exp(m_old - m_new)
beta  = exp(m_cur - m_new)
O_new = (alpha * l_old * O_old + beta * P_tilde_cur @ V_cur) / l_new
```

where `P_tilde_cur` has entries `exp(S_ij - m_cur)`. This is exact -- no approximation.

### 4. Flash Attention 1 vs 2 Loop Order

**FA1** (Dao et al., 2022): Outer loop over KV blocks (columns), inner loop over Q blocks (rows). Each KV block is loaded once, but O_i must be read and written for every KV block iteration, and statistics must be exchanged.

**FA2** (Dao, 2023): Outer loop over Q blocks (rows), inner loop over KV blocks (columns). Each Q block and its associated O_i, m_i, l_i are loaded once into SRAM for the duration of the inner loop. This is better because:

- The outer loop is **embarrassingly parallel** across Q blocks -- thread blocks do not need to communicate
- O_i is kept in registers/SRAM for the entire inner loop, reducing HBM writes
- Statistics (m_i, l_i) stay in registers, never written to HBM until the final normalization

### 5. Flash Attention 2 Additional Optimizations

1. **Reduced non-matmul FLOPs**: The online softmax rescaling was rewritten to minimize scalar operations. On A100, non-matmul FLOPs are 16x more expensive than matmul FLOPs (Tensor Core vs CUDA core), so minimizing them is critical.

2. **Warp partitioning change**: FA1 used "sliced-K" -- K and V split across warps, requiring shared-memory synchronization. FA2 splits Q across warps while K and V are shared, eliminating inter-warp communication.

3. **Sequence-length parallelism**: In addition to parallelizing over batch and heads, FA2 parallelizes over the sequence length dimension, improving occupancy on long sequences.

### 6. Flash Attention 3 (Hopper GPUs)

FA3 (Shah et al., 2024) adds three techniques for H100:

1. **Warp specialization**: Producer warps handle TMA (Tensor Memory Accelerator) data movement; consumer warps handle Tensor Core computation. Hopper's `setmaxnreg` dynamically reallocates registers between warp groups.
2. **Asynchronous softmax**: Softmax computation is interleaved/pipelined with the next tile's matmul using the WGMMA instruction.
3. **FP8 with block quantization**: 2x throughput per SM vs BF16. Incoherent processing with random orthogonal matrices reduces quantization error -- 2.6x more accurate than naive per-tensor FP8.

Performance: 840 TFLOPs/s BF16, 1.3 PFLOPs/s FP8 on H100 (85% utilization).

### 7. Cross-Attention in Diffusion Models

In diffusion UNets, cross-attention layers condition spatial features on text embeddings:

- **Q** comes from the spatial feature map (flattened H x W tokens)
- **K, V** come from the text encoder output (CLIP: 77 tokens x 768d; SDXL second encoder: 77 tokens x 1280d; T5-XXL in SD3/Flux: up to 256 tokens x 4096d)

The asymmetry (N_q >> N_kv) means the full K^T can often fit in SRAM. For d=64 and 77 KV tokens, K is 77 x 64 = 4928 elements = ~20 KB in FP32, well within the 192 KB SRAM budget.

### 8. Self-Attention and Cross-Attention: Same Kernel

Flash Attention handles both with the same kernel. The algorithm is parameterized by (N_q, N_kv, d) -- for self-attention N_q = N_kv, for cross-attention N_q != N_kv. The tiling proceeds identically: Q is tiled into blocks of size B_r, KV is tiled into blocks of size B_c. The only behavioral difference is the number of inner loop iterations (ceil(N_kv / B_c)).

The official `flash-attn` library API (`flash_attn_func`) accepts separate Q, K, V tensors of different sequence lengths natively.

---

## Key Numbers / Constants

### GPU Memory Hierarchy (A100)

| Parameter | Value |
|-----------|-------|
| HBM capacity | 40 GB (A100-40GB) or 80 GB (A100-80GB) |
| HBM bandwidth | 1.5 TB/s (40GB) or 2.0 TB/s (80GB) |
| SRAM per SM | 192 KB |
| Number of SMs | 108 |
| Total SRAM | ~20 MB |
| SRAM bandwidth | ~19 TB/s |
| Matmul vs non-matmul cost ratio | 1:16 (Tensor Core vs CUDA core) |

### Typical Diffusion Model Attention Dimensions

| Model | Resolution | Latent | Attention Levels | Max Spatial Tokens | KV Tokens | Head Dim |
|-------|-----------|--------|-----------------|-------------------|-----------|----------|
| SD 1.5 | 512x512 | 64x64 | 32x32, 16x16, 8x8 | 1024 (32x32) | 77 | 40 (8 heads, 320ch) |
| SDXL | 1024x1024 | 128x128 | 64x64, 32x32, 16x16 | 4096 (64x64) | 77 | 40 or 80 |
| SD3 | 1024x1024 | 128x128 | Joint attention | 16384 (128x128) | 77+256 (CLIP+T5) | 64 |
| Flux | 1024x1024 | 128x128 | Joint attention | 16384 (128x128) | 256 (T5) | 128 |

Note: SD 1.5 applies attention only at resolutions 32x32, 16x16, and 8x8 (not at 64x64). The latent is 64x64 but the first downsampling block does not use attention. SDXL adds attention at 64x64.

### Memory Comparison: Naive vs Flash Attention

| Scenario | N_q | N_kv | Naive S matrix (FP32) | Flash Attention extra mem |
|----------|-----|------|-----------------------|--------------------------|
| SD1.5 self-attn 32x32 | 1024 | 1024 | 4 MB | ~8 KB (m + l vectors) |
| SD1.5 cross-attn 32x32 | 1024 | 77 | 308 KB | ~8 KB |
| SDXL self-attn 64x64 | 4096 | 4096 | 64 MB | ~32 KB |
| SDXL cross-attn 64x64 | 4096 | 77 | 1.2 MB | ~32 KB |
| SD3 joint-attn 128x128 | 16384 | 16384 | 1 GB | ~128 KB |

Flash Attention stores O (N_q x d) plus m, l vectors (N_q each) -- all O(N_q * d) memory.

### Block Size Formula

```
B_c = ceil(M / (4 * d))    -- KV block size (number of KV tokens per tile)
B_r = min(ceil(M / (4 * d)), d)  -- Q block size (number of Q tokens per tile)
```

Where M = available SRAM in elements (not bytes). For FP32 with 192 KB SRAM: M = 192 * 1024 / 4 = 49152 elements.

| Head dim (d) | B_c = ceil(M/4d) | B_r = min(B_c, d) |
|-------------|-------------------|---------------------|
| 40 | 307 | 40 |
| 64 | 192 | 64 |
| 80 | 153 | 80 |
| 128 | 96 | 96 |

In practice, implementations use powers-of-2 block sizes (32, 64, 128, 256) for alignment. The official FA2 implementation typically uses B_r = B_c = 128 for d=64 and B_r = B_c = 64 for d=128.

### HBM Access Complexity

| Algorithm | HBM reads/writes |
|-----------|-----------------|
| Standard attention | O(N_q * N_kv + N_q * d) -- must read/write full S and P |
| Flash Attention | O(N_q * N_kv * d / M) -- reduced by factor M/d |

For typical M >> d, this is a significant reduction. With M = 49152, d = 64: reduction factor ~768x for HBM accesses of the attention matrix.

---

## Data Layouts / Formats

### Input Tensors

```
Q: [batch, n_heads, N_q, d]    -- spatial features (from conv/linear projection)
K: [batch, n_heads, N_kv, d]   -- from text encoder (cross-attn) or spatial (self-attn)
V: [batch, n_heads, N_kv, d]   -- same source as K
O: [batch, n_heads, N_q, d]    -- output, same shape as Q
```

Flash Attention implementations often prefer a **packed layout** `[batch, seq_len, n_heads, d]` (heads not leading) for better memory coalescing, then transpose internally.

### Per-Row Statistics (kept in SRAM, written once at end)

```
m: [batch, n_heads, N_q]   -- per-row running maximum of attention scores
l: [batch, n_heads, N_q]   -- per-row running sum of exp(score - m)
```

These are only needed during computation. In the forward-only case (inference), they can be discarded after the final normalization O_i = O_i / l_i. For training (backward pass), m and l must be saved for recomputation.

### Precision Considerations

- Attention scores S_ij should be computed in FP32 for numerical stability, even when Q, K, V are FP16/BF16
- The exp() and max() operations in online softmax must use FP32 accumulators
- Output O can be accumulated in FP32 and converted to FP16/BF16 at the end
- FA2/FA3 on GPU use FP16/BF16 for matmuls (Tensor Cores) but FP32 for softmax accumulation

---

## Algorithm Steps

### Flash Attention 2 Forward Pass (Inference)

This is the preferred algorithm (FA2 loop order with outer loop over Q blocks).

```
FLASH_ATTENTION_FORWARD(Q, K, V, scale):
  Input:
    Q  ∈ R^{N_q × d}   (queries, one head)
    K  ∈ R^{N_kv × d}   (keys)
    V  ∈ R^{N_kv × d}   (values)
    scale = 1/sqrt(d)

  Output:
    O  ∈ R^{N_q × d}    (attention output)

  // Block sizes (constrained by SRAM capacity M in elements)
  B_r ← chosen Q block size (e.g., 64 or 128)
  B_c ← chosen KV block size (e.g., 64 or 128)
  T_r ← ceil(N_q / B_r)     // number of Q blocks
  T_c ← ceil(N_kv / B_c)    // number of KV blocks

  // Outer loop: iterate over Q blocks (PARALLEL across thread blocks)
  for i = 0 to T_r - 1 do:

    // Load Q block into SRAM
    Q_i ← Q[i*B_r : (i+1)*B_r, :]          // B_r × d, from HBM

    // Initialize per-row accumulators (in SRAM / registers)
    O_i ← zeros(B_r, d)                      // running output
    m_i ← fill(B_r, -infinity)               // running row-wise max
    l_i ← zeros(B_r)                         // running row-wise sum of exp

    // Inner loop: iterate over KV blocks (SEQUENTIAL, accumulates statistics)
    for j = 0 to T_c - 1 do:

      // Load KV block into SRAM
      K_j ← K[j*B_c : (j+1)*B_c, :]        // B_c × d, from HBM
      V_j ← V[j*B_c : (j+1)*B_c, :]        // B_c × d, from HBM

      // Step 1: Compute attention scores for this tile
      S_ij ← Q_i @ K_j^T * scale            // B_r × B_c, in SRAM

      // Step 2: Compute local tile statistics
      m_ij ← rowmax(S_ij)                    // B_r, max of each row in this tile
      P_ij ← exp(S_ij - m_ij[:, None])       // B_r × B_c, safe exp
      l_ij ← rowsum(P_ij)                    // B_r, sum of exp per row in tile

      // Step 3: Update running maximum
      m_new ← max(m_i, m_ij)                 // B_r, element-wise max

      // Step 4: Compute rescaling factors
      alpha ← exp(m_i - m_new)               // B_r, rescale old accumulator
      beta  ← exp(m_ij - m_new)              // B_r, rescale current tile

      // Step 5: Update running sum (denominator)
      l_new ← alpha * l_i + beta * l_ij      // B_r

      // Step 6: Rescale old output and add new contribution
      //   O_i was accumulated with the old max m_i
      //   We must rescale it to be consistent with m_new
      O_i ← diag(alpha) @ O_i + diag(beta) @ P_ij @ V_j
      //   Equivalently per row r:
      //   O_i[r] = alpha[r] * O_i[r] + beta[r] * P_ij[r, :] @ V_j

      // Step 7: Update statistics
      m_i ← m_new
      l_i ← l_new

    end for  // KV blocks

    // Step 8: Final normalization -- divide by the total softmax denominator
    O_i ← diag(1 / l_i) @ O_i
    //   Equivalently per row r: O_i[r] = O_i[r] / l_i[r]

    // Write final output block to HBM
    O[i*B_r : (i+1)*B_r, :] ← O_i

  end for  // Q blocks

  return O
```

### Why This Is Exact

After processing all T_c KV blocks, for each row r of Q:

```
m_i[r] = max over all j of (max over columns of S[r, j*B_c:(j+1)*B_c])
        = max of entire row r of S

l_i[r] = sum over all j of (sum_k exp(S[r, j*B_c+k] - m_i[r]))
        = sum_k exp(S[r, k] - m_i[r])
        = the correct softmax denominator for row r

O_i[r] = sum over all j of (exp(S[r, j*B_c:...] - m_i[r]) @ V[j*B_c:...])
        = (softmax(S[r, :]) unnormalized) @ V

O_i[r] / l_i[r] = softmax(S[r, :]) @ V = standard attention output
```

The rescaling with alpha = exp(m_old - m_new) ensures that when a new tile introduces a larger maximum, all previously accumulated values are adjusted to be relative to the new maximum. This is the "telescoping" property of exponentials: exp(x - a) = exp(x - b) * exp(b - a).

### Adaptation for Cross-Attention

For cross-attention in diffusion models (N_q >> N_kv), the algorithm is identical. The inner loop simply has fewer iterations:

- Self-attention at 32x32 (1024 tokens), B_c=128: T_c = 8 inner iterations
- Cross-attention at 32x32 (Q=1024, KV=77), B_c=128: T_c = 1 inner iteration

When N_kv <= B_c, the entire KV sequence fits in a single tile and the inner loop body executes once. In this case, the online softmax simplifies (m_new = m_ij, alpha = 0 since m_i starts at -inf on first iteration) and Flash Attention reduces to the standard computation but still benefits from avoiding HBM materialization of S.

---

## Reference Implementations

1. **Official flash-attn library** (Dao-AILab/flash-attention)
   - CUDA kernels in Triton and hand-written CUDA
   - Supports FA1, FA2, FA3
   - API: `flash_attn_func(q, k, v, softmax_scale=None, causal=False)`
   - Handles variable sequence lengths via `flash_attn_varlen_func` with `cu_seqlens_q`, `cu_seqlens_k`
   - Head dimensions supported: 32, 64, 96, 128, 160, 192, 224, 256
   - Source: [github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)

2. **PyTorch scaled_dot_product_attention (SDPA)**
   - `torch.nn.functional.scaled_dot_product_attention`
   - Dispatches to Flash Attention backend automatically when available
   - Falls back to memory-efficient attention (xFormers-style) or math backend
   - Source: PyTorch documentation

3. **xFormers memory-efficient attention**
   - Similar tiling approach, predates flash-attn somewhat
   - Used extensively in Stable Diffusion / diffusers
   - Source: [github.com/facebookresearch/xformers](https://github.com/facebookresearch/xformers)

4. **HuggingFace diffusers**
   - Uses `torch.nn.functional.scaled_dot_product_attention` in attention processors
   - `AttnProcessor2_0` is the default SDPA-backed processor
   - Source: [github.com/huggingface/diffusers](https://github.com/huggingface/diffusers)

---

## Differences Between Implementations

| Aspect | Flash Attention 1 | Flash Attention 2 | Flash Attention 3 |
|--------|-------------------|--------------------|--------------------|
| Paper | Dao et al., 2022 | Dao, 2023 | Shah, Dao et al., 2024 |
| Loop order | Outer: KV, Inner: Q | Outer: Q, Inner: KV | Outer: Q, Inner: KV (pipelined) |
| Parallelism | Over batch, heads | Over batch, heads, seq_len | Over batch, heads, seq_len + warp specialization |
| Warp strategy | Sliced-K (sync required) | Sliced-Q (no sync) | Producer/consumer warp groups |
| A100 FLOPs util | 25-40% | 50-73% | N/A (H100 only) |
| H100 FLOPs util | N/A | ~35% | ~85% (BF16) |
| FP8 support | No | No | Yes (block quantization) |
| Causal masking | Yes | Yes (optimized) | Yes |
| Cross-attention | Yes (same kernel) | Yes (same kernel) | Yes (same kernel) |
| Head dims | 64, 128 | 32-256 | 64-256 |
| Backward pass | Recomputation from saved m, l | Recomputation (optimized) | Recomputation (async) |

### CPU vs GPU Considerations for SharpInference

For a pure C# implementation targeting CPU:

- No SRAM/HBM distinction -- the relevant hierarchy is **L1/L2 cache vs main memory (DRAM)**
- L1 cache: 32-48 KB per core; L2: 256 KB-1 MB per core; L3: shared, 16-64 MB
- The tiling strategy still applies: tile Q, K, V blocks to fit in L2 cache
- Block sizes should target L2: for d=64 in FP32, a block of 64 tokens = 64*64*4 = 16 KB, so B_r = B_c = 64-128 fits comfortably in L2
- SIMD (AVX2/AVX-512) should be used for the matmul tiles and exp() computation
- The sequential inner loop over KV blocks maps naturally to single-threaded execution per Q block, with parallelism over Q blocks using .NET thread pool

For CUDA (SharpInference.Cuda via PTX):

- Implement FA2 algorithm directly in PTX or CUDA
- Use shared memory (SRAM) for tiles, registers for m, l, O accumulators
- Parallelize outer loop over Q blocks across thread blocks
- Use Tensor Cores (wmma/mma instructions) for the Q_i @ K_j^T and P_ij @ V_j matmuls
- For diffusion cross-attention where N_kv is small (77), consider loading all of K, V into shared memory once

---

## Open Questions

- [ ] For the CPU implementation, does the exp() computation in the online softmax dominate? If so, consider fast-exp approximations (with error bounds).
- [ ] For the CUDA implementation, should we write a custom PTX kernel or wrap the official flash-attn library? Custom gives control but the official library is highly optimized.
- [ ] For FP16 inference, can the entire online softmax (including m, l) be done in FP16, or must m, l, and exp() use FP32? (Almost certainly FP32 is required for m, l, and exp accumulators based on all reference implementations.)

---

## Implementation Notes

### For SharpInference.Cpu

1. **Tiling strategy**: Tile Q into blocks of B_r rows. For each Q block, iterate over KV in blocks of B_c. Target B_r = B_c = 64 for d <= 80, B_r = B_c = 32 for d = 128, to fit tiles in L2 cache.

2. **Parallelism**: Use `Parallel.For` over Q blocks (outer loop). Each thread processes one Q block sequentially through all KV blocks. For multi-head attention, parallelize over (batch, head, q_block).

3. **SIMD usage**:
   - Tile matmul (Q_i @ K_j^T): use AVX2 `Vector256<float>` FMA for the inner dot products, or `TensorPrimitives.Dot` if available for the tile sizes
   - exp() computation: use `MathF.Exp` per element or a SIMD exp approximation. .NET's `Vector256` does not have a native exp, but `TensorPrimitives.Exp` on `Span<float>` may use SIMD internally
   - Row-max and row-sum: straightforward SIMD reductions

4. **Memory layout**: Store Q, K, V in row-major `[seq_len, d]` per head. Allocate O as same shape. Allocate m, l as `[seq_len]` vectors, reusable across heads.

5. **Cross-attention optimization**: When N_kv <= B_c (e.g., 77 CLIP tokens with B_c = 128), load all of K and V once and iterate only over Q blocks. This is the common case for diffusion cross-attention.

### For SharpInference.Cuda

1. **Kernel design**: One CUDA kernel for the forward pass. Grid dimensions: `(ceil(N_q / B_r), batch * n_heads)`. Each thread block handles one Q tile.

2. **Shared memory allocation**: `K_tile[B_c][d]` + `V_tile[B_c][d]` + `S_tile[B_r][B_c]` + `O_tile[B_r][d]`. For B_r = B_c = 64, d = 64 in FP16: (64*64 + 64*64 + 64*64 + 64*64) * 2 bytes = 32 KB, well within 48 KB default shared memory.

3. **Warp partitioning (FA2 style)**: Split Q_tile rows across warps. 4 warps per block, each warp handles B_r/4 rows. K and V tiles are shared across all warps (no sync needed for reads).

4. **Tensor Core usage**: Use `wmma` (Warp Matrix Multiply-Accumulate) for the B_r x B_c x d matmul tiles when d is a multiple of 16.

5. **For diffusion cross-attention with N_kv = 77**: Consider padding KV to 80 or 128 for alignment, with attention mask to ignore padding. This avoids wasted Tensor Core throughput from non-power-of-2 dimensions.

---

## Sources

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning (Dao, 2023)](https://arxiv.org/abs/2307.08691)
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision (Shah, Dao et al., 2024)](https://arxiv.org/abs/2407.08608)
- [Stanford CRFM: FlashAttention-2 Blog Post](https://crfm.stanford.edu/2023/07/17/flash2.html)
- [Reimplementing FlashAttention for Performance (Amine Diro)](https://aminediro.com/posts/flash_attn/)
- [Flash Attention: The Mathematical Tricks That Broke the Memory Wall (MdJawad)](https://mdjawad.com/posts/flash-attention/)
- [Basic Idea Behind Flash Attention (Damek Davis)](https://damek.github.io/random/basic-idea-behind-flash-attention/)
- [Flavors of Attention in Modern Diffusion Models (Sayak Paul)](https://sayak.dev/posts/attn-diffusion.html)
- [NVIDIA: Tuning Flash Attention for Peak Performance](https://developer.nvidia.com/blog/tuning-flash-attention-for-peak-performance-in-nvidia-cuda-tile/)
- [Dao-AILab/flash-attention GitHub Repository](https://github.com/Dao-AILab/flash-attention)
- [U-Net for Stable Diffusion (labml.ai)](https://nn.labml.ai/diffusion/stable_diffusion/model/unet.html)
