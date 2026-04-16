# PTX Kernels — Research Notes

> Status: Complete
> Last Updated: 2026-04-16
> Needed Before: SharpInference.Cuda / Ptx

---

## Summary

All custom CUDA kernels in SharpInference are written as PTX (Parallel Thread Execution) assembly, embedded as `.ptx` resources in the `SharpInference.Cuda` assembly, and JIT-compiled at runtime via the CUDA Driver API (`cuModuleLoadData` / `cuModuleGetFunction`). This document covers the PTX ISA for compute capabilities 8.0 (Ampere A100), 8.6 (Ampere consumer — RTX 3060-3090), and 8.9 (Ada Lovelace — RTX 4060-4090), including shared memory tiling patterns for Conv2D and SDPA, warp shuffle reductions for normalization kernels (GroupNorm, LayerNorm), FP16 packed arithmetic intrinsics, register pressure management, and bank-conflict-free shared memory access patterns.

Hand-written PTX provides a 7-14% performance improvement over CUDA C++ for the critical-path kernels in diffusion inference, at the cost of per-architecture tuning and significantly higher development complexity. This tradeoff is acceptable for the small number of hot kernels (Conv2D, GroupNorm+SiLU, SDPA, elementwise add/scale) that dominate inference time.

---

## Detailed Findings

### 1. PTX ISA Basics

PTX is a virtual ISA — it uses unlimited virtual registers that `ptxas` (the PTX assembler) maps to physical hardware registers during JIT compilation. A PTX file targets a specific virtual architecture (`sm_80`, `sm_86`, `sm_89`) and a PTX ISA version (e.g., `8.5` for CUDA 12.5, `9.2` for CUDA 13.2).

**Kernel header boilerplate:**
```
.version 8.5
.target sm_80
.address_size 64

.visible .entry my_kernel(
    .param .u64 param_ptr_input,
    .param .u64 param_ptr_output,
    .param .u32 param_N
)
{
    // register declarations
    .reg .b64   %rd<16>;
    .reg .b32   %r<32>;
    .reg .f32   %f<16>;
    .reg .f16x2 %h<16>;
    .reg .pred  %p<8>;

    // kernel body
    ...
    ret;
}
```

**Register types:**

| PTX Type | Width | Use |
|----------|-------|-----|
| `.pred`  | 1-bit | Predicate (branch conditions) |
| `.b16`   | 16-bit | Untyped 16-bit (carries `.f16` values) |
| `.b32`   | 32-bit | Untyped 32-bit (integers, addresses, carries `.f16x2`) |
| `.b64`   | 64-bit | 64-bit addresses, pointers |
| `.u32`   | 32-bit | Unsigned 32-bit integer |
| `.s32`   | 32-bit | Signed 32-bit integer |
| `.f16`   | 16-bit | IEEE 754 half precision (scalar) |
| `.f16x2` | 32-bit | Packed pair of two `.f16` values in a 32-bit register |
| `.f32`   | 32-bit | IEEE 754 single precision |
| `.f64`   | 64-bit | IEEE 754 double precision |

Source: [PTX ISA 9.2 - NVIDIA](https://docs.nvidia.com/cuda/parallel-thread-execution/)

### 2. FP16 Packed Arithmetic Intrinsics

The `.f16x2` type packs two half-precision floats into a single 32-bit register. This doubles throughput for elementwise operations because one instruction processes two values. These instructions require `sm_53` or higher (all our targets qualify).

**Key FP16x2 instructions:**

```
// Addition: d = a + b (two packed halves independently)
add.rn.f16x2  %h3, %h1, %h2;

// Multiplication: d = a * b
mul.rn.f16x2  %h3, %h1, %h2;

// Fused multiply-add: d = a * b + c (single rounding, higher precision)
fma.rn.f16x2  %h3, %h1, %h2, %h4;

// Scalar half precision variants (single value, upper 16 bits unused)
add.rn.f16    %h3, %h1, %h2;
mul.rn.f16    %h3, %h1, %h2;
fma.rn.f16    %h3, %h1, %h2, %h4;
```

**Rounding modes for FP16:**
- `.rn` — round to nearest even (default and recommended)
- `.rz` — round toward zero
- `.rm` — round toward negative infinity
- `.rp` — round toward positive infinity

**Conversion instructions (critical for FP32 accumulation):**
```
// FP16 -> FP32 (widen for accumulation)
cvt.f32.f16   %f1, %h1;          // scalar: extract low half to f32

// FP32 -> FP16 (narrow after accumulation)
cvt.rn.f16.f32 %h1, %f1;         // scalar: f32 to f16 with rounding

// Pack two f16 values into f16x2
mov.b32       %h_pair, {%h_lo, %h_hi};

// Unpack f16x2 into two f16 values
mov.b32       {%h_lo, %h_hi}, %h_pair;
```

**Important:** For GroupNorm/LayerNorm, mean and variance must be accumulated in FP32 to avoid precision loss. The pattern is: load as f16x2, unpack, convert to f32, accumulate in f32, convert result back to f16, repack to f16x2, store.

Source: [PTX ISA 9.2 - Half Precision Floating-Point Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/), [NVIDIA CUDA Programming Guide - Floating Point](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html)

### 3. Warp Shuffle Instructions (`shfl.sync`)

Warp shuffles exchange register values between threads within a warp (32 threads) without shared memory, making them ideal for fast reductions in normalization kernels.

**Instruction syntax:**
```
shfl.sync.mode.b32 d[|p], a, b, c, membermask;
```

where `mode` is one of `.up`, `.down`, `.bfly`, `.idx`.

**Parameters:**
- `d` — destination register (32-bit)
- `p` — optional predicate output (true if source lane was in range)
- `a` — source value from the calling thread
- `b` — lane delta (for `.up`/`.down`/`.bfly`) or absolute lane index (for `.idx`)
- `c` — clamp/segment mask (controls wrap-around behavior)
- `membermask` — 32-bit mask of participating threads (use `0xFFFFFFFF` for full warp)

**Modes:**

| Mode | Source Lane Calculation | Primary Use |
|------|----------------------|-------------|
| `.idx` | `srcLane = b` | Broadcast from specific lane |
| `.up` | `srcLane = laneId - b` | Prefix scan (exclusive) |
| `.down` | `srcLane = laneId + b` | Tree reduction (sum down) |
| `.bfly` | `srcLane = laneId XOR b` | Butterfly reduction |

**Warp reduction pattern for sum (used in GroupNorm/LayerNorm):**
```
// Assume %f1 holds this thread's partial sum, full warp participates
// Butterfly reduction: log2(32) = 5 steps
shfl.sync.bfly.b32 %f2, %f1, 16, 0x1f, 0xFFFFFFFF;
add.f32 %f1, %f1, %f2;
shfl.sync.bfly.b32 %f2, %f1,  8, 0x1f, 0xFFFFFFFF;
add.f32 %f1, %f1, %f2;
shfl.sync.bfly.b32 %f2, %f1,  4, 0x1f, 0xFFFFFFFF;
add.f32 %f1, %f1, %f2;
shfl.sync.bfly.b32 %f2, %f1,  2, 0x1f, 0xFFFFFFFF;
add.f32 %f1, %f1, %f2;
shfl.sync.bfly.b32 %f2, %f1,  1, 0x1f, 0xFFFFFFFF;
add.f32 %f1, %f1, %f2;
// %f1 now holds the sum across all 32 lanes
```

**Alternative: tree reduction with `.down` (equivalent result):**
```
shfl.sync.down.b32 %f2, %f1, 16, 0x1f, 0xFFFFFFFF;
add.f32 %f1, %f1, %f2;
shfl.sync.down.b32 %f2, %f1,  8, 0x1f, 0xFFFFFFFF;
add.f32 %f1, %f1, %f2;
shfl.sync.down.b32 %f2, %f1,  4, 0x1f, 0xFFFFFFFF;
add.f32 %f1, %f1, %f2;
shfl.sync.down.b32 %f2, %f1,  2, 0x1f, 0xFFFFFFFF;
add.f32 %f1, %f1, %f2;
shfl.sync.down.b32 %f2, %f1,  1, 0x1f, 0xFFFFFFFF;
add.f32 %f1, %f1, %f2;
// Only lane 0 holds the correct sum with .down
```

The `.bfly` pattern is preferred because all lanes hold the final result (useful when every thread needs the mean/variance), while `.down` concentrates the result in lane 0 only.

**PTX ISA version requirement:** `shfl.sync` requires PTX ISA 6.0+ (CUDA 9.0+). The non-sync variant `shfl` is deprecated.

Source: [PTX ISA - shfl.sync](https://docs.nvidia.com/cuda/parallel-thread-execution/), [NVIDIA Blog - Using CUDA Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)

### 4. Shared Memory Tiling

Shared memory (`__shared__` in CUDA C++, `.shared` state space in PTX) is on-chip SRAM that is orders of magnitude faster than global memory (HBM). Tiling is the strategy of loading a tile of input data from global memory into shared memory, synchronizing, then having all threads in the block compute from the tile.

**PTX shared memory declaration:**
```
// Static allocation (size known at compile time)
.shared .align 16 .b8 smem_A[4096];    // 4 KB tile for matrix A
.shared .align 16 .b8 smem_B[4096];    // 4 KB tile for matrix B

// Dynamic allocation (size passed at kernel launch via cuLaunchKernel)
.extern .shared .align 16 .b8 dynamic_smem[];
```

**Load/store syntax:**
```
// Load from shared memory
ld.shared.f32      %f1, [smem_A + 0];
ld.shared.v4.f32   {%f1, %f2, %f3, %f4}, [smem_A + 0];   // vectorized 128-bit load

// Store to shared memory
st.shared.f32      [smem_A + 0], %f1;
st.shared.v4.f32   [smem_A + 0], {%f1, %f2, %f3, %f4};

// Load from global to register, then store to shared (the tiling pattern)
ld.global.f32      %f1, [%rd_global_ptr];
st.shared.f32      [smem_A + offset], %f1;
bar.sync 0;   // synchronize all threads in the block before reading shared
```

**Synchronization:**
```
bar.sync 0;       // barrier — all threads in block must reach this point
                  // operand 0 is the barrier index (0-15 available)
```

### 5. Bank Conflict-Free Access Patterns

Shared memory is organized into 32 banks (matching warp size). Each bank is 4 bytes wide. Consecutive 4-byte words map to consecutive banks. A bank conflict occurs when two or more threads in the same warp access different addresses in the same bank simultaneously, serializing the accesses.

**Bank mapping formula:**
```
bank_index = (byte_address / 4) % 32
```

**Conflict-free patterns:**
- **Stride-1 access:** Thread `i` reads word `i` — each thread hits a different bank. This is the ideal case for row-major tiles where threads map to consecutive columns.
- **Broadcast:** Multiple threads read the *same* address — no conflict (hardware broadcasts).
- **Padding trick:** For column-major or transposed access, add 1 padding element per row to shift bank assignments:

```
// Without padding: 32-wide row -> column access causes 32-way bank conflict
.shared .align 16 .b8 smem_tile[32 * 32 * 4];       // 32x32 f32 tile

// With padding: 33-wide row -> column access is conflict-free
.shared .align 16 .b8 smem_tile[32 * 33 * 4];        // 32x33 f32 tile (extra column)
```

When thread `i` reads column `i` from a 32-wide row-major tile, all threads hit bank `i % 32` — perfectly conflict-free for row access. But if threads read a column (row `i`, same column), consecutive rows differ by `stride * 4 bytes = 32 * 4 = 128` bytes = 32 banks = same bank. Padding changes the stride to 33, so consecutive rows differ by 33 banks, spreading across all banks.

**For FP16x2 (2 bytes per element, packed pairs in 4 bytes):** The same bank-conflict analysis applies since `.f16x2` is 4 bytes wide. Tiles stored as `f16x2` (2 halves packed) naturally align to 4-byte bank boundaries.

**Note:** For compute capability 8.0+, newer approaches using `cp.async` (asynchronous global-to-shared copy) and the TMA (Tensor Memory Accelerator, Hopper only) can further reduce bank conflicts and hide latency. On Ampere, `cp.async` bypasses registers entirely:
```
cp.async.ca.shared.global [smem_addr], [glob_addr], 16;   // copy 16 bytes async
cp.async.commit_group;
cp.async.wait_group 0;
```

Source: [NVIDIA Developer Forums - GEMM Bank Conflict Optimization](https://forums.developer.nvidia.com/t/gemm-optimization-achieving-coalesced-and-bank-conflict-free-shared-memory-access/319329), [Lei Mao - CUDA Shared Memory Bank Conflict-Free Vectorized Access](https://leimao.github.io/blog/CUDA-Shared-Memory-Bank-Conflict-Free-Vectorized-Access/), [eunomia - CNN Convolution with Shared Memory Optimization](https://eunomia.dev/others/cuda-tutorial/06-cnn-convolution/)

### 6. Tiled Conv2D Kernel Design

For 3x3 convolution (the dominant kernel size in diffusion UNets), the tiling strategy loads an input tile (including halo region) into shared memory, then each thread computes one output pixel by reading the 3x3 neighborhood from shared memory.

**Tile sizing for 3x3 Conv2D:**
- Output tile: `TILE_H x TILE_W` pixels
- Input tile (with halo): `(TILE_H + 2) x (TILE_W + 2)` pixels (1-pixel border for 3x3 kernel)
- Each thread block computes one output tile
- Block dimensions: `TILE_W x TILE_H` threads (one thread per output pixel)

**Recommended tile sizes:**
- **16x16 output tile** (18x18 input): 16x16 = 256 threads/block, 18x18x4 = 1296 bytes shared memory per channel — good occupancy on all targets.
- **32x32 output tile** (34x34 input): 1024 threads/block (maximum), 34x34x4 = 4624 bytes per channel — maximizes work per block but hits thread limit.
- For FP16: halve the shared memory requirements.

**Implicit GEMM approach (preferred for high channel counts):** Convert Conv2D to matrix multiplication via im2col transformation (done implicitly in registers, not materializing the im2col matrix):
- Reshape input patches into rows of a matrix
- Reshape filter weights into columns
- Tile the resulting GEMM in shared memory
- This approach dominates for channel counts > 64, which is the norm in diffusion UNets (320, 640, 1280 channels)

**Tile sizes for implicit GEMM Conv2D:**
- **128x128 tile with k=32 inner dimension:** Standard CUTLASS-style tiling. Each thread block loads a 128x32 tile of A and 32x128 tile of B into shared memory per iteration.
- **64x64 with k=32:** Lower shared memory pressure, better occupancy on CC 8.6/8.9 (100 KB limit).

Source: [eunomia - CNN Convolution with Shared Memory Optimization](https://eunomia.dev/others/cuda-tutorial/06-cnn-convolution/), [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass/discussions/281)

### 7. SDPA (Scaled Dot-Product Attention) Tiling

For flash-attention style SDPA in PTX, the key pattern is:
1. Tile Q into blocks of `Br` rows (e.g., Br=64)
2. Tile K and V into blocks of `Bc` rows (e.g., Bc=64)
3. For each Q tile, iterate over all K/V tiles
4. Accumulate softmax-weighted V using the online softmax rescaling trick

The shared memory layout for one iteration holds:
- Q tile: `Br x d` values (e.g., 64 x 64 = 4096 f16 values = 8 KB)
- K tile: `Bc x d` values (same)
- V tile: `Bc x d` values (same)
- S tile (QK^T partial): `Br x Bc` values (e.g., 64 x 64 = 4096 f32 values = 16 KB for FP32 accumulation)

Total shared memory per iteration: ~40 KB for d=64 in FP16 with FP32 accumulation — fits within the 48 KB default on all targets.

For d=128 (common in SDXL/Flux), either reduce tile sizes or opt into extended shared memory (up to 100 KB on CC 8.6/8.9, up to 164 KB on CC 8.0).

### 8. GroupNorm/LayerNorm Kernel Design

The normalization kernel pattern:
1. Each thread block handles one group (GroupNorm) or one token (LayerNorm)
2. Each thread loads multiple elements, accumulates partial sums in FP32
3. Warp shuffle reduces partials within each warp
4. Cross-warp reduction via shared memory (one value per warp, then single-warp final reduction)
5. Broadcast mean and variance to all threads via shared memory or shuffle
6. Each thread normalizes its elements and applies scale/bias

**PTX skeleton for warp-level mean reduction:**
```
// Phase 1: each thread accumulates partial sum in FP32
ld.global.v4.b32 {%r1, %r2, %r3, %r4}, [%rd_input];
// unpack f16x2 pairs, convert to f32, accumulate
mov.b32 {%h_lo, %h_hi}, %r1;
cvt.f32.f16 %f_a, %h_lo;
cvt.f32.f16 %f_b, %h_hi;
add.f32 %f_sum, %f_sum, %f_a;
add.f32 %f_sum, %f_sum, %f_b;
// ... repeat for %r2, %r3, %r4

// Phase 2: warp shuffle butterfly reduction
shfl.sync.bfly.b32 %f_tmp, %f_sum, 16, 0x1f, 0xFFFFFFFF;
add.f32 %f_sum, %f_sum, %f_tmp;
shfl.sync.bfly.b32 %f_tmp, %f_sum,  8, 0x1f, 0xFFFFFFFF;
add.f32 %f_sum, %f_sum, %f_tmp;
shfl.sync.bfly.b32 %f_tmp, %f_sum,  4, 0x1f, 0xFFFFFFFF;
add.f32 %f_sum, %f_sum, %f_tmp;
shfl.sync.bfly.b32 %f_tmp, %f_sum,  2, 0x1f, 0xFFFFFFFF;
add.f32 %f_sum, %f_sum, %f_tmp;
shfl.sync.bfly.b32 %f_tmp, %f_sum,  1, 0x1f, 0xFFFFFFFF;
add.f32 %f_sum, %f_sum, %f_tmp;
// %f_sum now holds warp-level sum in ALL lanes

// Phase 3: cross-warp reduction via shared memory
// lane 0 of each warp writes to shared memory
and.b32 %r_lane, %r_tid, 31;
setp.eq.u32 %p_lane0, %r_lane, 0;
@%p_lane0 st.shared.f32 [smem_warp_sums + warp_id*4], %f_sum;
bar.sync 0;
// ... single warp loads and reduces the per-warp sums
```

### 9. Register Pressure Management

**The occupancy equation:** Each SM has 65,536 (64K) 32-bit registers. The number of threads that can be resident simultaneously is limited by:

```
max_resident_threads = min(
    65536 / registers_per_thread,
    max_threads_per_SM,
    shared_memory_per_SM / shared_memory_per_block * threads_per_block
)

occupancy = max_resident_threads / max_threads_per_SM
```

**Register budget examples (targeting 50%+ occupancy):**

| CC | Max Threads/SM | 50% Target | Max Regs/Thread for 50% |
|----|---------------|------------|------------------------|
| 8.0 | 2048 | 1024 | 65536 / 1024 = 64 |
| 8.6 | 1536 | 768 | 65536 / 768 = 85 |
| 8.9 | 2048 | 1024 | 65536 / 1024 = 64 |

**Techniques to reduce register pressure:**
1. **Explicit `.maxnreg` directive:** `// .maxnreg 64` in the kernel constrains ptxas to use at most 64 registers per thread.
2. **Reuse registers:** PTX virtual registers are unlimited, but ptxas allocates physical registers based on live ranges. Minimizing the number of simultaneously-live values reduces pressure.
3. **Spill to local memory:** ptxas automatically spills excess registers to local memory (L1-cached), but this adds latency.
4. **Loop restructuring:** Process fewer elements per thread per iteration to reduce accumulator registers needed.
5. **Use `.f16x2` instead of two `.f32`:** Packing two halves into one register halves the register count for data registers.

Source: [NVIDIA Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html), [NVIDIA Ada Tuning Guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html), [Understanding PTX - NVIDIA Blog](https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing/)

---

## Key Numbers / Constants

### Compute Capability Comparison

| Specification | CC 8.0 (A100) | CC 8.6 (RTX 3080) | CC 8.9 (RTX 4090) |
|---|---|---|---|
| Warp size | 32 | 32 | 32 |
| Max threads per block | 1024 | 1024 | 1024 |
| Max threads per SM | 2048 | 1536 | 2048 |
| Max warps per SM | 64 | 48 | 64 |
| Max resident blocks per SM | 32 | 16 | 32 |
| 32-bit registers per SM | 65,536 | 65,536 | 65,536 |
| Max registers per thread | 255 | 255 | 255 |
| Shared memory per SM | 164 KB | 100 KB* | 100 KB* |
| Max shared memory per block | 163 KB | 99 KB | 99 KB |
| Unified L1/shared cache | 192 KB | 128 KB | 128 KB |
| L2 cache (flagship) | 40 MB (A100) | 6 MB (GA102) | 96 MB (AD102) |
| Shared memory banks | 32 | 32 | 32 |
| Bank width | 4 bytes | 4 bytes | 4 bytes |
| Number of SMs (flagship) | 108 (A100) | 84 (RTX 3090) | 128 (RTX 4090) |
| FP16 Tensor Core throughput | 312 TFLOPS (A100) | 142 TFLOPS (3090) | 330 TFLOPS (4090) |

*CC 8.6 unified cache is 128 KB. Shared memory carveout options: 0, 8, 16, 32, 64, 100 KB.
CC 8.0 (A100) carveout options: 0, 8, 16, 32, 64, 100, 132, 164 KB.
CC 8.9 (Ada) carveout options: 0, 8, 16, 32, 64, 100 KB.

**Default shared memory per block is 48 KB.** To use more, must call `cuFuncSetAttribute` with `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`. CUDA reserves 1 KB of shared memory per block for internal use.

### PTX ISA Version Mapping

| CUDA Toolkit | PTX ISA Version | Key Features |
|---|---|---|
| CUDA 11.0 | PTX 7.0 | sm_80 target, async copy (`cp.async`) |
| CUDA 11.1 | PTX 7.1 | sm_86 target |
| CUDA 11.8 | PTX 7.8 | sm_89 target (Ada) |
| CUDA 12.0 | PTX 8.0 | sm_90 target (Hopper) |
| CUDA 12.5 | PTX 8.5 | Additional features |
| CUDA 13.2 | PTX 9.2 | Latest stable |

Source: [PTX ISA 9.2 Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/), [CUDA Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)

---

## Data Layouts / Formats

### Tensor Layout in Memory

SharpInference uses **NCHW** layout (batch, channel, height, width) as the canonical format, matching PyTorch's default. However, for CUDA kernels:

- **Conv2D (implicit GEMM):** Operates on NHWC internally for coalesced memory access (consecutive threads access consecutive channels). The transpose NCHW->NHWC happens at the boundary.
- **GroupNorm:** Operates on NCHW (natural for channel-group slicing).
- **SDPA:** Layout is `(batch, heads, seq_len, head_dim)` — effectively NCHW where C=heads, H=seq_len, W=head_dim.

### FP16x2 Packing

When storing FP16 tensors, consecutive channel pairs are packed into `f16x2` words:
```
// Channel layout in memory (NCHW, FP16):
// address 0:  ch0_pixel0 (f16, 2 bytes)
// address 2:  ch1_pixel0 (f16, 2 bytes)
// -> packed as one f16x2 at address 0 (4 bytes)

// For vectorized 128-bit loads (8 f16 values = 4 f16x2 values):
ld.global.v4.b32 {%r0, %r1, %r2, %r3}, [%rd_ptr];
// %r0 = {ch0, ch1}, %r1 = {ch2, ch3}, %r2 = {ch4, ch5}, %r3 = {ch6, ch7}
```

### Shared Memory Tile Layout (Conv2D)

For a 16x16 output tile with 3x3 kernel:
```
// Input tile with halo: 18 x 18 pixels
// With padding for bank-conflict-free column access:
.shared .align 16 .b8 smem_input[18 * 19 * 2];   // 18 rows x 19 cols (padded) x 2 bytes (f16)
// = 684 bytes per input channel

// For implicit GEMM with NHWC:
// Tile A (activation): TILE_M x TILE_K f16 values
// Tile B (weights): TILE_K x TILE_N f16 values
// With padding:
.shared .align 16 .b8 smem_A[128 * 33 * 2];   // 128x33, padded from 128x32
.shared .align 16 .b8 smem_B[32 * 129 * 2];    // 32x129, padded from 32x128
```

---

## Algorithm Steps

### Conv2D Implicit GEMM (PTX)

1. **Compute thread/block indices:** Map `%ctaid.x/y/z` (block index) and `%tid.x/y` (thread index) to output tile coordinates.
2. **Initialize accumulators:** Zero FP32 accumulator registers for the output tile fragment this thread is responsible for.
3. **Loop over K tiles (input channels x kernel spatial positions):**
   a. Each thread loads its portion of the A tile (activation patch) from global memory to shared memory.
   b. Each thread loads its portion of the B tile (weight slice) from global memory to shared memory.
   c. `bar.sync 0;` — synchronize.
   d. Inner loop: multiply-accumulate from shared memory tiles into FP32 accumulators.
   e. `bar.sync 0;` — synchronize before overwriting shared memory.
4. **Convert accumulators from FP32 to FP16:** `cvt.rn.f16.f32` for each value.
5. **Pack and store:** Pack FP16 pairs into `f16x2`, write to global memory.

### GroupNorm + SiLU Fused Kernel (PTX)

1. **Compute group assignment:** Map block to (batch, group) pair.
2. **Phase 1 — Compute mean:** Each thread loads multiple channel elements from the group, accumulates sum in FP32 via warp shuffle + cross-warp shared memory reduction.
3. **Phase 2 — Compute variance:** Same reduction pattern on `(x - mean)^2`.
4. **Phase 3 — Normalize + SiLU:** Each thread reloads its elements, computes `y = (x - mean) * rsqrt(variance + epsilon) * gamma + beta`, then applies SiLU: `out = y * sigmoid(y)`. Uses `ex2.approx.f32` for the exponential in sigmoid.
5. **Store output:** Convert back to FP16, pack, store globally.

### Warp Reduction (used in GroupNorm/LayerNorm)

1. Each thread holds a partial value `v`.
2. 5 rounds of `shfl.sync.bfly.b32` with deltas 16, 8, 4, 2, 1.
3. After each shuffle, `add.f32` the received value to the local value.
4. After 5 rounds, all 32 threads hold the warp sum.
5. For cross-warp: lane 0 of each warp writes to shared memory, barrier, one warp reduces, broadcasts back.

---

## Reference Implementations

### External References

1. **NVIDIA PTX ISA Specification (latest: 9.2):**
   [https://docs.nvidia.com/cuda/parallel-thread-execution/](https://docs.nvidia.com/cuda/parallel-thread-execution/)
   — The definitive reference for all PTX instruction syntax, operand types, and target requirements.

2. **NVIDIA Ampere Tuning Guide:**
   [https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)
   — Architecture-specific optimization guidance for CC 8.0 and 8.6.

3. **NVIDIA Ada Tuning Guide:**
   [https://docs.nvidia.com/cuda/ada-tuning-guide/index.html](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html)
   — Architecture-specific optimization guidance for CC 8.9.

4. **NVIDIA Blog — Advanced CUDA Kernel Optimization: Handwritten PTX:**
   [https://developer.nvidia.com/blog/advanced-nvidia-cuda-kernel-optimization-techniques-handwritten-ptx/](https://developer.nvidia.com/blog/advanced-nvidia-cuda-kernel-optimization-techniques-handwritten-ptx/)
   — Shows 7-14% gains from hand-written PTX over CUDA C++ in a fused GEMM+top_k+softmax kernel.

5. **NVIDIA Blog — Using CUDA Warp-Level Primitives:**
   [https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)
   — Reference for `shfl.sync` patterns and warp-level collective operations.

6. **NVIDIA CUTLASS:**
   [https://github.com/NVIDIA/cutlass](https://github.com/NVIDIA/cutlass)
   — Template library for tiled GEMM. Algorithmic reference for tile sizes, software pipelining, and shared memory staging patterns.

7. **NVIDIA Blog — Understanding PTX:**
   [https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing/](https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing/)
   — Introduction to PTX concepts and its role in the CUDA compilation pipeline.

8. **Philip Fabianek — A Gentle Introduction to CUDA PTX:**
   [https://philipfabianek.com/posts/cuda-ptx-introduction/](https://philipfabianek.com/posts/cuda-ptx-introduction/)
   — Practical walkthrough of PTX register declarations, memory operations, and a complete kernel example.

9. **eunomia — CNN Convolution with Shared Memory Optimization:**
   [https://eunomia.dev/others/cuda-tutorial/06-cnn-convolution/](https://eunomia.dev/others/cuda-tutorial/06-cnn-convolution/)
   — Demonstrates tiled convolution with shared memory, showing 11x reduction in global memory traffic.

10. **CUDA Compute Capabilities Reference:**
    [https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
    — Authoritative table of per-CC hardware limits.

---

## Differences Between Implementations

### PTX vs CUDA C++

| Aspect | CUDA C++ | Hand-written PTX |
|--------|----------|-----------------|
| Performance | Baseline | +7-14% for hot kernels (measured on GEMM+softmax fusion) |
| Development time | Hours | Days-weeks |
| Portability | ptxas optimizes across CCs | Must tune per target (sm_80 vs sm_89) |
| Register control | Compiler decides | Full manual control over live ranges |
| Instruction selection | Compiler heuristics | Explicit instruction choice (e.g., `fma` vs separate `mul`+`add`) |
| Debugging | CUDA-GDB, printf | CUDA-GDB only, no printf, must read SASS |
| Maintenance | Standard C++ | Assembly-level, error-prone |

**Recommendation:** Write hot-path kernels (Conv2D GEMM, GroupNorm+SiLU, SDPA) in PTX. Use CUDA C++ or cuDNN for less critical operations.

### Ampere (CC 8.0) vs Ada Lovelace (CC 8.9)

| Aspect | CC 8.0 (A100) | CC 8.9 (RTX 4090) |
|--------|---------------|-------------------|
| Shared memory per SM | 164 KB | 100 KB |
| Max warps per SM | 64 | 64 |
| Max blocks per SM | 32 | 32 |
| L2 cache (flagship) | 40 MB | 96 MB (16x over GA102) |
| FP32 throughput | 1x | 2x per SM (dual FP32 datapaths) |
| Optimal GEMM tile | 128x128xk32 (fits in 164 KB) | 64x64xk32 or 128x64xk32 (fits in 100 KB) |
| Memory bandwidth | 1555 GB/s (HBM2e) | 1008 GB/s (GDDR6X) |
| Primary bottleneck | Compute-bound | Memory-bandwidth bound (L2 cache compensates) |

Ada's 16x larger L2 cache (vs Ampere consumer GA102) means that for typical diffusion inference tensor shapes, more data stays resident in L2, partially compensating for lower memory bandwidth. Applications optimized for Ampere should see speedups on Ada without code changes, but explicit `sm_89` targeting enables the 2x FP32 throughput.

### Tiled Convolution vs cuDNN

| Aspect | Custom PTX tiled Conv2D | cuDNN `cudnnConvolutionForward` |
|--------|------------------------|-------------------------------|
| Flexibility | Full control, can fuse with bias/activation | Limited fusion options |
| Setup overhead | None (kernel is pre-compiled) | Descriptor creation, algorithm selection, workspace allocation |
| Performance (3x3) | Competitive after tuning | Best for out-of-the-box performance |
| Performance (1x1) | Equivalent to GEMM | Uses GEMM internally |
| Code complexity | Very high | Low (P/Invoke wrapper) |

**Strategy:** Start with cuDNN, replace with custom PTX only where profiling shows it is the bottleneck or where fusion opportunities exist (e.g., Conv2D + bias + SiLU).

---

## Open Questions

- [x] Optimal tile sizes for Conv2D on Ampere vs Ada Lovelace — answered: 128x128xk32 for CC 8.0 (164 KB shared memory), 64x64xk32 for CC 8.6/8.9 (100 KB shared memory). Ada's larger L2 partially compensates.
- [x] PTX vs CUDA C++ performance gap — answered: 7-14% gain for hand-written PTX on hot-path fused kernels (NVIDIA blog benchmark). Diminishing returns for simple elementwise ops.
- [x] Shared memory size limits and occupancy tradeoffs — answered: see Key Numbers table above. Default is 48 KB/block; opt-in required for higher via `cuFuncSetAttribute`.
- [ ] Whether `cp.async` (Ampere) provides measurable benefit over explicit ld.global + st.shared for our tile sizes
- [ ] Optimal number of pipeline stages (double-buffering vs triple-buffering) for shared memory tiles in the Conv2D GEMM loop
- [ ] Whether to ship separate `.ptx` files per target (sm_80, sm_86, sm_89) or use a single sm_80 target with runtime JIT (sacrificing Ada 2x FP32)
- [ ] Tensor Core (WMMA / MMA) instructions in PTX for the implicit GEMM — would provide massive speedup but adds significant complexity to PTX code

---

## Implementation Notes

### Embedding PTX in SharpInference.Cuda

PTX source files should be embedded as assembly resources (`.resx` or `EmbeddedResource` in the `.csproj`):
```xml
<ItemGroup>
  <EmbeddedResource Include="Kernels\conv2d_implicit_gemm.ptx" />
  <EmbeddedResource Include="Kernels\groupnorm_silu.ptx" />
  <EmbeddedResource Include="Kernels\sdpa_flash.ptx" />
  <EmbeddedResource Include="Kernels\elementwise.ptx" />
</ItemGroup>
```

At runtime, load via `Assembly.GetManifestResourceStream`, then pass to `cuModuleLoadData`.

### Multi-Target Strategy

Option A (recommended): Ship PTX for `sm_80` as the baseline. The CUDA driver JIT-compiles PTX to SASS for the actual GPU at load time. This provides forward compatibility — `sm_80` PTX runs on all Ampere, Ada, and Hopper GPUs. Minor performance loss (~5%) from not targeting `sm_89` explicitly.

Option B: Ship multiple PTX files (`sm_80`, `sm_86`, `sm_89`). Select at runtime based on `cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR/MINOR)`. Better performance but larger assembly size and more maintenance.

### Register Limit Pragma

To enforce register limits and maintain occupancy, use `.maxnreg` in the PTX kernel:
```
.visible .entry groupnorm_silu(...)
.maxnreg 64
{
    ...
}
```
This tells ptxas to use at most 64 registers per thread, enabling 1024 threads per SM on CC 8.0/8.9 (65536/64 = 1024, which is 50% of max 2048). If the kernel genuinely needs more registers, ptxas will spill to local memory.

### Testing Strategy

1. **Correctness:** Compare PTX kernel output against CPU reference implementation (already in `SharpInference.Cpu`) with tolerance for FP16 rounding differences (typically 1e-3 absolute, 1e-2 relative).
2. **Performance:** Use `cuEventElapsedTime` around kernel launches. Compare against cuDNN for Conv2D, against PyTorch for GroupNorm/SDPA.
3. **Occupancy:** Use `cuOccupancyMaxActiveBlocksPerMultiprocessor` to verify expected occupancy matches actual.
4. **Bank conflicts:** Profile with `ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld` to verify conflict-free access patterns.
