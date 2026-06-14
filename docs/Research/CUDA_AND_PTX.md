# CUDA Driver API & PTX Kernels — Research Notes

HartsyInference accesses CUDA GPUs entirely through the CUDA Driver API via P/Invoke — no CUDA Runtime API, no native shared libraries beyond the system-installed NVIDIA driver. PTX kernels are loaded at runtime via `cuModuleLoadData`, JIT-compiled for the target GPU, and launched via `cuLaunchKernel`. cuBLAS is used for GEMM operations (FP16/FP32 matrix multiply).

Hand-written PTX provides a 7-14% performance improvement over CUDA C++ for critical-path kernels in diffusion inference (Conv2D, GroupNorm+SiLU, SDPA, elementwise add/scale), at the cost of per-architecture tuning and higher development complexity.

All Driver API functions are exported from `nvcuda.dll` (Windows) / `libcuda.so` (Linux). cuBLAS functions from `cublas64_12.dll` / `libcublas.so.12`. All use `CallingConvention.Cdecl` and return status enums.

Sources: [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/), [cuBLAS](https://docs.nvidia.com/cuda/cublas/), [PTX ISA 9.2](https://docs.nvidia.com/cuda/parallel-thread-execution/), [managedCuda](https://github.com/kunzmi/managedCuda), [swigged.cuda](https://github.com/kaby76/swigged.cuda)

---

## CUDA Driver API

### Initialization & Device Management

C signatures ([Device Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html), [Initialization](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__INITIALIZE.html)):

```c
CUresult cuInit(unsigned int Flags);
CUresult cuDeviceGet(CUdevice* device, int ordinal);
CUresult cuDeviceGetCount(int* count);
CUresult cuDeviceGetName(char* name, int len, CUdevice dev);
CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev);
CUresult cuDeviceTotalMem_v2(size_t* bytes, CUdevice dev);
```

C# P/Invoke ([managedCuda DriverAPI.cs](https://github.com/kunzmi/managedCuda/blob/master/ManagedCUDA/DriverAPI.cs)):

```csharp
// CUdevice = int, CUcontext/CUmodule/CUfunction/CUstream = IntPtr
// CUdeviceptr = ulong (always 64-bit)

[DllImport("nvcuda", CallingConvention = CallingConvention.Cdecl)]
public static extern CUresult cuInit(uint Flags);

[DllImport("nvcuda", CallingConvention = CallingConvention.Cdecl)]
public static extern CUresult cuDeviceGet(out int device, int ordinal);

[DllImport("nvcuda", CallingConvention = CallingConvention.Cdecl)]
public static extern CUresult cuDeviceGetCount(out int count);

[DllImport("nvcuda", CallingConvention = CallingConvention.Cdecl)]
public static extern CUresult cuDeviceGetAttribute(out int pi, int attrib, int dev);

[DllImport("nvcuda", CallingConvention = CallingConvention.Cdecl)]
public static extern CUresult cuDeviceTotalMem_v2(out nuint bytes, int dev);
```

**Key attribute IDs**: `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75`, `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76`.

### Context Management

Source: [Context Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html)

```csharp
[DllImport("nvcuda", EntryPoint = "cuCtxCreate_v2")]
public static extern CUresult cuCtxCreate(out IntPtr pctx, uint flags, int dev);

[DllImport("nvcuda", EntryPoint = "cuCtxDestroy_v2")]
public static extern CUresult cuCtxDestroy(IntPtr ctx);

[DllImport("nvcuda")]
public static extern CUresult cuCtxSetCurrent(IntPtr ctx);

[DllImport("nvcuda")]
public static extern CUresult cuCtxGetCurrent(out IntPtr pctx);

[DllImport("nvcuda")]
public static extern CUresult cuCtxSynchronize();
```

CUDA 13.0+ introduced cuCtxCreate v4, but v2 remains available. Target v2 for compatibility.

### Module/Kernel API (PTX Loading & Launch)

Source: [Module Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html), [Execution Control](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html)

Pass a null-terminated PTX string as `const void* image`. The driver JIT-compiles it. In C#, pass as `byte[]` (UTF-8 + null terminator).

```csharp
[DllImport("nvcuda")]
public static extern CUresult cuModuleLoadData(out IntPtr module, byte[] image);

[DllImport("nvcuda")]
public static extern CUresult cuModuleLoadDataEx(out IntPtr module, byte[] image,
    uint numOptions, int[] options, IntPtr[] optionValues);

[DllImport("nvcuda")]
public static extern CUresult cuModuleGetFunction(out IntPtr hfunc, IntPtr hmod, string name);

[DllImport("nvcuda")]
public static extern CUresult cuLaunchKernel(IntPtr f,
    uint gridDimX, uint gridDimY, uint gridDimZ,
    uint blockDimX, uint blockDimY, uint blockDimZ,
    uint sharedMemBytes, IntPtr hStream,
    IntPtr[] kernelParams, IntPtr[] extra);
```

#### CUjit_option Enum

| Value | Name | Description |
|-------|------|-------------|
| 0 | CU_JIT_MAX_REGISTERS | Max registers per thread |
| 7 | CU_JIT_OPTIMIZATION_LEVEL | 0-4, default 4 |
| 9 | CU_JIT_TARGET | Target compute capability |
| 16 | CU_JIT_FAST_COMPILE | Fast compilation mode |
| 21 | CU_JIT_FTZ | Flush to zero |
| 24 | CU_JIT_FMA | Fused multiply-add |

#### PTX Kernel Loading and Launch Flow

```
1. cuInit(0)
2. cuDeviceGet(&device, 0)
3. cuCtxCreate(&ctx, 0, device)
4. Load PTX from embedded resource -> UTF-8 byte[] with null terminator
5. cuModuleLoadData(&module, ptxBytes)     // JIT compile
6. cuModuleGetFunction(&func, module, "kernel_name")
7. Pin kernel params on stack, create IntPtr[] pointing to them
8. cuLaunchKernel(func, gX, gY, gZ, bX, bY, bZ, sharedMem, stream, params, null)
9. cuStreamSynchronize(stream)
```

### Memory API

Source: [Memory Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html)

`CUdeviceptr` is `typedef unsigned long long` — always 64-bit.

```csharp
[DllImport("nvcuda", EntryPoint = "cuMemAlloc_v2")]
public static extern CUresult cuMemAlloc(out ulong dptr, nuint bytesize);

[DllImport("nvcuda", EntryPoint = "cuMemFree_v2")]
public static extern CUresult cuMemFree(ulong dptr);

[DllImport("nvcuda", EntryPoint = "cuMemcpyHtoD_v2")]
public static extern CUresult cuMemcpyHtoD(ulong dst, IntPtr src, nuint bytes);

[DllImport("nvcuda", EntryPoint = "cuMemcpyDtoH_v2")]
public static extern CUresult cuMemcpyDtoH(IntPtr dst, ulong src, nuint bytes);

[DllImport("nvcuda", EntryPoint = "cuMemcpyDtoD_v2")]
public static extern CUresult cuMemcpyDtoD(ulong dst, ulong src, nuint bytes);

// Async (CUDA 11.2+)
[DllImport("nvcuda")]
public static extern CUresult cuMemAllocAsync(out ulong dptr, nuint bytes, IntPtr stream);

[DllImport("nvcuda")]
public static extern CUresult cuMemFreeAsync(ulong dptr, IntPtr stream);
```

### Stream API

Source: [Stream Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html)

```csharp
[DllImport("nvcuda")]
public static extern CUresult cuStreamCreate(out IntPtr stream, uint Flags);

[DllImport("nvcuda")]
public static extern CUresult cuStreamDestroy(IntPtr stream);

[DllImport("nvcuda")]
public static extern CUresult cuStreamSynchronize(IntPtr stream);
```

### Error Handling

Source: [Error Handling](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__ERROR.html)

```csharp
[DllImport("nvcuda")]
public static extern CUresult cuGetErrorName(CUresult error, out IntPtr pStr);

[DllImport("nvcuda")]
public static extern CUresult cuGetErrorString(CUresult error, out IntPtr pStr);
// Usage: Marshal.PtrToStringAnsi(pStr)
```

### cuBLAS API

Separate library. Source: [cuBLAS docs](https://docs.nvidia.com/cuda/cublas/), [cublas_api.h](https://gitlab.com/nvidia/headers/cuda-individual/cublas/-/blob/main/cublas_api.h)

```csharp
// DLL: "cublas64_12" (Windows), "libcublas" (Linux)

[DllImport("cublas64_12", EntryPoint = "cublasCreate_v2")]
public static extern CublasStatus cublasCreate(out IntPtr handle);

[DllImport("cublas64_12", EntryPoint = "cublasDestroy_v2")]
public static extern CublasStatus cublasDestroy(IntPtr handle);

[DllImport("cublas64_12", EntryPoint = "cublasSetStream_v2")]
public static extern CublasStatus cublasSetStream(IntPtr handle, IntPtr stream);

// FP32 GEMM
[DllImport("cublas64_12", EntryPoint = "cublasSgemm_v2")]
public static extern CublasStatus cublasSgemm(IntPtr handle,
    int transa, int transb, int m, int n, int k,
    ref float alpha, ulong A, int lda, ulong B, int ldb,
    ref float beta, ulong C, int ldc);

// FP16 GEMM — alpha/beta as ushort: 1.0 = 0x3C00, 0.0 = 0x0000
[DllImport("cublas64_12")]
public static extern CublasStatus cublasHgemm(IntPtr handle,
    int transa, int transb, int m, int n, int k,
    ref ushort alpha, ulong A, int lda, ulong B, int ldb,
    ref ushort beta, ulong C, int ldc);

// Mixed-precision GEMM
[DllImport("cublas64_12")]
public static extern CublasStatus cublasGemmEx(IntPtr handle,
    int transa, int transb, int m, int n, int k,
    IntPtr alpha, ulong A, int Atype, int lda,
    ulong B, int Btype, int ldb,
    IntPtr beta, ulong C, int Ctype, int ldc,
    int computeType, int algo);
```

CUstream and cudaStream_t are interchangeable.

#### cuBLAS FP16 GEMM Flow

```
1. cublasCreate(&handle)
2. cublasSetStream(handle, stream)
3. cublasHgemm(handle, OP_N, OP_N, M, N, K,
               &alpha, A, lda, B, ldb, &beta, C, ldc)
   // alpha=0x3C00 (1.0), beta=0x0000 (0.0)
4. cuStreamSynchronize(stream)
```

### Handle Types

| C Type | C# Type | Description |
|--------|---------|-------------|
| CUdevice | `int` | Device ordinal |
| CUcontext | `IntPtr` | Context handle |
| CUmodule | `IntPtr` | Module (loaded PTX) |
| CUfunction | `IntPtr` | Kernel function |
| CUstream | `IntPtr` | Stream handle |
| CUdeviceptr | `ulong` | Device pointer (always 64-bit) |
| cublasHandle_t | `IntPtr` | cuBLAS handle |

### CUdeviceptr Struct

```csharp
[StructLayout(LayoutKind.Sequential)]
public struct CUdeviceptr : IEquatable<CUdeviceptr>
{
    public ulong Pointer;
    public static CUdeviceptr operator +(CUdeviceptr ptr, ulong offset)
        => new() { Pointer = ptr.Pointer + offset };
    public bool Equals(CUdeviceptr other) => Pointer == other.Pointer;
    public override int GetHashCode() => Pointer.GetHashCode();
}
```

### Enum Constants

#### cublasOperation_t

| Value | Name |
|-------|------|
| 0 | CUBLAS_OP_N (no transpose) |
| 1 | CUBLAS_OP_T (transpose) |
| 2 | CUBLAS_OP_C (conjugate transpose) |

#### cudaDataType

| Value | Name |
|-------|------|
| 0 | CUDA_R_32F (float) |
| 1 | CUDA_R_64F (double) |
| 2 | CUDA_R_16F (half) |
| 3 | CUDA_R_8I (int8) |
| 8 | CUDA_R_8U (uint8) |
| 14 | CUDA_R_16BF (bfloat16) |

#### cublasComputeType_t

| Value | Name |
|-------|------|
| 64 | CUBLAS_COMPUTE_16F |
| 68 | CUBLAS_COMPUTE_32F |
| 74 | CUBLAS_COMPUTE_32F_FAST_16F (Tensor Cores) |
| 75 | CUBLAS_COMPUTE_32F_FAST_16BF (Tensor Cores) |
| 77 | CUBLAS_COMPUTE_32F_FAST_TF32 (Tensor Cores) |

#### CUresult (Key Values)

| Value | Name |
|-------|------|
| 0 | CUDA_SUCCESS |
| 1 | CUDA_ERROR_INVALID_VALUE |
| 2 | CUDA_ERROR_OUT_OF_MEMORY |
| 3 | CUDA_ERROR_NOT_INITIALIZED |
| 100 | CUDA_ERROR_NO_DEVICE |
| 200 | CUDA_ERROR_INVALID_IMAGE |
| 218 | CUDA_ERROR_INVALID_PTX |
| 221 | CUDA_ERROR_JIT_COMPILER_NOT_FOUND |
| 700 | CUDA_ERROR_ILLEGAL_ADDRESS |
| 701 | CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES |
| 719 | CUDA_ERROR_LAUNCH_FAILED |
| 999 | CUDA_ERROR_UNKNOWN |

Full enum: [swigged.cuda CUresult.cs](https://github.com/kaby76/swigged.cuda/blob/master/swigged.cuda/CUresult.cs)

### Minimum Driver Versions

| CUDA Toolkit | Min Driver | Architecture |
|---|---|---|
| CUDA 11.x | >= 450 | Maxwell-Hopper |
| CUDA 12.x | >= 525 | Maxwell-Blackwell |
| CUDA 13.0 | >= 580.65 | Turing+ only (SM 7.5+) |

Source: [CUDA Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/), [Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/)

### Differences: managedCuda vs HartsyInference

| Aspect | managedCuda | HartsyInference |
|--------|-------------|----------------|
| CUdeviceptr | Struct wrapping ulong | Struct (same) |
| Binding style | DllImport | LibraryImport (.NET 7+, faster) |
| DLL resolution | Hardcoded | NativeLibrary.SetDllImportResolver |
| Error handling | Exception wrappers | Check + throw on every call |

---

## PTX Kernels

### PTX ISA Basics

PTX is a virtual ISA — it uses unlimited virtual registers that `ptxas` maps to physical hardware registers during JIT compilation. A PTX file targets a specific virtual architecture (`sm_80`, `sm_86`, `sm_89`) and a PTX ISA version (e.g., `8.5` for CUDA 12.5, `9.2` for CUDA 13.2).

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

### FP16 Packed Arithmetic Intrinsics

The `.f16x2` type packs two half-precision floats into a single 32-bit register, doubling throughput for elementwise operations. Requires `sm_53` or higher.

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

**Rounding modes:** `.rn` (nearest even, default), `.rz` (toward zero), `.rm` (toward -inf), `.rp` (toward +inf).

**Conversion instructions (critical for FP32 accumulation):**
```
// FP16 -> FP32 (widen for accumulation)
cvt.f32.f16   %f1, %h1;

// FP32 -> FP16 (narrow after accumulation)
cvt.rn.f16.f32 %h1, %f1;

// Pack two f16 values into f16x2
mov.b32       %h_pair, {%h_lo, %h_hi};

// Unpack f16x2 into two f16 values
mov.b32       {%h_lo, %h_hi}, %h_pair;
```

**Important:** For GroupNorm/LayerNorm, mean and variance must be accumulated in FP32 to avoid precision loss. Pattern: load as f16x2, unpack, convert to f32, accumulate in f32, convert result back to f16, repack to f16x2, store.

### Warp Shuffle Instructions (`shfl.sync`)

Warp shuffles exchange register values between threads within a warp (32 threads) without shared memory, ideal for fast reductions in normalization kernels. Requires PTX ISA 6.0+ (CUDA 9.0+). The non-sync variant `shfl` is deprecated.

**Instruction syntax:**
```
shfl.sync.mode.b32 d[|p], a, b, c, membermask;
```

**Modes:**

| Mode | Source Lane Calculation | Primary Use |
|------|----------------------|-------------|
| `.idx` | `srcLane = b` | Broadcast from specific lane |
| `.up` | `srcLane = laneId - b` | Prefix scan (exclusive) |
| `.down` | `srcLane = laneId + b` | Tree reduction (sum down) |
| `.bfly` | `srcLane = laneId XOR b` | Butterfly reduction |

**Warp reduction pattern for sum (used in GroupNorm/LayerNorm):**
```
// Butterfly reduction: log2(32) = 5 steps
// All lanes hold the final result (preferred over .down where only lane 0 gets result)
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

### Shared Memory Tiling

Shared memory (`.shared` state space in PTX) is on-chip SRAM, orders of magnitude faster than global memory. Tiling loads input data from global memory into shared memory, synchronizes, then threads compute from the tile.

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

// Tiling pattern: global -> register -> shared
ld.global.f32      %f1, [%rd_global_ptr];
st.shared.f32      [smem_A + offset], %f1;
bar.sync 0;   // synchronize all threads in block before reading shared
```

**Synchronization:** `bar.sync 0;` — barrier index 0-15 available.

### Bank Conflict-Free Access Patterns

Shared memory: 32 banks, 4 bytes wide each. Bank mapping: `bank_index = (byte_address / 4) % 32`.

**Conflict-free patterns:**
- **Stride-1 access:** Thread `i` reads word `i` — each thread hits a different bank.
- **Broadcast:** Multiple threads read the *same* address — no conflict.
- **Padding trick:** Add 1 padding element per row to shift bank assignments for column-major/transposed access:

```
// Without padding: column access causes 32-way bank conflict
.shared .align 16 .b8 smem_tile[32 * 32 * 4];       // 32x32 f32 tile

// With padding: column access is conflict-free
.shared .align 16 .b8 smem_tile[32 * 33 * 4];        // 32x33 f32 tile (extra column)
```

For `.f16x2` (4 bytes wide), the same bank-conflict analysis applies since packed pairs naturally align to 4-byte bank boundaries.

**Asynchronous copy (CC 8.0+):** `cp.async` bypasses registers for global-to-shared copy:
```
cp.async.ca.shared.global [smem_addr], [glob_addr], 16;   // copy 16 bytes async
cp.async.commit_group;
cp.async.wait_group 0;
```

### Tiled Conv2D Kernel Design

**Direct tiling for 3x3 convolution:**
- Output tile: `TILE_H x TILE_W` pixels; input tile with halo: `(TILE_H + 2) x (TILE_W + 2)`
- **16x16 output** (18x18 input): 256 threads/block, 1296 bytes shared per channel — good occupancy
- **32x32 output** (34x34 input): 1024 threads/block (max), 4624 bytes per channel

**Implicit GEMM approach (preferred for channel counts > 64):** Convert Conv2D to matrix multiplication via im2col done implicitly in registers. Dominates for diffusion UNet channel counts (320, 640, 1280).

**Implicit GEMM tile sizes:**
- **128x128 with k=32:** Standard CUTLASS-style. Fits in 164 KB shared memory (CC 8.0).
- **64x64 with k=32:** Lower shared memory pressure, better occupancy on CC 8.6/8.9 (100 KB limit).

**Conv2D Implicit GEMM algorithm:**
1. Compute thread/block indices: map `%ctaid.x/y/z` and `%tid.x/y` to output tile coordinates.
2. Initialize FP32 accumulator registers to zero.
3. Loop over K tiles (input channels x kernel spatial positions):
   a. Load A tile (activation patch) and B tile (weight slice) from global to shared memory.
   b. `bar.sync 0;`
   c. Multiply-accumulate from shared memory into FP32 accumulators.
   d. `bar.sync 0;` before overwriting shared memory.
4. Convert accumulators FP32 -> FP16 via `cvt.rn.f16.f32`.
5. Pack into `f16x2`, write to global memory.

### SDPA (Scaled Dot-Product Attention) Tiling

Flash-attention style SDPA in PTX:
1. Tile Q into blocks of `Br` rows (e.g., Br=64)
2. Tile K and V into blocks of `Bc` rows (e.g., Bc=64)
3. For each Q tile, iterate over all K/V tiles
4. Accumulate softmax-weighted V using the online softmax rescaling trick

Shared memory layout per iteration:
- Q tile: `Br x d` (e.g., 64x64 = 8 KB in FP16)
- K tile: `Bc x d` (same)
- V tile: `Bc x d` (same)
- S tile (QK^T partial): `Br x Bc` (e.g., 64x64 = 16 KB in FP32 for accumulation)

Total: ~40 KB for d=64 in FP16 with FP32 accumulation — fits within 48 KB default. For d=128 (SDXL/Flux), reduce tile sizes or opt into extended shared memory (up to 100 KB on CC 8.6/8.9, up to 164 KB on CC 8.0).

### GroupNorm/LayerNorm Kernel Design

1. Each thread block handles one group (GroupNorm) or one token (LayerNorm).
2. Each thread loads multiple elements, accumulates partial sums in FP32.
3. Warp shuffle reduces partials within each warp.
4. Cross-warp reduction via shared memory (one value per warp, then single-warp final reduction).
5. Broadcast mean and variance to all threads.
6. Each thread normalizes its elements and applies scale/bias.

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

// Phase 2: warp shuffle butterfly reduction (5 steps)
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

// Phase 3: cross-warp reduction via shared memory
and.b32 %r_lane, %r_tid, 31;
setp.eq.u32 %p_lane0, %r_lane, 0;
@%p_lane0 st.shared.f32 [smem_warp_sums + warp_id*4], %f_sum;
bar.sync 0;
// ... single warp loads and reduces the per-warp sums
```

**GroupNorm + SiLU fused kernel algorithm:**
1. Map block to (batch, group) pair.
2. Compute mean via warp shuffle + cross-warp shared memory reduction.
3. Compute variance via same reduction on `(x - mean)^2`.
4. Normalize + SiLU: `y = (x - mean) * rsqrt(variance + epsilon) * gamma + beta`, then `out = y * sigmoid(y)` using `ex2.approx.f32` for the exponential.
5. Convert back to FP16, pack, store.

### Register Pressure Management

**The occupancy equation:** Each SM has 65,536 (64K) 32-bit registers.

```
max_resident_threads = min(
    65536 / registers_per_thread,
    max_threads_per_SM,
    shared_memory_per_SM / shared_memory_per_block * threads_per_block
)
occupancy = max_resident_threads / max_threads_per_SM
```

**Register budget (targeting 50%+ occupancy):**

| CC | Max Threads/SM | 50% Target | Max Regs/Thread for 50% |
|----|---------------|------------|------------------------|
| 8.0 | 2048 | 1024 | 64 |
| 8.6 | 1536 | 768 | 85 |
| 8.9 | 2048 | 1024 | 64 |

**Techniques to reduce register pressure:**
1. `.maxnreg` directive (e.g., `.maxnreg 64`) constrains ptxas.
2. Minimize simultaneously-live values to help ptxas allocate physical registers.
3. Spill to local memory (L1-cached) happens automatically but adds latency.
4. Loop restructuring: fewer elements per thread per iteration.
5. Use `.f16x2` instead of two `.f32` to halve data register count.

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
| Number of SMs (flagship) | 108 (A100) | 84 (RTX 3090) | 128 (RTX 4090) |
| FP16 Tensor Core throughput | 312 TFLOPS (A100) | 142 TFLOPS (3090) | 330 TFLOPS (4090) |

*CC 8.6 shared memory carveout options: 0, 8, 16, 32, 64, 100 KB. CC 8.0 (A100): 0, 8, 16, 32, 64, 100, 132, 164 KB. CC 8.9 (Ada): 0, 8, 16, 32, 64, 100 KB.

**Default shared memory per block is 48 KB.** To use more, call `cuFuncSetAttribute` with `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`. CUDA reserves 1 KB per block for internal use.

### PTX ISA Version Mapping

| CUDA Toolkit | PTX ISA Version | Key Features |
|---|---|---|
| CUDA 11.0 | PTX 7.0 | sm_80 target, async copy (`cp.async`) |
| CUDA 11.1 | PTX 7.1 | sm_86 target |
| CUDA 11.8 | PTX 7.8 | sm_89 target (Ada) |
| CUDA 12.0 | PTX 8.0 | sm_90 target (Hopper) |
| CUDA 12.5 | PTX 8.5 | Additional features |
| CUDA 13.2 | PTX 9.2 | Latest stable |

### Tensor Layout in Memory

HartsyInference uses **NCHW** layout (batch, channel, height, width) as canonical, matching PyTorch's default. For CUDA kernels:
- **Conv2D (implicit GEMM):** NHWC internally for coalesced access. Transpose NCHW->NHWC at the boundary.
- **GroupNorm:** NCHW (natural for channel-group slicing).
- **SDPA:** `(batch, heads, seq_len, head_dim)` — effectively NCHW where C=heads, H=seq_len, W=head_dim.

### FP16x2 Packing

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

```
// Input tile with halo: 18 x 18 pixels
// With padding for bank-conflict-free column access:
.shared .align 16 .b8 smem_input[18 * 19 * 2];   // 18 rows x 19 cols (padded) x 2 bytes (f16)

// For implicit GEMM with NHWC:
.shared .align 16 .b8 smem_A[128 * 33 * 2];   // 128x33, padded from 128x32
.shared .align 16 .b8 smem_B[32 * 129 * 2];    // 32x129, padded from 32x128
```

### Differences: PTX vs CUDA C++

| Aspect | CUDA C++ | Hand-written PTX |
|--------|----------|-----------------|
| Performance | Baseline | +7-14% for hot kernels |
| Development time | Hours | Days-weeks |
| Portability | ptxas optimizes across CCs | Must tune per target |
| Register control | Compiler decides | Full manual control over live ranges |
| Instruction selection | Compiler heuristics | Explicit choice (e.g., `fma` vs `mul`+`add`) |
| Debugging | CUDA-GDB, printf | CUDA-GDB only, no printf, must read SASS |

### Differences: Ampere (CC 8.0) vs Ada Lovelace (CC 8.9)

| Aspect | CC 8.0 (A100) | CC 8.9 (RTX 4090) |
|--------|---------------|-------------------|
| Shared memory per SM | 164 KB | 100 KB |
| L2 cache (flagship) | 40 MB | 96 MB (16x over GA102) |
| FP32 throughput | 1x | 2x per SM (dual FP32 datapaths) |
| Optimal GEMM tile | 128x128xk32 | 64x64xk32 or 128x64xk32 |
| Memory bandwidth | 1555 GB/s (HBM2e) | 1008 GB/s (GDDR6X) |
| Primary bottleneck | Compute-bound | Memory-bandwidth bound (L2 compensates) |

### Differences: Tiled Convolution vs cuDNN

| Aspect | Custom PTX tiled Conv2D | cuDNN `cudnnConvolutionForward` |
|--------|------------------------|-------------------------------|
| Flexibility | Full control, can fuse with bias/activation | Limited fusion options |
| Setup overhead | None (kernel is pre-compiled) | Descriptor creation, algo selection, workspace allocation |
| Performance (3x3) | Competitive after tuning | Best out-of-the-box |
| Code complexity | Very high | Low (P/Invoke wrapper) |

**Strategy:** Start with cuDNN, replace with custom PTX only where profiling shows bottleneck or fusion opportunities exist (e.g., Conv2D + bias + SiLU).

---

## Open Questions

- [ ] LibraryImport source-gen with ulong CUdeviceptr edge cases
- [ ] CUmodule caching strategy (cache and reuse expected best practice)
- [ ] Whether `cp.async` (Ampere) provides measurable benefit over explicit ld.global + st.shared for our tile sizes
- [ ] Optimal number of pipeline stages (double-buffering vs triple-buffering) for shared memory tiles in the Conv2D GEMM loop
- [ ] Whether to ship separate `.ptx` files per target (sm_80, sm_86, sm_89) or use a single sm_80 target with runtime JIT (sacrificing Ada 2x FP32)
- [ ] Tensor Core (WMMA / MMA) instructions in PTX for the implicit GEMM — massive speedup but significant complexity

---

## Implementation Notes

1. **LibraryImport over DllImport** — .NET 7+ source-gen is faster. [MS docs](https://learn.microsoft.com/en-us/dotnet/standard/native-interop/tutorial-custom-marshaller)
2. **Cross-platform DLL resolution** — `NativeLibrary.SetDllImportResolver` to map nvcuda->libcuda.so.1, cublas64_12->libcublas.so.12
3. **CUdeviceptr struct** — type safety, prevents host/device pointer confusion
4. **PTX caching** — `Encoding.UTF8.GetBytes(ptx + "\0")`, cache CUmodule handles (JIT ~100ms+)
5. **Error checking** — check EVERY call. CudaCheck() helper that throws. Silent corruption is the #1 issue.
6. **FP16 constants** — 1.0f16 = 0x3C00, 0.0f16 = 0x0000 as ushort
7. **Stream-ordered memory** — prefer cuMemAllocAsync/cuMemFreeAsync on hot paths
8. **Kernel params** — pin on stack, create IntPtr[] pointing to values
9. **Compute targeting** — CU_JIT_TARGET or omit for current device
10. **Thread safety** — cuCtxSetCurrent per thread, or primary context (cuDevicePrimaryCtxRetain)
11. **Embedding PTX** — Use `EmbeddedResource` in `.csproj`, load via `Assembly.GetManifestResourceStream`:
    ```xml
    <ItemGroup>
      <EmbeddedResource Include="Kernels\conv2d_implicit_gemm.ptx" />
      <EmbeddedResource Include="Kernels\groupnorm_silu.ptx" />
      <EmbeddedResource Include="Kernels\sdpa_flash.ptx" />
      <EmbeddedResource Include="Kernels\elementwise.ptx" />
    </ItemGroup>
    ```
12. **Multi-target strategy** — Ship PTX for `sm_80` as baseline (forward compatible with Ampere, Ada, Hopper, ~5% perf loss vs explicit targeting). Optionally ship `sm_86`/`sm_89` variants selected at runtime via `cuDeviceGetAttribute`.
13. **Register limit** — Use `.maxnreg 64` in kernel to enforce occupancy targets. ptxas spills to local memory if needed.
14. **Testing** — Correctness: compare against CPU reference (1e-3 abs, 1e-2 rel tolerance). Performance: `cuEventElapsedTime`. Occupancy: `cuOccupancyMaxActiveBlocksPerMultiprocessor`. Bank conflicts: `ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld`.

## References

- [NVIDIA CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/) — canonical P/Invoke signatures
- [NVIDIA cuBLAS](https://docs.nvidia.com/cuda/cublas/) — GEMM APIs
- [NVIDIA PTX ISA 9.2](https://docs.nvidia.com/cuda/parallel-thread-execution/) — definitive PTX instruction reference
- [NVIDIA Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html) — CC 8.0/8.6 optimization
- [NVIDIA Ada Tuning Guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html) — CC 8.9 optimization
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) — tiled GEMM algorithmic reference
- [NVIDIA Blog: Advanced CUDA Kernel Optimization (Handwritten PTX)](https://developer.nvidia.com/blog/advanced-nvidia-cuda-kernel-optimization-techniques-handwritten-ptx/) — 7-14% PTX gains
- [NVIDIA Blog: Using CUDA Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/) — shfl.sync patterns
- [NVIDIA Blog: Understanding PTX](https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing/) — PTX intro
- [managedCuda](https://github.com/kunzmi/managedCuda) — complete C# CUDA + cuBLAS bindings
- [swigged.cuda](https://github.com/kaby76/swigged.cuda) — auto-generated bindings, good for enum completeness
- [cublas_api.h](https://gitlab.com/nvidia/headers/cuda-individual/cublas/-/blob/main/cublas_api.h) — all type definitions
- [CUDA Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html) — per-CC hardware limits
- [Philip Fabianek: A Gentle Introduction to CUDA PTX](https://philipfabianek.com/posts/cuda-ptx-introduction/) — practical walkthrough
- [eunomia: CNN Convolution with Shared Memory](https://eunomia.dev/others/cuda-tutorial/06-cnn-convolution/) — tiled convolution example
- [Lei Mao: CUDA Shared Memory Bank Conflict-Free Access](https://leimao.github.io/blog/CUDA-Shared-Memory-Bank-Conflict-Free-Vectorized-Access/)
