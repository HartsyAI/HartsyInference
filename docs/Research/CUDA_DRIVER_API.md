# CUDA Driver API — Research Notes

> Status: Complete
> Last Updated: 2026-04-16
> Needed Before: SharpInference.Cuda

## Summary

SharpInference accesses CUDA GPUs entirely through the CUDA Driver API via P/Invoke — no CUDA Runtime API, no native shared libraries beyond the system-installed NVIDIA driver. This approach (proven by dotLLM) loads PTX kernels at runtime via `cuModuleLoadData`, JIT-compiles them for the target GPU, and launches them via `cuLaunchKernel`. cuBLAS is used for GEMM operations (FP16/FP32 matrix multiply).

All functions are exported from `nvcuda.dll` (Windows) / `libcuda.so` (Linux) for the Driver API, and `cublas64_12.dll` / `libcublas.so.12` for cuBLAS. All use `CallingConvention.Cdecl` and return status enums.

Sources: [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/), [cuBLAS](https://docs.nvidia.com/cuda/cublas/), [managedCuda](https://github.com/kunzmi/managedCuda), [swigged.cuda](https://github.com/kaby76/swigged.cuda)

## Detailed Findings

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

### Module/Kernel API (PTX Loading)

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

CUstream and cudaStream_t are interchangeable. Source: [managedCuda CudaBlasNativeMethods.cs](https://github.com/kunzmi/managedCuda/blob/master/CudaBlas/CudaBlasNativeMethods.cs)

## Key Numbers / Constants

### cublasOperation_t
| Value | Name |
|-------|------|
| 0 | CUBLAS_OP_N (no transpose) |
| 1 | CUBLAS_OP_T (transpose) |
| 2 | CUBLAS_OP_C (conjugate transpose) |

### cudaDataType
| Value | Name |
|-------|------|
| 0 | CUDA_R_32F (float) |
| 1 | CUDA_R_64F (double) |
| 2 | CUDA_R_16F (half) |
| 3 | CUDA_R_8I (int8) |
| 8 | CUDA_R_8U (uint8) |
| 14 | CUDA_R_16BF (bfloat16) |

### cublasComputeType_t
| Value | Name |
|-------|------|
| 64 | CUBLAS_COMPUTE_16F |
| 68 | CUBLAS_COMPUTE_32F |
| 74 | CUBLAS_COMPUTE_32F_FAST_16F (Tensor Cores) |
| 75 | CUBLAS_COMPUTE_32F_FAST_16BF (Tensor Cores) |
| 77 | CUBLAS_COMPUTE_32F_FAST_TF32 (Tensor Cores) |

### CUresult (Key Values)
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

## Data Layouts / Formats

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

## Algorithm Steps

### PTX Kernel Loading and Launch
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

### cuBLAS FP16 GEMM
```
1. cublasCreate(&handle)
2. cublasSetStream(handle, stream)
3. cublasHgemm(handle, OP_N, OP_N, M, N, K,
               &alpha, A, lda, B, ldb, &beta, C, ldc)
   // alpha=0x3C00 (1.0), beta=0x0000 (0.0)
4. cuStreamSynchronize(stream)
```

## Reference Implementations

| Implementation | Location | Notes |
|---------------|----------|-------|
| managedCuda | [GitHub](https://github.com/kunzmi/managedCuda) | Complete C# CUDA + cuBLAS bindings |
| swigged.cuda | [GitHub](https://github.com/kaby76/swigged.cuda) | Auto-generated, good for enum completeness |
| NVIDIA Driver API | [docs](https://docs.nvidia.com/cuda/cuda-driver-api/) | Canonical signatures |
| NVIDIA cuBLAS | [docs](https://docs.nvidia.com/cuda/cublas/) | GEMM APIs |
| cublas_api.h | [GitLab](https://gitlab.com/nvidia/headers/cuda-individual/cublas/-/blob/main/cublas_api.h) | All type definitions |

## Differences Between Implementations

| Aspect | managedCuda | SharpInference |
|--------|-------------|----------------|
| CUdeviceptr | Struct wrapping ulong | Struct (same) |
| Binding style | DllImport | LibraryImport (.NET 7+, faster) |
| DLL resolution | Hardcoded | NativeLibrary.SetDllImportResolver |
| Error handling | Exception wrappers | Check + throw on every call |

## Open Questions

- [x] ~~Minimum CUDA driver version~~ — CUDA 12.x >= 525; 13.x >= 580
- [x] ~~cuMemPool behavior~~ — Available CUDA 11.2+, stable
- [x] ~~cuBLAS workspace~~ — Managed internally by cuBLAS
- [ ] LibraryImport source-gen with ulong CUdeviceptr edge cases
- [ ] CUmodule caching strategy (cache and reuse expected best practice)

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
