# Conv2D CUDA — Research Notes

> Status: Complete
> Last Updated: 2026-04-16
> Needed Before: SharpInference.Cuda

## Summary

Conv2D is the most frequently executed operation in diffusion UNets (Stable Diffusion 1.5 has ~70 Conv2D layers across its encoder, middle block, and decoder). The initial strategy for SharpInference is to use cuDNN via P/Invoke for correctness and competitive performance, with the option to later replace hot-path convolutions with custom PTX kernels. cuDNN provides algorithm auto-selection (Winograd for 3x3, implicit GEMM for 1x1), automatic Tensor Core utilization on Ampere+ when NHWC format is used, and workspace-managed execution that abstracts the complexity of tiled convolution.

All cuDNN functions are in split DLLs starting with cuDNN 8+: `cudnn64_9.dll` (core), `cudnn_cnn_infer64_9.dll` (CNN inference), `cudnn_ops64_9.dll` (ops) on Windows; `libcudnn.so.9`, `libcudnn_cnn_infer.so.9`, `libcudnn_ops.so.9` on Linux. The legacy API (`cudnnConvolutionForward`) is deprecated as of cuDNN 9.x in favor of the Graph API but remains functional and is the simplest path for initial implementation. All functions return `cudnnStatus_t` and use `CallingConvention.Cdecl`.

Sources: [cuDNN API Reference (8.9.2)](https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-892/api/index.html), [cuDNN CNN Library](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-cnn-library.html), [cuDNN Support Matrix](https://docs.nvidia.com/deeplearning/cudnn/latest/reference/support-matrix.html), [NVIDIA Conv Performance Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html), [cuDNN Legacy API Guide](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/developer/legacy-api.html)

## Detailed Findings

### cuDNN Handle and Stream Management

C signatures ([cuDNN Ops Library](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-ops-library.html)):

```c
cudnnStatus_t cudnnCreate(cudnnHandle_t *handle);
cudnnStatus_t cudnnDestroy(cudnnHandle_t handle);
cudnnStatus_t cudnnSetStream(cudnnHandle_t handle, cudaStream_t streamId);
```

C# P/Invoke:

```csharp
// cudnnHandle_t = IntPtr
// DLL: "cudnn64_9" (Windows), "libcudnn" (Linux)

[DllImport("cudnn64_9", CallingConvention = CallingConvention.Cdecl)]
public static extern CudnnStatus cudnnCreate(out IntPtr handle);

[DllImport("cudnn64_9", CallingConvention = CallingConvention.Cdecl)]
public static extern CudnnStatus cudnnDestroy(IntPtr handle);

[DllImport("cudnn64_9", CallingConvention = CallingConvention.Cdecl)]
public static extern CudnnStatus cudnnSetStream(IntPtr handle, IntPtr streamId);
```

The cuDNN handle is thread-local. One handle per thread, or synchronize access. `cudnnSetStream` binds the handle to a CUDA stream (use the same stream as cuBLAS for ordering).

### Tensor Descriptor Creation

C signatures:

```c
cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t *tensorDesc);
cudnnStatus_t cudnnSetTensor4dDescriptor(
    cudnnTensorDescriptor_t tensorDesc,
    cudnnTensorFormat_t format,
    cudnnDataType_t dataType,
    int n, int c, int h, int w);
cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc);
```

C# P/Invoke:

```csharp
// cudnnTensorDescriptor_t = IntPtr

[DllImport("cudnn64_9", CallingConvention = CallingConvention.Cdecl)]
public static extern CudnnStatus cudnnCreateTensorDescriptor(out IntPtr tensorDesc);

[DllImport("cudnn64_9", CallingConvention = CallingConvention.Cdecl)]
public static extern CudnnStatus cudnnSetTensor4dDescriptor(
    IntPtr tensorDesc,
    int format,      // cudnnTensorFormat_t
    int dataType,    // cudnnDataType_t
    int n, int c, int h, int w);

[DllImport("cudnn64_9", CallingConvention = CallingConvention.Cdecl)]
public static extern CudnnStatus cudnnDestroyTensorDescriptor(IntPtr tensorDesc);
```

For NHWC format with FP16: `cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, N, C, H, W)`. The N, C, H, W parameters are always in logical NCHW order regardless of the format parameter — the format controls the physical memory layout.

### Filter Descriptor Creation

C signatures:

```c
cudnnStatus_t cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t *filterDesc);
cudnnStatus_t cudnnSetFilter4dDescriptor(
    cudnnFilterDescriptor_t filterDesc,
    cudnnDataType_t dataType,
    cudnnTensorFormat_t format,
    int k, int c, int h, int w);
cudnnStatus_t cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t filterDesc);
```

C# P/Invoke:

```csharp
// cudnnFilterDescriptor_t = IntPtr

[DllImport("cudnn64_9", CallingConvention = CallingConvention.Cdecl)]
public static extern CudnnStatus cudnnCreateFilterDescriptor(out IntPtr filterDesc);

[DllImport("cudnn64_9", CallingConvention = CallingConvention.Cdecl)]
public static extern CudnnStatus cudnnSetFilter4dDescriptor(
    IntPtr filterDesc,
    int dataType,    // cudnnDataType_t
    int format,      // cudnnTensorFormat_t
    int k, int c, int h, int w);

[DllImport("cudnn64_9", CallingConvention = CallingConvention.Cdecl)]
public static extern CudnnStatus cudnnDestroyFilterDescriptor(IntPtr filterDesc);
```

Parameters: `k` = number of output feature maps, `c` = number of input feature maps, `h` = filter height, `w` = filter width. For a 3x3 conv with 320 input and 320 output channels: `(desc, CUDNN_DATA_HALF, CUDNN_TENSOR_NHWC, 320, 320, 3, 3)`.

### Convolution Descriptor

C signatures:

```c
cudnnStatus_t cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t *convDesc);
cudnnStatus_t cudnnSetConvolution2dDescriptor(
    cudnnConvolutionDescriptor_t convDesc,
    int padH, int padW,
    int strideH, int strideW,
    int dilationH, int dilationW,
    cudnnConvolutionMode_t mode,
    cudnnDataType_t computeType);
cudnnStatus_t cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t convDesc);

// Enable Tensor Cores
cudnnStatus_t cudnnSetConvolutionMathType(
    cudnnConvolutionDescriptor_t convDesc,
    cudnnMathType_t mathType);

// Grouped convolution (depthwise when groups == C)
cudnnStatus_t cudnnSetConvolutionGroupCount(
    cudnnConvolutionDescriptor_t convDesc,
    int groupCount);
```

C# P/Invoke:

```csharp
[DllImport("cudnn64_9", CallingConvention = CallingConvention.Cdecl)]
public static extern CudnnStatus cudnnCreateConvolutionDescriptor(out IntPtr convDesc);

[DllImport("cudnn64_9", CallingConvention = CallingConvention.Cdecl)]
public static extern CudnnStatus cudnnSetConvolution2dDescriptor(
    IntPtr convDesc,
    int padH, int padW,
    int strideH, int strideW,
    int dilationH, int dilationW,
    int mode,         // cudnnConvolutionMode_t
    int computeType); // cudnnDataType_t

[DllImport("cudnn64_9", CallingConvention = CallingConvention.Cdecl)]
public static extern CudnnStatus cudnnDestroyConvolutionDescriptor(IntPtr convDesc);

[DllImport("cudnn64_9", CallingConvention = CallingConvention.Cdecl)]
public static extern CudnnStatus cudnnSetConvolutionMathType(IntPtr convDesc, int mathType);

[DllImport("cudnn64_9", CallingConvention = CallingConvention.Cdecl)]
public static extern CudnnStatus cudnnSetConvolutionGroupCount(IntPtr convDesc, int groupCount);
```

For diffusion models: always use `CUDNN_CROSS_CORRELATION` (1), not `CUDNN_CONVOLUTION` (0). Neural network "convolution" is actually cross-correlation. Use `CUDNN_DATA_FLOAT` (0) as computeType even with FP16 I/O — this gives FP32 accumulation for numerical stability.

### Algorithm Selection

C signatures:

```c
cudnnStatus_t cudnnGetConvolutionForwardAlgorithm_v7(
    cudnnHandle_t handle,
    cudnnTensorDescriptor_t srcDesc,
    cudnnFilterDescriptor_t filterDesc,
    cudnnConvolutionDescriptor_t convDesc,
    cudnnTensorDescriptor_t destDesc,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionFwdAlgoPerf_t *perfResults);

cudnnStatus_t cudnnFindConvolutionForwardAlgorithm(
    cudnnHandle_t handle,
    cudnnTensorDescriptor_t xDesc,
    cudnnFilterDescriptor_t wDesc,
    cudnnConvolutionDescriptor_t convDesc,
    cudnnTensorDescriptor_t yDesc,
    const int requestedAlgoCount,
    int *returnedAlgoCount,
    cudnnConvolutionFwdAlgoPerf_t *perfResults);

cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(
    cudnnHandle_t handle,
    cudnnTensorDescriptor_t xDesc,
    cudnnFilterDescriptor_t wDesc,
    cudnnConvolutionDescriptor_t convDesc,
    cudnnTensorDescriptor_t yDesc,
    cudnnConvolutionFwdAlgo_t algo,
    size_t *sizeInBytes);
```

C# P/Invoke:

```csharp
[StructLayout(LayoutKind.Sequential)]
public struct CudnnConvolutionFwdAlgoPerf
{
    public int algo;            // cudnnConvolutionFwdAlgo_t
    public int status;          // cudnnStatus_t
    public float time;          // milliseconds
    public nuint memory;        // workspace bytes
    public int determinism;     // cudnnDeterminism_t
    public int mathType;        // cudnnMathType_t
    public int reserved0;
    public int reserved1;
    public int reserved2;
}

[DllImport("cudnn64_9", CallingConvention = CallingConvention.Cdecl)]
public static extern CudnnStatus cudnnGetConvolutionForwardAlgorithm_v7(
    IntPtr handle,
    IntPtr srcDesc,
    IntPtr filterDesc,
    IntPtr convDesc,
    IntPtr destDesc,
    int requestedAlgoCount,
    out int returnedAlgoCount,
    [Out] CudnnConvolutionFwdAlgoPerf[] perfResults);

[DllImport("cudnn64_9", CallingConvention = CallingConvention.Cdecl)]
public static extern CudnnStatus cudnnFindConvolutionForwardAlgorithm(
    IntPtr handle,
    IntPtr xDesc,
    IntPtr wDesc,
    IntPtr convDesc,
    IntPtr yDesc,
    int requestedAlgoCount,
    out int returnedAlgoCount,
    [Out] CudnnConvolutionFwdAlgoPerf[] perfResults);

[DllImport("cudnn64_9", CallingConvention = CallingConvention.Cdecl)]
public static extern CudnnStatus cudnnGetConvolutionForwardWorkspaceSize(
    IntPtr handle,
    IntPtr xDesc,
    IntPtr wDesc,
    IntPtr convDesc,
    IntPtr yDesc,
    int algo,           // cudnnConvolutionFwdAlgo_t
    out nuint sizeInBytes);
```

**`cudnnGetConvolutionForwardAlgorithm_v7`** uses heuristics (fast, no GPU work). **`cudnnFindConvolutionForwardAlgorithm`** benchmarks all algorithms (slow, runs actual convolutions). For SharpInference: use `_v7` at first call, optionally run `Find` once and cache results per unique (shape, dtype, algo) tuple. Request `CUDNN_CONVOLUTION_FWD_ALGO_COUNT` (8) algorithms and pick the first with `status == CUDNN_STATUS_SUCCESS`.

### Output Dimension Query

```c
cudnnStatus_t cudnnGetConvolution2dForwardOutputDim(
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t inputTensorDesc,
    const cudnnFilterDescriptor_t filterDesc,
    int *n, int *c, int *h, int *w);
```

```csharp
[DllImport("cudnn64_9", CallingConvention = CallingConvention.Cdecl)]
public static extern CudnnStatus cudnnGetConvolution2dForwardOutputDim(
    IntPtr convDesc,
    IntPtr inputTensorDesc,
    IntPtr filterDesc,
    out int n, out int c, out int h, out int w);
```

Output dimension formula: `outputDim = 1 + (inputDim + 2*pad - (((filterDim-1)*dilation)+1)) / stride`

### Convolution Forward Execution

C signature ([cuDNN CNN Library](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-cnn-library.html)):

```c
cudnnStatus_t cudnnConvolutionForward(
    cudnnHandle_t handle,
    const void *alpha,
    cudnnTensorDescriptor_t xDesc,
    const void *x,
    cudnnFilterDescriptor_t wDesc,
    const void *w,
    cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionFwdAlgo_t algo,
    void *workSpace,
    size_t workSpaceSizeInBytes,
    const void *beta,
    cudnnTensorDescriptor_t yDesc,
    void *y);
```

C# P/Invoke:

```csharp
[DllImport("cudnn_cnn_infer64_9", CallingConvention = CallingConvention.Cdecl)]
public static extern CudnnStatus cudnnConvolutionForward(
    IntPtr handle,
    ref float alpha,         // host pointer to scaling factor
    IntPtr xDesc,
    ulong x,                 // CUdeviceptr (device memory)
    IntPtr wDesc,
    ulong w,                 // CUdeviceptr
    IntPtr convDesc,
    int algo,                // cudnnConvolutionFwdAlgo_t
    ulong workSpace,         // CUdeviceptr (or 0 if no workspace)
    nuint workSpaceSizeInBytes,
    ref float beta,          // host pointer to scaling factor
    IntPtr yDesc,
    ulong y);                // CUdeviceptr
```

Computes: `y = alpha * conv(x, w) + beta * y`. Typical usage: `alpha = 1.0f, beta = 0.0f`. The alpha/beta pointers must match the compute type (float for FP32 compute, even with FP16 tensors). Pass device pointers as `ulong` (CUdeviceptr). The workspace pointer can be 0 (null) if `workSpaceSizeInBytes` is 0.

**Important**: `cudnnConvolutionForward` lives in the CNN inference library (`cudnn_cnn_infer64_9.dll`), not the core library. The descriptor functions live in the ops library (`cudnn_ops64_9.dll`) or core library (`cudnn64_9.dll`).

### Bias Addition After Convolution

cuDNN convolution does not handle bias internally. Use `cudnnAddTensor` to add bias:

```c
cudnnStatus_t cudnnAddTensor(
    cudnnHandle_t handle,
    const void *alpha,
    cudnnTensorDescriptor_t aDesc,
    const void *A,
    const void *beta,
    cudnnTensorDescriptor_t cDesc,
    void *C);
```

```csharp
[DllImport("cudnn64_9", CallingConvention = CallingConvention.Cdecl)]
public static extern CudnnStatus cudnnAddTensor(
    IntPtr handle,
    ref float alpha,
    IntPtr aDesc,        // bias descriptor: shape (1, C, 1, 1) for NCHW
    ulong A,             // bias device pointer
    ref float beta,
    IntPtr cDesc,        // output descriptor (same as yDesc from conv)
    ulong C);            // output device pointer (in-place)
```

Bias descriptor shape: `(1, outChannels, 1, 1)` for NCHW, `(1, 1, 1, outChannels)` for NHWC. Call with `alpha=1.0f, beta=1.0f` to add bias to existing convolution output.

### NHWC vs NCHW Performance on Ampere+

Source: [NVIDIA Convolutional Layers Performance Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html)

**NHWC is strongly recommended** for all GPU convolution on Ampere and newer architectures:

- NHWC avoids layout transpose overhead that NCHW incurs when Tensor Cores are used
- cuDNN automatically converts NCHW to NHWC internally when Tensor Cores are requested, adding overhead
- The performance gap is most significant for small filter sizes (1x1) where computation is low relative to data movement
- On A100 (Ampere), NHWC convolution with 3x3 filters, 1024 channels, 64x64 spatial achieved ~250 TFLOPS

**Tensor Core alignment requirements** ([NVIDIA DL Performance Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html)):

| Data Type | Channel Alignment (C and K) |
|-----------|----------------------------|
| FP16      | Divisible by 8             |
| TF32      | Divisible by 4             |
| INT8      | Divisible by 16            |

cuDNN 7.6.3+ automatically pads channels for Tensor Core alignment when using packed NCHW, but NHWC with natively aligned channels avoids padding overhead entirely. All SD 1.5/SDXL channel counts (320, 640, 1280) are divisible by 8, so FP16 Tensor Cores work without padding.

For the first layer with 4-channel latent input: 4 is not divisible by 8, so cuDNN will auto-pad. This is a single layer and negligible.

### cuDNN Version Compatibility Matrix

Source: [cuDNN Support Matrix](https://docs.nvidia.com/deeplearning/cudnn/latest/reference/support-matrix.html)

| cuDNN Version | CUDA Toolkit | Min Linux Driver | Min Windows Driver | GPU Architectures (SM) |
|---|---|---|---|---|
| 9.21.0 (CUDA 13.x) | 13.0, 13.1, 13.2 | >= 580.65.06 | N/A | Turing (7.5), Ampere (8.0, 8.6), Ada (8.9), Hopper (9.0), Blackwell (12.0) |
| 9.21.0 (CUDA 12.x) | 12.0 - 12.9 | >= 525.60.13 | >= 527.41 | Turing (7.5), Ampere (8.0, 8.6), Ada (8.9), Hopper (9.0), Blackwell* (12.0) |

*Blackwell on CUDA 12.x requires CUDA >= 12.8, Linux driver >= 570.26, Windows driver >= 570.65.

**Recommended target**: cuDNN 9.x with CUDA 12.x build for maximum driver compatibility (driver >= 527.41 on Windows). The CUDA 12.x cuDNN build is forward-compatible with all CUDA 12.x toolkit versions.

### Algorithm Performance by Kernel Size

Source: [NVIDIA DL Performance Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html), [cuDNN CNN Library](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-cnn-library.html)

**3x3 convolutions** (dominant in diffusion UNet residual blocks):
- Winograd (`CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED`, value 7) is typically fastest for stride=1, dilation=1
- Winograd transforms the 3x3 convolution into element-wise multiplications in a transformed domain, reducing arithmetic from 9 multiplies to ~4 per output
- Falls back to implicit precomp GEMM for stride > 1 or dilation > 1

**1x1 convolutions** (used in channel projection layers):
- Implicit GEMM (`CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM`, value 0) or implicit precomp GEMM (value 1)
- 1x1 convolution is mathematically equivalent to a batched matrix multiply, so cuBLAS GEMM can also be used directly
- Zero workspace required for implicit GEMM

**Downsampling convolutions** (stride=2, common at resolution boundaries):
- FFT-based algorithms do not support stride > 1
- Winograd does not support stride > 1
- Implicit precomp GEMM (value 1) is the typical best choice

## Key Numbers / Constants

### cudnnStatus_t

| Value | Name |
|-------|------|
| 0 | CUDNN_STATUS_SUCCESS |
| 1 | CUDNN_STATUS_NOT_INITIALIZED |
| 2 | CUDNN_STATUS_ALLOC_FAILED |
| 3 | CUDNN_STATUS_BAD_PARAM |
| 4 | CUDNN_STATUS_INTERNAL_ERROR |
| 5 | CUDNN_STATUS_INVALID_VALUE |
| 6 | CUDNN_STATUS_ARCH_MISMATCH |
| 7 | CUDNN_STATUS_MAPPING_ERROR |
| 8 | CUDNN_STATUS_EXECUTION_FAILED |
| 9 | CUDNN_STATUS_NOT_SUPPORTED |
| 10 | CUDNN_STATUS_LICENSE_ERROR |

Source: [Rust cudnn-sys](https://docs.rs/cudnn-sys/latest/cudnn_sys/enum.cudnnStatus_t.html), [managedCuda](https://surban.github.io/managedCuda/api/ManagedCuda.CudaDNN.cudnnStatus.html)

### cudnnDataType_t

| Value | Name | Description |
|-------|------|-------------|
| 0 | CUDNN_DATA_FLOAT | 32-bit float |
| 1 | CUDNN_DATA_DOUBLE | 64-bit double |
| 2 | CUDNN_DATA_HALF | 16-bit float (FP16) |
| 3 | CUDNN_DATA_INT8 | 8-bit signed integer |
| 4 | CUDNN_DATA_INT32 | 32-bit signed integer |
| 5 | CUDNN_DATA_INT8x4 | Vectorized INT8 |
| 6 | CUDNN_DATA_UINT8 | 8-bit unsigned integer |
| 7 | CUDNN_DATA_UINT8x4 | Vectorized UINT8 |
| 8 | CUDNN_DATA_INT8x32 | Vectorized INT8 |
| 14 | CUDNN_DATA_BFLOAT16 | 16-bit bfloat |

Source: [torch-cudnn ffi.lua](https://github.com/NVIDIA/torch-cudnn/blob/master/ffi.lua)

### cudnnTensorFormat_t

| Value | Name | Description |
|-------|------|-------------|
| 0 | CUDNN_TENSOR_NCHW | Row-major: batch, channels, height, width |
| 1 | CUDNN_TENSOR_NHWC | Row-major: batch, height, width, channels |
| 2 | CUDNN_TENSOR_NCHW_VECT_C | Vectorized channels (INT8x4/INT8x32) |

### cudnnConvolutionMode_t

| Value | Name | Description |
|-------|------|-------------|
| 0 | CUDNN_CONVOLUTION | Mathematical convolution (kernel is flipped) |
| 1 | CUDNN_CROSS_CORRELATION | Cross-correlation (no kernel flip — used by all neural networks) |

### cudnnConvolutionFwdAlgo_t

| Value | Name | Workspace | Notes |
|-------|------|-----------|-------|
| 0 | CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM | None | Good for 1x1 |
| 1 | CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM | Small | General purpose, good fallback |
| 2 | CUDNN_CONVOLUTION_FWD_ALGO_GEMM | Large | Explicit matrix product |
| 3 | CUDNN_CONVOLUTION_FWD_ALGO_DIRECT | None | Rarely selected |
| 4 | CUDNN_CONVOLUTION_FWD_ALGO_FFT | Large | Good for large filters |
| 5 | CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING | Medium | FFT with tiling |
| 6 | CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD | Small | Fast for 3x3 stride 1 |
| 7 | CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED | Medium | Fastest for 3x3 stride 1 |

### cudnnMathType_t

| Value | Name | Description |
|-------|------|-------------|
| 0 | CUDNN_DEFAULT_MATH | No Tensor Cores (except TF32 on Ampere+ by default) |
| 1 | CUDNN_TENSOR_OP_MATH | Enable Tensor Core operations |
| 2 | CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION | Tensor Cores + auto-convert FP32 to FP16 |
| 3 | CUDNN_FMA_MATH | FMA-only, no Tensor Cores, no TF32 |

Source: [cuDNN Core Concepts](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/developer/core-concepts.html), [NVIDIA Tensor Ops Blog](https://developer.nvidia.com/blog/tensor-ops-made-easier-in-cudnn/)

### Typical Diffusion UNet Tensor Shapes (SD 1.5, batch=1)

| Stage | Input Shape (NCHW) | Filter | Stride | Output Shape |
|-------|-------------------|--------|--------|--------------|
| Down block 1 | (1, 320, 64, 64) | 3x3 | 1 | (1, 320, 64, 64) |
| Down block 2 | (1, 320, 64, 64) | 3x3 | 1 | (1, 640, 64, 64) |
| Downsample | (1, 640, 64, 64) | 3x3 | 2 | (1, 640, 32, 32) |
| Down block 3 | (1, 640, 32, 32) | 3x3 | 1 | (1, 1280, 32, 32) |
| Downsample | (1, 1280, 32, 32) | 3x3 | 2 | (1, 1280, 16, 16) |
| Down block 4 | (1, 1280, 16, 16) | 3x3 | 1 | (1, 1280, 16, 16) |
| Middle | (1, 1280, 8, 8) | 3x3 | 1 | (1, 1280, 8, 8) |
| Up blocks | Mirror of down path | 3x3 | 1 | Mirror |
| 1x1 proj | (1, 1280, H, W) | 1x1 | 1 | (1, 320, H, W) |

All channel counts (320, 640, 1280) are divisible by 8 — Tensor Core compatible with FP16 NHWC.

### Workspace Size Guidelines

| Algorithm | Typical Workspace Range |
|-----------|------------------------|
| Implicit GEMM (0) | 0 bytes |
| Implicit Precomp GEMM (1) | 4 KB - 64 MB |
| Explicit GEMM (2) | 10 MB - 500 MB+ |
| FFT (4) | 50 MB - 500 MB+ |
| FFT Tiling (5) | 10 MB - 200 MB |
| Winograd (6) | 1 KB - 10 MB |
| Winograd Nonfused (7) | 1 MB - 100 MB |

Strategy: Allocate a single workspace buffer of 256 MB at initialization. If `cudnnGetConvolutionForwardWorkspaceSize` returns a size exceeding the buffer, fall back to a smaller algorithm. 256 MB covers all Winograd and most implicit GEMM cases. Source: [PyTorch workspace limit discussion](https://github.com/pytorch/pytorch/issues/49207)

## Data Layouts / Formats

### Handle Types

| C Type | C# Type | Description |
|--------|---------|-------------|
| cudnnHandle_t | `IntPtr` | cuDNN library handle |
| cudnnTensorDescriptor_t | `IntPtr` | Tensor descriptor |
| cudnnFilterDescriptor_t | `IntPtr` | Filter/weight descriptor |
| cudnnConvolutionDescriptor_t | `IntPtr` | Convolution operation descriptor |

### NCHW Memory Layout

For a tensor (N=1, C=3, H=2, W=2):
```
Memory: [c0h0w0, c0h0w1, c0h1w0, c0h1w1, c1h0w0, c1h0w1, ...]
Stride: [C*H*W, H*W, W, 1] = [12, 4, 2, 1]
```

### NHWC Memory Layout

For a tensor (N=1, C=3, H=2, W=2):
```
Memory: [h0w0c0, h0w0c1, h0w0c2, h0w1c0, h0w1c1, h0w1c2, ...]
Stride: [H*W*C, 1, W*C, C] = [12, 1, 6, 3]
```

NHWC is preferred because channels are contiguous in memory, enabling coalesced Tensor Core access.

### cuDNN DLL Split (v9.x)

| DLL (Windows) | Linux SO | Contains |
|---------------|----------|----------|
| `cudnn64_9.dll` | `libcudnn.so.9` | Core: handle, descriptors, error handling |
| `cudnn_ops64_9.dll` | `libcudnn_ops.so.9` | Ops: tensor ops, activation, pooling, softmax |
| `cudnn_cnn_infer64_9.dll` | `libcudnn_cnn_infer.so.9` | CNN inference: convolution forward, bias add |
| `cudnn_adv64_9.dll` | `libcudnn_adv.so.9` | Advanced: RNN, attention |
| `cudnn_graph64_9.dll` | `libcudnn_graph.so.9` | Graph API (modern replacement) |

Source: [cuDNN Installation Guide](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/windows.html)

## Algorithm Steps

### Complete Conv2D Forward Pass

```
 1. Create descriptors (once per unique shape, cache and reuse):
    a. cudnnCreateTensorDescriptor(&xDesc)
    b. cudnnSetTensor4dDescriptor(xDesc, NHWC, HALF, N, C_in, H, W)
    c. cudnnCreateFilterDescriptor(&wDesc)
    d. cudnnSetFilter4dDescriptor(wDesc, HALF, NHWC, C_out, C_in, kH, kW)
    e. cudnnCreateConvolutionDescriptor(&convDesc)
    f. cudnnSetConvolution2dDescriptor(convDesc, padH, padW, strH, strW,
                                       dilH, dilW, CROSS_CORRELATION, FLOAT)
    g. cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH)
    h. cudnnGetConvolution2dForwardOutputDim(convDesc, xDesc, wDesc, &n, &c, &h, &w)
    i. cudnnCreateTensorDescriptor(&yDesc)
    j. cudnnSetTensor4dDescriptor(yDesc, NHWC, HALF, n, c, outH, outW)

 2. Select algorithm (once per unique shape, cache result):
    a. cudnnGetConvolutionForwardAlgorithm_v7(handle, xDesc, wDesc, convDesc,
                                               yDesc, 8, &count, perfResults)
    b. Pick first perfResults[i] where status == SUCCESS and memory <= workspaceLimit

 3. Query workspace:
    a. cudnnGetConvolutionForwardWorkspaceSize(handle, xDesc, wDesc, convDesc,
                                                yDesc, algo, &wsSize)
    b. Ensure workspace buffer >= wsSize (reuse pre-allocated buffer)

 4. Execute convolution:
    a. float alpha = 1.0f, beta = 0.0f;
    b. cudnnConvolutionForward(handle, &alpha, xDesc, x, wDesc, w,
                                convDesc, algo, workspace, wsSize,
                                &beta, yDesc, y)

 5. Add bias (if present):
    a. cudnnSetTensor4dDescriptor(biasDesc, NHWC, HALF, 1, C_out, 1, 1)
    b. float biasAlpha = 1.0f, biasBeta = 1.0f;
    c. cudnnAddTensor(handle, &biasAlpha, biasDesc, bias, &biasBeta, yDesc, y)

 6. Synchronize if needed:
    a. cuStreamSynchronize(stream)
```

### Descriptor Caching Strategy

```
Key = (N, C_in, C_out, H, W, kH, kW, padH, padW, strH, strW, dilH, dilW, dtype, format)

Dictionary<ConvKey, CachedConvPlan> cache;
struct CachedConvPlan {
    IntPtr xDesc, wDesc, yDesc, convDesc, biasDesc;
    int algo;
    nuint workspaceSize;
}

On first call for a given key:
  - Create all descriptors
  - Run algorithm selection
  - Query workspace size
  - Store in cache

On subsequent calls:
  - Lookup by key
  - Call cudnnConvolutionForward directly
```

## Reference Implementations

| Implementation | Location | Notes |
|---------------|----------|-------|
| cuDNN Legacy API | [docs (8.9.2)](https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-892/api/index.html) | Canonical function signatures |
| cuDNN CNN Library | [docs (latest)](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-cnn-library.html) | Conv forward, algorithm selection |
| cuDNN Ops Library | [docs (latest)](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-ops-library.html) | Tensor/filter descriptors |
| Peter Goldsborough tutorial | [blog](http://www.goldsborough.me/cuda/ml/cudnn/c++/2017/10/01/14-37-23-convolutions_with_cudnn/) | End-to-end C++ example |
| odashi cuDNN gist | [GitHub Gist](https://gist.github.com/odashi/1c20ba90388cf02330e1b95963d78039) | Minimal complete example |
| managedCuda cuDNN | [GitHub](https://surban.github.io/managedCuda/api/ManagedCuda.CudaDNN.html) | C# cuDNN wrapper (reference for P/Invoke patterns) |
| PyTorch cuDNN integration | [GitHub](https://github.com/pytorch/pytorch/commit/34561dadcddf6ce3c76daf16f09a09adc9c7b73b) | Bias handling outside cuDNN conv |
| NVIDIA DL Performance Guide | [docs](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html) | NHWC vs NCHW, Tensor Core alignment |
| Conv2D GPU Baselines blog | [kernyan.com](https://www.kernyan.com/cuda/cpu/2025/07/17/Conv2D_GPU_Baseline.html) | cuDNN vs CUTLASS analysis |

## Differences Between Implementations

| Aspect | managedCuda | SharpInference (planned) |
|--------|-------------|--------------------------|
| cuDNN version target | cuDNN 7.x/8.x | cuDNN 9.x (latest stable) |
| Binding style | DllImport with wrappers | LibraryImport (.NET 10) |
| DLL resolution | Hardcoded names | NativeLibrary.SetDllImportResolver |
| Tensor format | NCHW default | NHWC for Tensor Core perf |
| Error handling | Exception wrappers | CudnnCheck() helper, throw on non-success |
| Descriptor lifecycle | Manual create/destroy | IDisposable wrappers, descriptor cache |
| Algorithm selection | Per-call | Cached by shape key |
| Workspace | Per-call allocation | Single pre-allocated buffer (256 MB) |

| Aspect | PyTorch cuDNN | SharpInference (planned) |
|--------|--------------|--------------------------|
| API level | Runtime API (cudaStream_t) | Driver API (CUstream) — interchangeable |
| Bias handling | Separate addmm or custom kernel | cudnnAddTensor |
| Algorithm cache | torch.backends.cudnn.benchmark | Always-on cache by shape key |
| Graph API | Not used for conv | Possible future migration |
| Workspace limit | Configurable (default unlimited) | 256 MB cap with fallback |

## Open Questions

- [x] cuDNN version compatibility matrix — cuDNN 9.21.0 with CUDA 12.x requires driver >= 527.41 (Windows)
- [x] NHWC vs NCHW performance — NHWC is strictly better on Ampere+; cuDNN auto-converts NCHW to NHWC for Tensor Cores (adding overhead)
- [x] Workspace size requirements — 256 MB covers all practical algorithms for diffusion tensors; Winograd 3x3 needs 1-100 MB; implicit GEMM needs 0
- [ ] Which cuDNN 9.x sub-DLL exports which function — need to verify at runtime whether `cudnnConvolutionForward` is in `cudnn_cnn_infer64_9.dll` or re-exported from `cudnn64_9.dll`
- [ ] cuDNN Graph API migration — the legacy API is deprecated in 9.x; Graph API offers fused conv+bias+activation but is significantly more complex to P/Invoke
- [ ] Grouped convolution performance — SD 1.5 uses standard (group=1) convolutions, but ControlNet and some SDXL variants use grouped convolutions
- [ ] cudnnConvolutionBiasActivationForward — fused conv+bias+ReLU in a single call; may help for ResBlock patterns but only supports ReLU, not SiLU/GELU

## Implementation Notes

1. **LibraryImport over DllImport** — .NET 7+ source-gen is faster and avoids marshalling overhead. Use `[LibraryImport("cudnn64_9")]` with `SetLastError = false`.

2. **Cross-platform DLL resolution** — Use `NativeLibrary.SetDllImportResolver` to map `cudnn64_9` to `libcudnn.so.9`, `cudnn_cnn_infer64_9` to `libcudnn_cnn_infer.so.9`, etc.

3. **Descriptor caching** — cuDNN descriptors are cheap to create (~microseconds) but the algorithm selection is expensive. Cache `(shape_key -> algo, workspace_size)` pairs. Descriptors themselves can be pooled.

4. **Single workspace buffer** — Allocate 256 MB once at startup. All convolutions share it (they execute sequentially on the same stream). If a selected algorithm needs more, fall back to implicit GEMM (0 workspace).

5. **NHWC everywhere** — Store all tensors in NHWC format from the start. This avoids layout conversion overhead and enables Tensor Core utilization on Ampere+. Weight tensors must also be NHWC — convert once during model loading.

6. **FP16 I/O with FP32 compute** — Set tensor descriptors to `CUDNN_DATA_HALF`, convolution compute type to `CUDNN_DATA_FLOAT`. This gives FP16 memory bandwidth with FP32 accumulation precision. Alpha/beta are `float` pointers.

7. **Tensor Core enablement** — Call `cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH)` (value 1) for every convolution descriptor. Without this, Tensor Cores are not used even if data is FP16 NHWC.

8. **Cross-correlation, not convolution** — Always use `CUDNN_CROSS_CORRELATION` (1). Neural network frameworks call it "convolution" but the operation is mathematically cross-correlation (no kernel flip).

9. **Error checking** — Check every cuDNN call. Create a `CudnnCheck(CudnnStatus status)` helper that throws `CudnnException` with the status code name on failure. Silent failures cause data corruption.

10. **Stream interop** — CUstream (driver API) and cudaStream_t (runtime API) are bitwise identical. Pass the same stream handle to `cudnnSetStream` and `cublasSetStream` to ensure correct ordering between conv and GEMM operations.

11. **1x1 optimization path** — For 1x1 convolutions (no padding, stride 1), consider bypassing cuDNN entirely and using cuBLAS GEMM directly. A 1x1 conv on (N, C_in, H, W) is equivalent to a matrix multiply of shape (C_out, C_in) x (C_in, H*W). This can be faster due to cuBLAS's highly optimized GEMM.

12. **Deprecation awareness** — The legacy API (`cudnnConvolutionForward` and friends) is marked deprecated in cuDNN 9.x. NVIDIA recommends the Graph API for new code. However, the legacy API remains functional and is simpler to P/Invoke. Plan to migrate to Graph API in a future phase if performance warrants it.
