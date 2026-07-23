// W8A8 IMMA chain kernels (INFERENCE_ACCEL_GRIND §H5, ViDiT-Q/SmoothQuant recipe):
//   1. Per-row (per-token) dynamic INT8 activation quantization — one block per row, two-phase
//      (absmax reduce, then quantize); rowScale[m] = absmax/127 is the DEQUANT scale.
//   2. Dequant epilogue: D_i32[M,N] (from cublasLtMatmul COMPUTE_32I) × rowScale[m] × wScale[n]
//      (+ optional bias[n]) → F16 or F32. cuBLASLt cannot apply per-row×per-col float scale vectors
//      to an int32 result, so this one elementwise pass replaces the F16 GEMM's fused epilogue.
// Weights quantize per-output-channel ON THE HOST at load time (they are static); only activations
// pay a per-step kernel. All kernels grid-stride and carry no sm_89 dependencies (Ampere targets).

#include <cuda_fp16.h>

#define ROW_THREADS 256u

// ── 1a. Per-row absmax + quantize, F16 input ────────────────────────────────────────────────
// x [rows, cols] row-major F16 → q [rows, cols] int8, rowScale [rows] F32 (dequant scale absmax/127).
// One block per row: phase 1 grid-strides the row for absmax (shared reduce), phase 2 re-reads and
// quantizes with round-to-nearest. absmax==0 writes scale 1.0 (all-zero row stays a well-defined 0).
extern "C" __global__ void w8a8_quant_rowwise_f16(
    const __half* __restrict__ x, signed char* __restrict__ q,
    float* __restrict__ rowScale, unsigned int cols)
{
    __shared__ float sm[ROW_THREADS];
    __shared__ float sInv;
    unsigned int row = blockIdx.x;
    unsigned int tid = threadIdx.x;
    const __half* xr = x + (unsigned long long)row * cols;
    signed char* qr = q + (unsigned long long)row * cols;

    float m = 0.0f;
    for (unsigned int i = tid; i < cols; i += ROW_THREADS)
    {
        float v = fabsf(__half2float(xr[i]));
        if (v > m) m = v;
    }
    sm[tid] = m;
    __syncthreads();
    for (unsigned int s = ROW_THREADS >> 1; s > 0; s >>= 1)
    {
        if (tid < s && sm[tid + s] > sm[tid]) sm[tid] = sm[tid + s];
        __syncthreads();
    }
    if (tid == 0)
    {
        float amax = sm[0];
        rowScale[row] = amax > 0.0f ? amax / 127.0f : 1.0f;
        sInv = amax > 0.0f ? 127.0f / amax : 0.0f;
    }
    __syncthreads();
    float inv = sInv;
    for (unsigned int i = tid; i < cols; i += ROW_THREADS)
    {
        float v = __half2float(xr[i]) * inv;
        int iv = __float2int_rn(v);
        if (iv > 127) iv = 127;
        if (iv < -127) iv = -127;
        qr[i] = (signed char)iv;
    }
}

// ── 1b. F32-input twin ──────────────────────────────────────────────────────────────────────
extern "C" __global__ void w8a8_quant_rowwise_f32(
    const float* __restrict__ x, signed char* __restrict__ q,
    float* __restrict__ rowScale, unsigned int cols)
{
    __shared__ float sm[ROW_THREADS];
    __shared__ float sInv;
    unsigned int row = blockIdx.x;
    unsigned int tid = threadIdx.x;
    const float* xr = x + (unsigned long long)row * cols;
    signed char* qr = q + (unsigned long long)row * cols;

    float m = 0.0f;
    for (unsigned int i = tid; i < cols; i += ROW_THREADS)
    {
        float v = fabsf(xr[i]);
        if (v > m) m = v;
    }
    sm[tid] = m;
    __syncthreads();
    for (unsigned int s = ROW_THREADS >> 1; s > 0; s >>= 1)
    {
        if (tid < s && sm[tid + s] > sm[tid]) sm[tid] = sm[tid + s];
        __syncthreads();
    }
    if (tid == 0)
    {
        float amax = sm[0];
        rowScale[row] = amax > 0.0f ? amax / 127.0f : 1.0f;
        sInv = amax > 0.0f ? 127.0f / amax : 0.0f;
    }
    __syncthreads();
    float inv = sInv;
    for (unsigned int i = tid; i < cols; i += ROW_THREADS)
    {
        float v = xr[i] * inv;
        int iv = __float2int_rn(v);
        if (iv > 127) iv = 127;
        if (iv < -127) iv = -127;
        qr[i] = (signed char)iv;
    }
}

// ── 2a. Dequant epilogue → F16 ──────────────────────────────────────────────────────────────
// d [rows, cols] int32 row-major, rowScale [rows] F32, wScale [cols] F32 (per-output-channel),
// bias [cols] F32 or null → out [rows, cols] F16: out = d·rowScale[r]·wScale[c] (+ bias[c]).
extern "C" __global__ void w8a8_dequant_bias_f16(
    const int* __restrict__ d, const float* __restrict__ rowScale,
    const float* __restrict__ wScale, const float* __restrict__ bias,
    __half* __restrict__ out, unsigned int rows, unsigned int cols)
{
    unsigned long long n = (unsigned long long)rows * cols;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x)
    {
        unsigned int r = (unsigned int)(i / cols);
        unsigned int c = (unsigned int)(i - (unsigned long long)r * cols);
        float v = (float)d[i] * rowScale[r] * wScale[c];
        if (bias) v += bias[c];
        out[i] = __float2half(v);
    }
}

// ── 2b. Dequant epilogue → F32 ──────────────────────────────────────────────────────────────
extern "C" __global__ void w8a8_dequant_bias_f32(
    const int* __restrict__ d, const float* __restrict__ rowScale,
    const float* __restrict__ wScale, const float* __restrict__ bias,
    float* __restrict__ out, unsigned int rows, unsigned int cols)
{
    unsigned long long n = (unsigned long long)rows * cols;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x)
    {
        unsigned int r = (unsigned int)(i / cols);
        unsigned int c = (unsigned int)(i - (unsigned long long)r * cols);
        float v = (float)d[i] * rowScale[r] * wScale[c];
        if (bias) v += bias[c];
        out[i] = v;
    }
}
