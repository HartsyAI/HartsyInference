// The generic norm-and-modulate (IBackend.NormModulate) and chunk-activate (IBackend.ChunkActivate) kernels.
//
//   norm_modulate_rows_<in>_<out>     one block per row along the last axis: y = (norm(x) · w + b) · scale + shift, the norm
//                                     (LayerNorm, RMSNorm or the L2-floored RMS norm) reduced by a block-wide sum.
//   norm_modulate_flat_<in>_<out>     one thread per element for NormKind.None: the affine and modulation alone.
//   norm_modulate_channel_<in>_<out>  one thread per row along the channel axis of [N, C, ...], reducing C in F64.
//   chunk_activate_f32                cuts [rows, count·width] into count outputs through 1 + x, tanh, sigmoid or identity.
//
// The geometry is one by-value parameter (NormModulateGeom, mirrored field for field by CudaKernels.NormModulate.cs), so a
// launch captures into a graph without constant memory. Every absent vector is a null pointer and its flag bit clear.
// Contract and host reference: NormModulateContract / NormModulateReference, ChunkActivateReference.
//
// Build: ./build.sh (nvrtc through ../build_common.sh).

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define NM_WEIGHT 1
#define NM_BIAS 2
#define NM_SCALE 4
#define NM_SHIFT 8
#define NM_INDEX 16

#define KIND_NONE 0
#define KIND_LAYER 1
#define KIND_RMS 2
#define KIND_L2 3

struct NormModulateGeom
{
    unsigned long long rows;
    long long spatial;
    long long rowsPerMod;
    long long rowStride;
    long long scaleOffset;
    long long shiftOffset;
    float eps;
    int width;
    int kind;
    int onePlus;
    int flags;
};

__device__ __forceinline__ float load_act(const float* p, size_t i) { return p[i]; }
__device__ __forceinline__ float load_act(const __half* p, size_t i) { return __half2float(p[i]); }
__device__ __forceinline__ float load_act(const __nv_bfloat16* p, size_t i) { return __bfloat162float(p[i]); }
__device__ __forceinline__ void store_act(float* p, size_t i, float v) { p[i] = v; }
__device__ __forceinline__ void store_act(__half* p, size_t i, float v) { p[i] = __float2half(v); }
__device__ __forceinline__ void store_act(__nv_bfloat16* p, size_t i, float v) { p[i] = __float2bfloat16(v); }

// Block-wide sum; blockDim.x is a multiple of 32 and at most 1024.
__device__ __forceinline__ float block_sum(float v)
{
    __shared__ float warpSums[32];
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xffffffffu, v, offset);
    unsigned int warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    if (lane == 0) warpSums[warp] = v;
    __syncthreads();
    unsigned int warps = (blockDim.x + 31u) >> 5;
    v = lane < warps ? warpSums[lane] : 0.0f;
    if (warp == 0)
    {
        for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xffffffffu, v, offset);
        if (lane == 0) warpSums[0] = v;
    }
    __syncthreads();
    float total = warpSums[0];
    __syncthreads();
    return total;
}

__device__ __forceinline__ size_t row_base(const NormModulateGeom& g, unsigned long long row)
{
    return (size_t)(row / g.spatial) * (size_t)g.width * (size_t)g.spatial + (size_t)(row % g.spatial);
}

__device__ __forceinline__ long long modulation_row(const NormModulateGeom& g, const int* rowIndex, unsigned long long row)
{
    if (g.flags & NM_INDEX) return rowIndex[row];
    return g.rowsPerMod == 0 ? 0 : (long long)(row / (unsigned long long)g.rowsPerMod);
}

// The affine and modulation of one element after its norm.
__device__ __forceinline__ float finish(const NormModulateGeom& g, float v, int d, const float* weight, const float* bias,
    const float* scale, const float* shift, long long m)
{
    if (g.flags & NM_WEIGHT) v *= weight[d];
    if (g.flags & NM_BIAS) v += bias[d];
    if (g.flags & NM_SCALE)
    {
        float s = scale[g.scaleOffset + m * g.rowStride + d];
        v *= g.onePlus ? 1.0f + s : s;
    }
    if (g.flags & NM_SHIFT) v += shift[g.shiftOffset + m * g.rowStride + d];
    return v;
}

template <typename TIn, typename TOut>
__device__ __forceinline__ void norm_modulate_rows(TOut* __restrict__ out, const TIn* __restrict__ in, const float* __restrict__ weight,
    const float* __restrict__ bias, const float* __restrict__ scale, const float* __restrict__ shift, const int* __restrict__ rowIndex,
    const NormModulateGeom g)
{
    for (unsigned long long row = blockIdx.x; row < g.rows; row += gridDim.x)
    {
        size_t base = row_base(g, row);
        float mean = 0.0f, factor = 1.0f;
        if (g.kind == KIND_LAYER)
        {
            float partial = 0.0f;
            for (int i = threadIdx.x; i < g.width; i += blockDim.x) partial += load_act(in, base + i);
            mean = block_sum(partial) / (float)g.width;
            float vpart = 0.0f;
            for (int i = threadIdx.x; i < g.width; i += blockDim.x)
            {
                float d = load_act(in, base + i) - mean;
                vpart += d * d;
            }
            factor = rsqrtf(block_sum(vpart) / (float)g.width + g.eps);
        }
        else if (g.kind != KIND_NONE)
        {
            float partial = 0.0f;
            for (int i = threadIdx.x; i < g.width; i += blockDim.x)
            {
                float v = load_act(in, base + i);
                partial += v * v;
            }
            float sumSq = block_sum(partial);
            factor = g.kind == KIND_RMS ? rsqrtf(sumSq / (float)g.width + g.eps) : sqrtf((float)g.width) / fmaxf(sqrtf(sumSq), g.eps);
        }
        long long m = modulation_row(g, rowIndex, row);
        for (int i = threadIdx.x; i < g.width; i += blockDim.x)
        {
            float v = load_act(in, base + i);
            if (g.kind == KIND_LAYER) v = (v - mean) * factor;
            else if (g.kind != KIND_NONE) v *= factor;
            store_act(out, base + i, finish(g, v, i, weight, bias, scale, shift, m));
        }
    }
}

template <typename TIn, typename TOut>
__device__ __forceinline__ void norm_modulate_flat(TOut* __restrict__ out, const TIn* __restrict__ in, const float* __restrict__ weight,
    const float* __restrict__ bias, const float* __restrict__ scale, const float* __restrict__ shift, const int* __restrict__ rowIndex,
    const NormModulateGeom g)
{
    unsigned long long total = g.rows * (unsigned long long)g.width;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (unsigned long long)gridDim.x * blockDim.x)
    {
        unsigned long long rowSpan = (unsigned long long)g.width * (unsigned long long)g.spatial;
        unsigned long long n = i / rowSpan, rem = i % rowSpan;
        int d = (int)(rem / (unsigned long long)g.spatial);
        unsigned long long row = n * (unsigned long long)g.spatial + rem % (unsigned long long)g.spatial;
        long long m = modulation_row(g, rowIndex, row);
        store_act(out, i, finish(g, load_act(in, i), d, weight, bias, scale, shift, m));
    }
}

template <typename TIn, typename TOut>
__device__ __forceinline__ void norm_modulate_channel(TOut* __restrict__ out, const TIn* __restrict__ in, const float* __restrict__ weight,
    const float* __restrict__ bias, const NormModulateGeom g)
{
    for (unsigned long long row = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; row < g.rows; row += (unsigned long long)gridDim.x * blockDim.x)
    {
        size_t base = row_base(g, row);
        size_t stride = (size_t)g.spatial;
        double sum = 0.0, sumSq = 0.0;
        for (int d = 0; d < g.width; d++)
        {
            double v = load_act(in, base + d * stride);
            sum += v;
            sumSq += v * v;
        }
        float mean = 0.0f, factor = 1.0f;
        if (g.kind == KIND_LAYER)
        {
            double m = sum / g.width;
            double variance = 0.0;
            for (int d = 0; d < g.width; d++)
            {
                double diff = load_act(in, base + d * stride) - m;
                variance += diff * diff;
            }
            mean = (float)m;
            factor = (float)(1.0 / sqrt(variance / g.width + (double)g.eps));
        }
        else if (g.kind == KIND_RMS)
        {
            factor = (float)(1.0 / sqrt(sumSq / g.width + (double)g.eps));
        }
        else if (g.kind == KIND_L2)
        {
            factor = sqrtf((float)g.width) / fmaxf((float)sqrt(sumSq), g.eps);
        }
        for (int d = 0; d < g.width; d++)
        {
            float v = load_act(in, base + d * stride);
            if (g.kind == KIND_LAYER) v = (v - mean) * factor;
            else if (g.kind != KIND_NONE) v *= factor;
            if (g.flags & NM_WEIGHT) v *= weight[d];
            if (g.flags & NM_BIAS) v += bias[d];
            store_act(out, base + d * stride, v);
        }
    }
}

struct ChunkActivateArgs
{
    float* out[8];
    int act[8];
    unsigned long long rows;
    int width;
    int count;
};

__device__ __forceinline__ float chunk_activate(float x, int act)
{
    if (act == 1) return 1.0f + x;
    if (act == 2) return tanhf(x);
    if (act == 3) return 1.0f / (1.0f + expf(-x));
    return x;
}

extern "C" {

#define NORM_MODULATE(SUFFIX, TIN, TOUT) \
__global__ void norm_modulate_rows_##SUFFIX(TOUT* __restrict__ out, const TIN* __restrict__ in, const float* __restrict__ weight, \
    const float* __restrict__ bias, const float* __restrict__ scale, const float* __restrict__ shift, const int* __restrict__ rowIndex, \
    const NormModulateGeom g) \
{ norm_modulate_rows<TIN, TOUT>(out, in, weight, bias, scale, shift, rowIndex, g); } \
__global__ void norm_modulate_flat_##SUFFIX(TOUT* __restrict__ out, const TIN* __restrict__ in, const float* __restrict__ weight, \
    const float* __restrict__ bias, const float* __restrict__ scale, const float* __restrict__ shift, const int* __restrict__ rowIndex, \
    const NormModulateGeom g) \
{ norm_modulate_flat<TIN, TOUT>(out, in, weight, bias, scale, shift, rowIndex, g); } \
__global__ void norm_modulate_channel_##SUFFIX(TOUT* __restrict__ out, const TIN* __restrict__ in, const float* __restrict__ weight, \
    const float* __restrict__ bias, const NormModulateGeom g) \
{ norm_modulate_channel<TIN, TOUT>(out, in, weight, bias, g); }

NORM_MODULATE(f32_f32, float, float)
NORM_MODULATE(f32_f16, float, __half)
NORM_MODULATE(f16_f16, __half, __half)
NORM_MODULATE(f16_f32, __half, float)
NORM_MODULATE(bf16_bf16, __nv_bfloat16, __nv_bfloat16)

__global__ void chunk_activate_f32(const float* __restrict__ in, const ChunkActivateArgs a)
{
    unsigned long long total = a.rows * (unsigned long long)a.width * (unsigned long long)a.count;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (unsigned long long)gridDim.x * blockDim.x)
    {
        unsigned long long packed = (unsigned long long)a.width * (unsigned long long)a.count;
        unsigned long long row = i / packed;
        int rem = (int)(i % packed);
        int c = rem / a.width, d = rem % a.width;
        a.out[c][row * (unsigned long long)a.width + d] = chunk_activate(in[i], a.act[c]);
    }
}

}
