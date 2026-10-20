// F16-cache twins of the two fused graph-decode scatter kernels in lm_f32.cu (lm_qkv_rope_scatter_f32 and
// lm_qknorm_rope_scatter_f32): identical arithmetic, but the key and value caches are __half and the converted value is stored.
// q stays F32. They live in their own module so the tuned lm_f32.ptx is not rebuilt.
//
// Build: ../lm/build.sh (nvcc -ptx -arch=sm_80).

#include <cuda_fp16.h>

// F32 to F16 with saturation: a value beyond the F16 range stores as +-65504 instead of +-inf (a NaN stays a NaN, which the comparisons below do not touch),
// so one extreme key or value cannot poison the softmax of every later token.
__device__ __forceinline__ __half kv_half_sat(float v)
{
    return __float2half_rn(v > 65504.0f ? 65504.0f : (v < -65504.0f ? -65504.0f : v));
}


extern "C" __global__ void lm_qkv_rope_scatter_f16kv(
    float* __restrict__ qOut,
    __half* __restrict__ kCache,
    __half* __restrict__ vCache,
    const float* __restrict__ qIn,
    const float* __restrict__ kIn,
    const float* __restrict__ vIn,
    const float* __restrict__ cosTable,
    const float* __restrict__ sinTable,
    unsigned int nq,
    unsigned int nkv,
    unsigned int headDim,
    unsigned int rotaryDim,
    int interleaved,
    unsigned int maxSeq,
    const int* __restrict__ devicePos)
{
    unsigned long long gid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long qElems = (unsigned long long)nq * headDim;
    unsigned long long kvElems = (unsigned long long)nkv * headDim;
    unsigned long long total = qElems + 2ull * kvElems;
    if (gid >= total) return;
    int pos = devicePos[1];
    if (gid >= qElems && (unsigned int)pos >= maxSeq) return;   // a slot past the capacity writes no K/V

    if (gid >= qElems + kvElems) {
        // v: plain copy into the cache slot (same addressing as lm_kv_append_f32 at tNew=1).
        unsigned long long vi = gid - qElems - kvElems;
        unsigned int h = (unsigned int)(vi / headDim);
        unsigned int i = (unsigned int)(vi % headDim);
        vCache[(((unsigned long long)h * maxSeq) + (unsigned int)pos) * headDim + i] = kv_half_sat(vIn[vi]);
        return;
    }

    bool isQ = gid < qElems;
    unsigned long long si = isQ ? gid : gid - qElems;       // element within the q or k section
    unsigned int i = (unsigned int)(si % headDim);          // dim within the head
    unsigned long long headBase = si - i;                   // this head's first element (section-relative)
    const float* src = isQ ? qIn : kIn;
    size_t baseCs = (size_t)pos * headDim;

    float outv;
    if (interleaved) {
        // lm_rope_decode_interleaved: pair p = i>>1 uses table entry p; partial rotary rides on
        // identity (cos=1, sin=0) table rows past rotaryDim/2.
        unsigned int p = i >> 1;
        float c = cosTable[baseCs + p];
        float s = sinTable[baseCs + p];
        float self = src[headBase + i];
        float other = src[headBase + (i ^ 1u)];
        outv = (i & 1u) == 0u ? self * c - other * s : self * c + other * s;
    } else {
        // lm_rope_decode_splithalf: rotate [0, rdim) as lower/upper halves; pass through the rest.
        unsigned int rdim = (rotaryDim == 0u || rotaryDim > headDim) ? headDim : rotaryDim;
        unsigned int half = rdim >> 1;
        float self = src[headBase + i];
        if (i < half) {
            outv = self * cosTable[baseCs + i] - src[headBase + i + half] * sinTable[baseCs + i];
        } else if (i < rdim) {
            outv = self * cosTable[baseCs + i] + src[headBase + i - half] * sinTable[baseCs + i];
        } else {
            outv = self;
        }
    }

    if (isQ) {
        qOut[si] = outv;
    } else {
        unsigned int h = (unsigned int)(si / headDim);
        kCache[(((unsigned long long)h * maxSeq) + (unsigned int)pos) * headDim + i] = kv_half_sat(outv);
    }
}

extern "C" __global__ void lm_qknorm_rope_scatter_f16kv(
    float* __restrict__ qOut,
    __half* __restrict__ kCache,
    __half* __restrict__ vCache,
    const float* __restrict__ qIn,
    const float* __restrict__ kIn,
    const float* __restrict__ vIn,
    const float* __restrict__ qNormW,
    const float* __restrict__ kNormW,
    const float* __restrict__ cosTable,
    const float* __restrict__ sinTable,
    unsigned int nq,
    unsigned int nkv,
    unsigned int headDim,
    unsigned int rotaryDim,
    int interleaved,
    float eps,
    unsigned int maxSeq,
    const int* __restrict__ devicePos)
{
    extern __shared__ float sdata_qns[];
    unsigned int h = blockIdx.x;
    if (h >= nq + 2u * nkv) return;
    int pos = devicePos[1];
    if (h >= nq && (unsigned int)pos >= maxSeq) return;   // a slot past the capacity writes no K/V

    if (h >= nq + nkv) {
        // v head: plain copy into the cache slot (identical addressing to lm_kv_append_f32, tNew=1).
        unsigned int vh = h - nq - nkv;
        const float* src = vIn + (size_t)vh * headDim;
        __half* dst = vCache + (((unsigned long long)vh * maxSeq) + (unsigned int)pos) * headDim;
        for (unsigned int i = threadIdx.x; i < headDim; i += blockDim.x)
            dst[i] = kv_half_sat(src[i]);
        return;
    }

    bool isQ = h < nq;
    unsigned int sh = isQ ? h : h - nq;                     // head index within its section
    const float* src = (isQ ? qIn : kIn) + (size_t)sh * headDim;
    const float* w = isQ ? qNormW : kNormW;

    // dit_rmsnorm_f32 reduction, verbatim (normDim = headDim, one block per row/head).
    float partial = 0.0f;
    for (unsigned int i = threadIdx.x; i < headDim; i += blockDim.x)
    {
        float v = src[i];
        partial += v * v;
    }
    sdata_qns[threadIdx.x] = partial;
    __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
            sdata_qns[threadIdx.x] += sdata_qns[threadIdx.x + s];
        __syncthreads();
    }
    float invRms = rsqrtf(sdata_qns[0] / (float)headDim + eps);

    size_t baseCs = (size_t)pos * headDim;
    float* qDst = qOut + (size_t)sh * headDim;
    __half* kDst = kCache + (((unsigned long long)sh * maxSeq) + (unsigned int)pos) * headDim;
    for (unsigned int i = threadIdx.x; i < headDim; i += blockDim.x)
    {
        float self = src[i] * invRms * w[i];                // dit_rmsnorm_f32's output expression
        float outv;
        if (interleaved) {
            unsigned int p = i >> 1;
            unsigned int j = i ^ 1u;
            float other = src[j] * invRms * w[j];
            float c = cosTable[baseCs + p];
            float s = sinTable[baseCs + p];
            outv = (i & 1u) == 0u ? self * c - other * s : self * c + other * s;
        } else {
            unsigned int rdim = (rotaryDim == 0u || rotaryDim > headDim) ? headDim : rotaryDim;
            unsigned int half = rdim >> 1;
            if (i < half) {
                unsigned int j = i + half;
                float other = src[j] * invRms * w[j];
                outv = self * cosTable[baseCs + i] - other * sinTable[baseCs + i];
            } else if (i < rdim) {
                unsigned int j = i - half;
                float other = src[j] * invRms * w[j];
                outv = self * cosTable[baseCs + i] + other * sinTable[baseCs + i];
            } else {
                outv = self;
            }
        }
        if (isQ) qDst[i] = outv; else kDst[i] = kv_half_sat(outv);
    }
}

// The eager (non-graph) F16 KV append: lm_kv_append_f16 of lm_f32.cu with a saturating store. Kept here so the tuned lm_f32.ptx is not rebuilt.
extern "C" __global__ void lm_kv_append_f16_sat(
    __half* __restrict__ buffer,
    const float* __restrict__ newKv,
    unsigned int heads,
    unsigned int maxSeq,
    unsigned int tNew,
    unsigned int headDim,
    unsigned int offset,
    unsigned long long total,
    const int* __restrict__ dPos)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    if (dPos != nullptr) offset = (unsigned int)dPos[1];

    unsigned int d = (unsigned int)(i % headDim);
    unsigned long long rem = i / headDim;
    unsigned int ti = (unsigned int)(rem % tNew);
    unsigned int h = (unsigned int)(rem / tNew);

    if (offset + ti >= maxSeq) return;
    buffer[(((unsigned long long)h * maxSeq) + (offset + ti)) * headDim + d] = kv_half_sat(newKv[i]);
}

// F32 to F16 with the same saturation as the F16 KV store, for activations headed into a cuDNN fused-attention graph (CudnnSdpa):
// a value past +-65504 stores as the limit rather than infinity, and a NaN stays a NaN. A plain cast would turn one extreme Q/K/V into an
// infinity that the fused softmax then spreads over every row. Elementwise over `count` values.
extern "C" __global__ void cast_f32_to_f16_sat(
    __half* __restrict__ output,
    const float* __restrict__ input,
    unsigned int count)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    output[i] = kv_half_sat(input[i]);
}
