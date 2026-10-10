// F16-cache twins of the two fused graph-decode scatter kernels in lm_f32.cu (lm_qkv_rope_scatter_f32 and
// lm_qknorm_rope_scatter_f32): identical arithmetic, but the key and value caches are __half and the converted value is stored.
// q stays F32. They live in their own module so the tuned lm_f32.ptx is not rebuilt.
//
// Build: ../lm/build.sh (nvcc -ptx -arch=sm_80).

#include <cuda_fp16.h>

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

    if (gid >= qElems + kvElems) {
        // v: plain copy into the cache slot (same addressing as lm_kv_append_f32 at tNew=1).
        unsigned long long vi = gid - qElems - kvElems;
        unsigned int h = (unsigned int)(vi / headDim);
        unsigned int i = (unsigned int)(vi % headDim);
        vCache[(((unsigned long long)h * maxSeq) + (unsigned int)pos) * headDim + i] = __float2half_rn(vIn[vi]);
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
        kCache[(((unsigned long long)h * maxSeq) + (unsigned int)pos) * headDim + i] = __float2half_rn(outv);
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

    if (h >= nq + nkv) {
        // v head: plain copy into the cache slot (identical addressing to lm_kv_append_f32, tNew=1).
        unsigned int vh = h - nq - nkv;
        const float* src = vIn + (size_t)vh * headDim;
        __half* dst = vCache + (((unsigned long long)vh * maxSeq) + (unsigned int)pos) * headDim;
        for (unsigned int i = threadIdx.x; i < headDim; i += blockDim.x)
            dst[i] = __float2half_rn(src[i]);
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
        if (isQ) qDst[i] = outv; else kDst[i] = __float2half_rn(outv);
    }
}
