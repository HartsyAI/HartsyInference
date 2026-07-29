// stepcache.cu — device-side gate reduction for the across-step feature cache
// (TeaCache / First-Block-Cache family, see docs/Research/STEP_ACCELERATION.md §2).
//
// The cache gate needs the relative-L1 distance sum|a−b| / sum|b| between this
// step's first-block output and the previous step's. Reading either tensor's
// DataPointer on the host would evict it from the GPU activation cache and force
// a pageable re-upload mid-forward (the exact stream-drain pathology the
// drain-free loops removed), so the reduction must run device-side: each block
// tree-reduces its partial |a−b| and |b| sums in shared memory and atomically
// accumulates into a 2-float result buffer the caller zeroes beforehand. The
// host then reads back 8 bytes once per forward.
//
// Compile: nvcc -ptx -arch=sm_80 stepcache.cu -o stepcache.ptx  (see build.sh)

#include <cuda_fp16.h>

// 64-bit indexing throughout: seqLen × hidden exceeds u32 at video scale (PHASE_3_DEVIATIONS #12).

extern "C" __global__ void stepcache_rel_l1_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ sums,          // sums[0] += Σ|a−b|, sums[1] += Σ|b|; caller pre-zeroes
    unsigned long long n)
{
    __shared__ float sDiff[256];
    __shared__ float sRef[256];

    float diff = 0.0f;
    float ref = 0.0f;
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long i = idx; i < n; i += stride)
    {
        float av = a[i];
        float bv = b[i];
        diff += fabsf(av - bv);
        ref += fabsf(bv);
    }

    sDiff[threadIdx.x] = diff;
    sRef[threadIdx.x] = ref;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            sDiff[threadIdx.x] += sDiff[threadIdx.x + s];
            sRef[threadIdx.x] += sRef[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x < 32)
    {
        float d = sDiff[threadIdx.x] + sDiff[threadIdx.x + 32];
        float r = sRef[threadIdx.x] + sRef[threadIdx.x + 32];
        for (int off = 16; off > 0; off >>= 1)
        {
            d += __shfl_down_sync(0xffffffffu, d, off);
            r += __shfl_down_sync(0xffffffffu, r, off);
        }
        if (threadIdx.x == 0)
        {
            atomicAdd(&sums[0], d);
            atomicAdd(&sums[1], r);
        }
    }
}

extern "C" __global__ void stepcache_rel_l1_f16(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    float* __restrict__ sums,
    unsigned long long n)
{
    __shared__ float sDiff[256];
    __shared__ float sRef[256];

    float diff = 0.0f;
    float ref = 0.0f;
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long i = idx; i < n; i += stride)
    {
        // F32 accumulate: an F16 running sum saturates at seqLen×hidden scale.
        float av = __half2float(a[i]);
        float bv = __half2float(b[i]);
        diff += fabsf(av - bv);
        ref += fabsf(bv);
    }

    sDiff[threadIdx.x] = diff;
    sRef[threadIdx.x] = ref;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            sDiff[threadIdx.x] += sDiff[threadIdx.x + s];
            sRef[threadIdx.x] += sRef[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x < 32)
    {
        float d = sDiff[threadIdx.x] + sDiff[threadIdx.x + 32];
        float r = sRef[threadIdx.x] + sRef[threadIdx.x + 32];
        for (int off = 16; off > 0; off >>= 1)
        {
            d += __shfl_down_sync(0xffffffffu, d, off);
            r += __shfl_down_sync(0xffffffffu, r, off);
        }
        if (threadIdx.x == 0)
        {
            atomicAdd(&sums[0], d);
            atomicAdd(&sums[1], r);
        }
    }
}
