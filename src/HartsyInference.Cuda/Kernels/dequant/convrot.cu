// ConvRot activation rotation for ComfyUI's int8_tensorwise + convrot weights (comfy-kitchen
// tensor/int8_utils.py `_rotate_activation`). The quantizer stored W @ H^T, so the consumer owes the
// GEMM an x @ H, applied independently to each contiguous `group`-wide slice of the input dim.
//
// H = kron(h4, ..., h4) / sqrt(group) with h4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]], so the
// transform factors into log4(group) radix-4 stages and no matrix is ever materialized. Each stage is
// out[t] = (sum - 2*v[3-t]) / 2 over its 4 lanes — the 1/2 per stage accumulates to the 1/sqrt(group)
// normalization. Verified equal to comfy-kitchen's _build_hadamard to the last bit in float64.
//
// One block handles `groupsPerBlock` groups through shared memory; the butterflies of a stage touch
// disjoint slots, so only the stage boundaries need a barrier.

#include <cuda_fp16.h>

#define CONVROT_STAGE(SH, GROUP, QUARTER, WORK)                                        \
    for (unsigned int stride = 1; stride < (GROUP); stride <<= 2)                      \
    {                                                                                  \
        for (unsigned int t = threadIdx.x; t < (WORK); t += blockDim.x)                \
        {                                                                              \
            unsigned int gi = t / (QUARTER);                                           \
            unsigned int u = t - gi * (QUARTER);                                       \
            unsigned int lane = u % stride;                                            \
            unsigned int blk = (u / stride) * (stride << 2);                           \
            float* a = (SH) + gi * (GROUP) + blk + lane;                               \
            float v0 = a[0], v1 = a[stride], v2 = a[2 * stride], v3 = a[3 * stride];   \
            float s = v0 + v1 + v2 + v3;                                               \
            a[0] = 0.5f * (s - 2.0f * v3);                                             \
            a[stride] = 0.5f * (s - 2.0f * v2);                                        \
            a[2 * stride] = 0.5f * (s - 2.0f * v1);                                    \
            a[3 * stride] = 0.5f * (s - 2.0f * v0);                                    \
        }                                                                              \
        __syncthreads();                                                               \
    }

extern "C" __global__ void convrot_rotate_f16(
    const __half* __restrict__ x, __half* __restrict__ out,
    unsigned int group, unsigned int groupsPerBlock, unsigned long long totalGroups)
{
    extern __shared__ float sh[];
    unsigned int quarter = group >> 2;
    unsigned long long firstGroup = (unsigned long long)blockIdx.x * groupsPerBlock;
    if (firstGroup >= totalGroups) return;
    unsigned long long remaining = totalGroups - firstGroup;
    unsigned int localGroups = remaining < groupsPerBlock ? (unsigned int)remaining : groupsPerBlock;
    unsigned long long base = firstGroup * group;
    unsigned int total = localGroups * group;

    for (unsigned int i = threadIdx.x; i < total; i += blockDim.x) sh[i] = __half2float(x[base + i]);
    __syncthreads();
    CONVROT_STAGE(sh, group, quarter, localGroups * quarter)
    for (unsigned int i = threadIdx.x; i < total; i += blockDim.x) out[base + i] = __float2half(sh[i]);
}

extern "C" __global__ void convrot_rotate_f32(
    const float* __restrict__ x, float* __restrict__ out,
    unsigned int group, unsigned int groupsPerBlock, unsigned long long totalGroups)
{
    extern __shared__ float sh[];
    unsigned int quarter = group >> 2;
    unsigned long long firstGroup = (unsigned long long)blockIdx.x * groupsPerBlock;
    if (firstGroup >= totalGroups) return;
    unsigned long long remaining = totalGroups - firstGroup;
    unsigned int localGroups = remaining < groupsPerBlock ? (unsigned int)remaining : groupsPerBlock;
    unsigned long long base = firstGroup * group;
    unsigned int total = localGroups * group;

    for (unsigned int i = threadIdx.x; i < total; i += blockDim.x) sh[i] = x[base + i];
    __syncthreads();
    CONVROT_STAGE(sh, group, quarter, localGroups * quarter)
    for (unsigned int i = threadIdx.x; i < total; i += blockDim.x) out[base + i] = sh[i];
}
