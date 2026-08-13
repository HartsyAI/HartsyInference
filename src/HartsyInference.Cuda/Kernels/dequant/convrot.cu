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

// ── Fused ConvRot + per-row int8 quantization ───────────────────────────────────────────────
// The split version cost 7 bytes per element (read x, write the rotated f16, read it back, write int8);
// staging the row in shared memory instead costs 3 (read x, write int8), because the per-row absmax the
// quantizer needs is computed off the SAME shared copy the quantize pass then reads. One block per row, so
// `group` must divide `cols`. Shared = cols floats + the reduction scratch, which caps the usable `cols`;
// the caller keeps the two-kernel path above for anything wider (see HasFusedConvRotQuant).
//
// Matches convrot_rotate_f16 -> w8a8_quant_rowwise_f16 exactly, INCLUDING the rounding: same radix-4
// stages, same amax/127 scale, same __float2int_rn + clamp to +/-127.
#define CONVROT_QUANT_BODY(LOADEXPR, ROUNDEXPR)                                               \
    extern __shared__ float sh[];                                                             \
    __shared__ float sm[256];                                                                 \
    __shared__ float sInv;                                                                    \
    unsigned int row = blockIdx.x;                                                            \
    unsigned int tid = threadIdx.x;                                                           \
    signed char* qr = q + (unsigned long long)row * cols;                                     \
    for (unsigned int i = tid; i < cols; i += blockDim.x) sh[i] = (LOADEXPR);                 \
    __syncthreads();                                                                          \
    unsigned int quarter = group >> 2;                                                        \
    CONVROT_STAGE(sh, group, quarter, (cols / group) * quarter)                               \
    /* The split pair stored the rotated row as f16 before quantizing, so round through f16 here \
       too -- otherwise this kernel is MORE accurate than the path it replaces and silently      \
       changes output. Keeping it bit-identical makes this a pure traffic win. */                \
    for (unsigned int i = tid; i < cols; i += blockDim.x) sh[i] = (ROUNDEXPR);                   \
    __syncthreads();                                                                             \
    float m = 0.0f;                                                                           \
    for (unsigned int i = tid; i < cols; i += blockDim.x)                                     \
    { float a = fabsf(sh[i]); if (a > m) m = a; }                                             \
    sm[tid] = m;                                                                              \
    __syncthreads();                                                                          \
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1)                                    \
    {                                                                                         \
        if (tid < s && sm[tid + s] > sm[tid]) sm[tid] = sm[tid + s];                          \
        __syncthreads();                                                                      \
    }                                                                                         \
    if (tid == 0)                                                                             \
    {                                                                                         \
        float amax = sm[0];                                                                   \
        rowScale[row] = amax > 0.0f ? amax / 127.0f : 1.0f;                                   \
        sInv = amax > 0.0f ? 127.0f / amax : 0.0f;                                            \
    }                                                                                         \
    __syncthreads();                                                                          \
    float inv = sInv;                                                                         \
    for (unsigned int i = tid; i < cols; i += blockDim.x)                                     \
    {                                                                                         \
        int iv = __float2int_rn(sh[i] * inv);                                                 \
        if (iv > 127) iv = 127;                                                               \
        if (iv < -127) iv = -127;                                                             \
        qr[i] = (signed char)iv;                                                              \
    }

extern "C" __global__ void convrot_quant_rowwise_f16(
    const __half* __restrict__ x, signed char* __restrict__ q,
    float* __restrict__ rowScale, unsigned int cols, unsigned int group)
{
    const __half* xr = x + (unsigned long long)blockIdx.x * cols;
    CONVROT_QUANT_BODY(__half2float(xr[i]), __half2float(__float2half(sh[i])))
}

extern "C" __global__ void convrot_quant_rowwise_f32(
    const float* __restrict__ x, signed char* __restrict__ q,
    float* __restrict__ rowScale, unsigned int cols, unsigned int group)
{
    const float* xr = x + (unsigned long long)blockIdx.x * cols;
    CONVROT_QUANT_BODY(xr[i], sh[i])
}
