// Kernels around the per-expert GEMMs of MoE prefill (many tokens per expert): the expert-major gather of activations, the
// activation product that feeds the down GEMM, and the combine back into token order.
//
// 16-bit operands are F16 or BF16 (bf16 != 0), whichever the GEMM resolved to; the bits are written raw so one kernel serves both.

#include <cuda_fp16.h>
#include <cuda_bf16.h>

__device__ __forceinline__ unsigned short moe_pack16(float v, int bf16)
{
    return bf16 ? __bfloat16_as_ushort(__float2bfloat16_rn(v)) : __half_as_ushort(__float2half_rn(v));
}

// out[r] = 16-bit(x[perm[r]]) for the expert-major rows r < rows. Rows whose perm is -1 are zero.
// Launch: grid = rows, block = 256.
extern "C" __global__ void moe_gather_rows_16(
    unsigned short* __restrict__ out,
    const float* __restrict__ x,
    const int* __restrict__ perm,
    int rows, int hidden, int bf16)
{
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int t = perm[r];
    for (int c = threadIdx.x; c < hidden; c += blockDim.x)
        out[(size_t)r * hidden + c] = t >= 0 ? moe_pack16(x[(size_t)t * hidden + c], bf16) : (unsigned short)0;
}

// out = 16-bit(act(gate) * up), elementwise over count values: SiLU, or tanh-GELU when gelu != 0.
// Launch: grid = ceil(count / 256), block = 256.
extern "C" __global__ void moe_act_mul_16(
    unsigned short* __restrict__ out,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    long long count, int gelu, int bf16)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float g = gate[i];
    float a;
    if (gelu) {
        const float c = 0.7978845608028654f;   // sqrt(2/pi)
        a = 0.5f * g * (1.0f + tanhf(c * (g + 0.044715f * g * g * g)));
    } else {
        a = g / (1.0f + expf(-g));
    }
    out[i] = moe_pack16(a * up[i], bf16);
}

// out[t] = sigmoid(sharedGateLogit[t]) * shared[t] + sum_j topkWeight[t,j] * expertOut[pairSlot[t,j]]   (pair slots outside
// [0, expertRows) are skipped; shared and sharedGateLogit may be null). Sums in slot order.
// Launch: grid = (tokens, ceil(hidden / 256)), block = 256.
extern "C" __global__ void moe_combine_pairs_f32(
    float* __restrict__ out,
    const float* __restrict__ expertOut,
    const int* __restrict__ pairSlot,
    const float* __restrict__ topkWeight,
    const float* __restrict__ shared,
    const float* __restrict__ sharedGateLogit,
    int hidden, int topk, int expertRows)
{
    const int t = blockIdx.x;
    const int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= hidden) return;

    float acc = 0.0f;
    if (shared != nullptr) {
        float s = shared[(size_t)t * hidden + c];
        if (sharedGateLogit != nullptr) s *= 1.0f / (1.0f + expf(-sharedGateLogit[t]));
        acc = s;
    }
    for (int j = 0; j < topk; ++j) {
        const int slot = pairSlot[(size_t)t * topk + j];
        if (slot >= 0 && slot < expertRows) acc += topkWeight[(size_t)t * topk + j] * expertOut[(size_t)slot * hidden + c];
    }
    out[(size_t)t * hidden + c] = acc;
}
