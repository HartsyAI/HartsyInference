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

// out = 16-bit(act(gate) * up) where gate and up are themselves 16-bit (the grouped GEMM writes 16-bit outputs).
// Launch: grid = ceil(count / 256), block = 256.
extern "C" __global__ void moe_act_mul_16x16(
    unsigned short* __restrict__ out,
    const unsigned short* __restrict__ gate,
    const unsigned short* __restrict__ up,
    long long count, int gelu, int bf16)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float g, u;
    if (bf16) {
        g = __bfloat162float(__ushort_as_bfloat16(gate[i]));
        u = __bfloat162float(__ushort_as_bfloat16(up[i]));
    } else {
        g = __half2float(__ushort_as_half(gate[i]));
        u = __half2float(__ushort_as_half(up[i]));
    }
    float a;
    if (gelu) {
        const float c = 0.7978845608028654f;
        a = 0.5f * g * (1.0f + tanhf(c * (g + 0.044715f * g * g * g)));
    } else {
        a = g / (1.0f + expf(-g));
    }
    out[i] = moe_pack16(a * u, bf16);
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

// In-place F16 -> BF16 over count values (each value is read before it is written, so the buffer may be the source and the destination):
// the second half of a quantized -> BF16 dequant that lands in the destination buffer first as F16. Eight values per thread.
// Launch: grid = ceil(max(count / 8, 1) / 256), block = 256.
extern "C" __global__ void moe_cast_f16_to_bf16_inplace(unsigned short* __restrict__ buf, long long count)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long n8 = count >> 3;
    if (i < n8) {
        uint4 v = reinterpret_cast<uint4*>(buf)[i];
        unsigned int w[4] = {v.x, v.y, v.z, v.w};
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float lo = __half2float(__ushort_as_half((unsigned short)(w[j] & 0xFFFFu)));
            const float hi = __half2float(__ushort_as_half((unsigned short)(w[j] >> 16)));
            w[j] = (unsigned int)__bfloat16_as_ushort(__float2bfloat16_rn(lo)) | ((unsigned int)__bfloat16_as_ushort(__float2bfloat16_rn(hi)) << 16);
        }
        reinterpret_cast<uint4*>(buf)[i] = make_uint4(w[0], w[1], w[2], w[3]);
    }
    const long long tail = count - (n8 << 3);
    if (i < tail) {
        const long long at = (n8 << 3) + i;
        buf[at] = __bfloat16_as_ushort(__float2bfloat16_rn(__half2float(__ushort_as_half(buf[at]))));
    }
}
