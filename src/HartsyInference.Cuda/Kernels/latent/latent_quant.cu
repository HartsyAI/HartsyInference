// Latent-cache quantization kernels: FP8 e4m3 + ue8m0/32, FP4 e2m1 + e4m3/16 and FP4 e2m1 + ue8m0/32, matching
// the reference act_quant / fp4_act_quant (see latent_codec.cuh).
//
// act_quant_dequant_inplace_f32
//   Replaces every value with its quantize-dequantize round trip, one thread per scale group (F32: no-op).
//
// act_quant_dequant_fp8x128_inplace_f32
//   The same round trip for FP8 e4m3 with a UE8M0 scale per 128 elements (the DSA indexer's FP8x128 encoding).

#include "latent_codec.cuh"

extern "C" __global__ void act_quant_dequant_inplace_f32(float* __restrict__ x, long long groups, int enc)
{
    const long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups) return;
    const int group = lc_group_size(enc);
    float* v = x + g * group;
    float vals[32];
    unsigned char out[32];
    for (int e = 0; e < group; e++) vals[e] = v[e];
    unsigned int scaleByte;
    const float scale = lc_quantize_group(enc, vals, group, out, &scaleByte);
    for (int e = 0; e < group; e++)
    {
        const float c = enc == LC_FP8 ? lc_decode_e4m3(out[e]) : lc_decode_e2m1(out[e]);
        v[e] = __fmul_rn(c, scale);
    }
}

extern "C" __global__ void act_quant_dequant_fp8x128_inplace_f32(float* __restrict__ x, long long groups)
{
    const long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups) return;
    float* v = x + g * 128;
    float amax = 0.0f;
    for (int e = 0; e < 128; e++) amax = fmaxf(amax, fabsf(v[e]));
    const unsigned int scaleByte = lc_ue8m0_byte(fmaxf(amax, 1e-4f), 1.0f / 448.0f);
    const float scale = lc_decode_ue8m0(scaleByte);
    for (int e = 0; e < 128; e++)
    {
        const float q = __fdiv_rn(v[e], scale);
        const float c = q < -448.0f ? -448.0f : (q > 448.0f ? 448.0f : q);
        v[e] = __fmul_rn(lc_decode_e4m3(lc_encode_e4m3(c)), scale);
    }
}
