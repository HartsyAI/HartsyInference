// Latent-cache quantization kernels: FP8 e4m3 + ue8m0/32, FP4 e2m1 + e4m3/16 and FP4 e2m1 + ue8m0/32, matching
// the reference act_quant / fp4_act_quant (see latent_codec.cuh).
//
// latent_row_winner_init / latent_row_winner_mark / latent_quantize_rows_f32
//   QuantizeLatentRows writes source row i into cache row physicalRows[i]. Negative destinations are skipped and
//   destinations past the last row are ignored (the CPU reference throws). When several source rows share a
//   destination the LAST one wins, as in the sequential reference: winner[p] = atomicMax(i) is order independent,
//   so the result is deterministic. One thread per (source row, scale group); F32 caches copy one element per thread.
//
// act_quant_dequant_inplace_f32
//   Replaces every value with its quantize-dequantize round trip, one thread per scale group (F32: no-op).

#include "latent_codec.cuh"

extern "C" __global__ void latent_row_winner_init(int* __restrict__ winner, int destRows)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < destRows) winner[r] = -1;
}

extern "C" __global__ void latent_row_winner_mark(int* __restrict__ winner, const int* __restrict__ physicalRows,
                                                  int count, int destRows)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int p = physicalRows[i];
    if (p >= 0 && p < destRows) atomicMax(&winner[p], i);
}

extern "C" __global__ void latent_quantize_rows_f32(
    unsigned char* __restrict__ codes,
    unsigned char* __restrict__ scales,
    const float* __restrict__ rows,
    const int* __restrict__ physicalRows,
    const int* __restrict__ winner,
    int count, int dim, int enc, int destRows)
{
    const int group = lc_group_size(enc);
    const int units = group == 0 ? dim : dim / group;
    const long long u = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= (long long)count * units) return;
    const int i = (int)(u / units), g = (int)(u % units);
    const int p = physicalRows[i];
    if (p < 0 || p >= destRows || winner[p] != i) return;
    if (enc == LC_F32)
    {
        ((float*)codes)[(long long)p * dim + g] = rows[(long long)i * dim + g];
        return;
    }
    float vals[32];
    unsigned char out[32];
    for (int e = 0; e < group; e++) vals[e] = rows[(long long)i * dim + g * group + e];
    unsigned int scaleByte;
    lc_quantize_group(enc, vals, group, out, &scaleByte);
    scales[(long long)p * units + g] = (unsigned char)scaleByte;
    if (enc == LC_FP8)
    {
        unsigned char* dst = codes + (long long)p * dim + g * group;
        for (int e = 0; e < group; e++) dst[e] = out[e];
    }
    else
    {
        unsigned char* dst = codes + (long long)p * (dim >> 1) + ((g * group) >> 1);
        for (int e = 0; e < group; e += 2) dst[e >> 1] = (unsigned char)(out[e] | (out[e + 1] << 4));
    }
}

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
