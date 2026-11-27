// int8_tensorwise rows -> BF16: I8 [rows, cols] x one F32 scale per row, so a layer that needs a float GEMM
// (full_precision_matrix_mult) keeps its weight resident at one byte per parameter and unpacks it transiently.
//
// Two rounding modes, chosen by the weight (QuantWeightInfo.ReferenceRounding):
//   0: the product is formed in F32 and the BF16 store truncates, bit for bit as Int8ConvRotCodec.DequantToBf16 stores it.
//   1: the reference's bf16 QuantizedLinear.forward (longcat_video/modules/quantization.py @ 58eb336, lines 27-31): the level
//      and the scale are BF16 (the level is exact; the scale rounds to nearest-even), and the BF16 product rounds to
//      nearest-even. The product of an 8-bit level and an 8-bit scale fits in F32 exactly, so the rounding is the only step.
// Rotated (ConvRot) weights are not served here.

__device__ __forceinline__ unsigned int bf16_round_nearest_even(unsigned int u)
{
    // u is a float's bits; the result is the BF16 bits (upper half), NaNs stay NaNs.
    if ((u & 0x7fffffffu) > 0x7f800000u) return (u >> 16) | 0x40u;
    return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}

extern "C" __global__ void dequant_int8_to_bf16(
    const signed char* __restrict__ w, const float* __restrict__ rowScale, unsigned short* __restrict__ out,
    unsigned int rows, unsigned int cols, int referenceRounding)
{
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;
    for (unsigned int row = blockIdx.y; row < rows; row += gridDim.y)
    {
        unsigned long long i = (unsigned long long)row * cols + col;
        if (referenceRounding)
        {
            float scaleBf16 = __uint_as_float(bf16_round_nearest_even(__float_as_uint(rowScale[row])) << 16);
            out[i] = (unsigned short)bf16_round_nearest_even(__float_as_uint((float)w[i] * scaleBf16));
        }
        else
        {
            out[i] = (unsigned short)(__float_as_uint((float)w[i] * rowScale[row]) >> 16);
        }
    }
}
