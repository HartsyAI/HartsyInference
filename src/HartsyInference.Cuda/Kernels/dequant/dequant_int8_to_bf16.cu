// int8_tensorwise rows -> BF16: I8 [rows, cols] x one F32 scale per row, so a layer that needs a float GEMM
// (full_precision_matrix_mult) keeps its weight resident at one byte per parameter and unpacks it transiently. Same
// values as HartsyInference.Core.Tensors.Int8ConvRotCodec.DequantToBf16 without rotation: the product is formed in F32
// and the BF16 store truncates, bit for bit. Rotated (ConvRot) weights are not served here.

extern "C" __global__ void dequant_int8_to_bf16(
    const signed char* __restrict__ w, const float* __restrict__ rowScale, unsigned short* __restrict__ out,
    unsigned int rows, unsigned int cols)
{
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;
    for (unsigned int row = blockIdx.y; row < rows; row += gridDim.y)
    {
        unsigned long long i = (unsigned long long)row * cols + col;
        out[i] = (unsigned short)(__float_as_uint((float)w[i] * rowScale[row]) >> 16);
    }
}
