// Linear over an FP8 e4m3 weight with F32 block scales, dequantized in the product (see IBackend.Fp8BlockLinear):
//   y[m, n] = sum_k x[m, k] * e4m3(w[n, k]) * scale[n / blockRows, k / blockCols] + bias[n]
// and its block-diagonal form (IBackend.BlockDiagonalLinear), where output n reads only the input slice of its group:
//   y[m, n] = sum_k x[m, (n / groupRows) * K + k] * w(n, k)      with w an F32 weight or the FP8 decode above.
// One block of 128 threads per output column n; the block loops over the rows m and reduces each dot product in shared
// memory. The weight stays packed in device memory; each element is decoded where it is read.

#define FP8_LINEAR_THREADS 128

__device__ __forceinline__ float fp8_linear_decode_e4m3(unsigned int code)
{
    const int e = (code >> 3) & 0xF, m = code & 7;
    if (e == 0xF && m == 7) return __int_as_float(0x7fc00000);
    const float sign = (code & 0x80u) ? -1.0f : 1.0f;
    if (e == 0) return __fmul_rn(__fmul_rn(sign, (float)m), 0.001953125f);
    return __fmul_rn(sign, __int_as_float(((e + 120) << 23) | (m << 20)));
}

// FP8 decodes a byte weight with its block scale; otherwise the weight is F32.
template <bool FP8>
__device__ __forceinline__ void block_linear(const float* x, const void* w, const float* scale, const float* bias, float* y,
                                             int M, int N, int K, int blockRows, int blockCols, int gridCols, int inWidth, int groupRows)
{
    __shared__ float partial[FP8_LINEAR_THREADS];
    const int n = blockIdx.x;
    const unsigned char* wRow8 = (const unsigned char*)w + (long long)n * K;
    const float* wRow32 = (const float*)w + (long long)n * K;
    const float* scaleRow = FP8 ? scale + (long long)(n / blockRows) * gridCols : nullptr;
    const long long xOffset = (long long)(n / groupRows) * K;
    const float biasN = bias == nullptr ? 0.0f : bias[n];
    for (int m = blockIdx.y; m < M; m += gridDim.y)
    {
        const float* xRow = x + (long long)m * inWidth + xOffset;
        float acc = 0.0f;
        for (int k = threadIdx.x; k < K; k += FP8_LINEAR_THREADS)
        {
            const float wv = FP8 ? __fmul_rn(fp8_linear_decode_e4m3(wRow8[k]), scaleRow[k / blockCols]) : wRow32[k];
            acc = __fadd_rn(acc, __fmul_rn(xRow[k], wv));
        }
        partial[threadIdx.x] = acc;
        __syncthreads();
        for (int s = FP8_LINEAR_THREADS / 2; s > 0; s >>= 1)
        {
            if (threadIdx.x < s) partial[threadIdx.x] = __fadd_rn(partial[threadIdx.x], partial[threadIdx.x + s]);
            __syncthreads();
        }
        if (threadIdx.x == 0) y[(long long)m * N + n] = __fadd_rn(partial[0], biasN);
        __syncthreads();
    }
}

extern "C" __global__ void fp8_block_linear_f32(const float* x, const unsigned char* w, const float* scale, const float* bias,
                                                float* y, int M, int N, int K, int blockRows, int blockCols, int gridCols)
{
    block_linear<true>(x, w, scale, bias, y, M, N, K, blockRows, blockCols, gridCols, K, N);
}

extern "C" __global__ void block_diag_linear_fp8_f32(const float* x, const unsigned char* w, const float* scale, float* y, int M,
                                                     int N, int K, int blockRows, int blockCols, int gridCols, int inWidth, int groupRows)
{
    block_linear<true>(x, w, scale, nullptr, y, M, N, K, blockRows, blockCols, gridCols, inWidth, groupRows);
}

extern "C" __global__ void block_diag_linear_f32(const float* x, const float* w, float* y, int M, int N, int K, int inWidth,
                                                 int groupRows)
{
    block_linear<false>(x, w, nullptr, nullptr, y, M, N, K, 1, 1, 0, inWidth, groupRows);
}
