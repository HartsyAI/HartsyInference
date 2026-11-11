// y = x · Wᵀ with W an MXFP4 weight kept packed: E2M1 nibbles (low nibble the even column) and one E8M0 scale per 32 columns, element
// w = E2M1(nibble) · 2^(scale − 127). One block per output row; its threads stride the columns for every input row, then reduce.
// Matches Mxfp4LinearReference.Apply (Core). Sums are tree-ordered, so values agree with the sequential reference to rounding.

#define MXFP4_LINEAR_THREADS 256

__device__ __constant__ float kE2M1Magnitude[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

extern "C" {

__global__ void mxfp4_linear_f32(
    float* __restrict__ output,            // [n, rows]
    const float* __restrict__ input,       // [n, cols]
    const unsigned char* __restrict__ packed,  // [rows, cols / 2]
    const unsigned char* __restrict__ scales,  // [rows, cols / 32]
    unsigned int n,
    unsigned int rows,
    unsigned int cols)
{
    __shared__ float partial[MXFP4_LINEAR_THREADS];
    unsigned int o = blockIdx.x;
    if (o >= rows) return;
    unsigned int tid = threadIdx.x;
    unsigned int blocksPerRow = cols / 32u;
    for (unsigned int r = 0; r < n; r++)
    {
        float acc = 0.0f;
        for (unsigned int c = tid; c < cols; c += MXFP4_LINEAR_THREADS)
        {
            unsigned char b = packed[(size_t)o * (cols / 2u) + c / 2u];
            unsigned int nib = (c & 1u) ? (b >> 4) : (b & 0xFu);
            float mag = kE2M1Magnitude[nib & 7u];
            float w = (nib & 8u) ? -mag : mag;
            float scale = exp2f((float)scales[(size_t)o * blocksPerRow + c / 32u] - 127.0f);
            acc += input[(size_t)r * cols + c] * w * scale;
        }
        partial[tid] = acc;
        __syncthreads();
        for (unsigned int s = MXFP4_LINEAR_THREADS / 2; s > 0; s >>= 1)
        {
            if (tid < s) partial[tid] += partial[tid + s];
            __syncthreads();
        }
        if (tid == 0) output[(size_t)r * rows + o] = partial[0];
        __syncthreads();
    }
}

}  // extern "C"
