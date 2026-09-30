// Batched 2-D transpose [B, D1, D2] -> [B, D2, D1] through 32x32 shared-memory tiles, coalesced on both the read and
// the write. The layout step around cuDNN's channels-last convolution engines: NCHW activations ([N, C, HW]) become
// NHWC ([N, HW, C]) and back, KCRS weights become KRSC. The element is moved as raw bits, so the 16-bit kernel serves
// F16 and BF16 alike.
//
// Grid (ceil(D2/32), ceil(D1/32), B), block (32, 8): each thread moves 4 elements per phase. The +1 column pad keeps
// the column read of the tile free of bank conflicts.
//
// Build: nvcc -ptx -arch=sm_80 channels_last.cu (PTX must say .version 9.0).

#define CL_TILE 32
#define CL_ROWS 8

template <typename T>
__device__ __forceinline__ void cl_transpose(T* __restrict__ out, const T* __restrict__ in,
    unsigned int d1, unsigned int d2)
{
    __shared__ T tile[CL_TILE][CL_TILE + 1];
    const unsigned long long plane = (unsigned long long)d1 * d2;
    const T* src = in + blockIdx.z * plane;
    T* dst = out + blockIdx.z * plane;
    const unsigned int c0 = blockIdx.x * CL_TILE, r0 = blockIdx.y * CL_TILE;

    for (unsigned int r = threadIdx.y; r < CL_TILE; r += CL_ROWS)
    {
        const unsigned int row = r0 + r, col = c0 + threadIdx.x;
        if (row < d1 && col < d2) tile[r][threadIdx.x] = src[(unsigned long long)row * d2 + col];
    }
    __syncthreads();
    for (unsigned int r = threadIdx.y; r < CL_TILE; r += CL_ROWS)
    {
        const unsigned int row = c0 + r, col = r0 + threadIdx.x;   // output is [D2][D1]
        if (row < d2 && col < d1) dst[(unsigned long long)row * d1 + col] = tile[threadIdx.x][r];
    }
}

extern "C" __global__ void transpose_tiled_b16(unsigned short* __restrict__ out, const unsigned short* __restrict__ in,
    unsigned int d1, unsigned int d2)
{
    cl_transpose<unsigned short>(out, in, d1, d2);
}

extern "C" __global__ void transpose_tiled_b32(unsigned int* __restrict__ out, const unsigned int* __restrict__ in,
    unsigned int d1, unsigned int d2)
{
    cl_transpose<unsigned int>(out, in, d1, d2);
}
