// Block-scaled activation quantization for the native Blackwell GEMM path (BlockScaledGemmExecutor): an F32 or F16
// [rows, cols] activation -> packed e2m1 [rows, cols/2] + one E4M3 scale per 16-element block in cuBLASLt's blocked
// layout + the per-tensor scalars the GEMM reads as alpha/beta. Two passes like fp8_quant: absmax (fp8_quant's own
// kernels) -> finalize -> quantize, all on the stream. The per-tensor scale lands in DEVICE memory and the GEMM
// reads alpha from there (pointer mode DEVICE), so dynamic quantization costs no host sync.
//
// Nibble order matches the weights (HIGH nibble = even element, dequant_nvfp4_to_f16.cu). cuBLASLt's own
// convention does not matter as long as both operands agree: the dot product sums over K, so swapping elements
// 2i and 2i+1 in both operands leaves every product in place, and a pair never straddles a 16-element block.
//
// NVFP4 recipe (TensorRT-LLM / comfy.float): sf = amax / (448·6) is the per-tensor DEQUANT scale; a block's E4M3
// scale is block_amax / (6·sf), which cannot exceed 448; an element is e2m1(x / (e4m3(scale)·sf)). Dequantization
// is then e2m1 · e4m3 · sf — the same left-to-right product the weight side forms.
#include "block_scale.cuh"

#define REDUCE_THREADS 256u

// Pass 2 (single block): blockMax[0..numBlocks) -> scalars[0] = sf (1.0 for an all-zero tensor),
// scalars[1] = alpha = weightScale · sf, scalars[2] = 0 (beta, read from device beside alpha).
extern "C" __global__ void block_quant_finalize_nvfp4(
    const float* __restrict__ blockMax, unsigned int numBlocks, float weightScale, float* __restrict__ scalars)
{
    __shared__ float sm[REDUCE_THREADS];
    unsigned int tid = threadIdx.x;
    float m = 0.0f;
    for (unsigned int i = tid; i < numBlocks; i += blockDim.x)
    {
        float v = blockMax[i];
        if (v > m) m = v;
    }
    sm[tid] = m;
    __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (tid < s && sm[tid + s] > sm[tid]) sm[tid] = sm[tid + s];
        __syncthreads();
    }
    if (tid == 0)
    {
        float sf = sm[0] > 0.0f ? sm[0] / (448.0f * 6.0f) : 1.0f;
        scalars[0] = sf;
        scalars[1] = weightScale * sf;
        scalars[2] = 0.0f;
    }
}

// Pass 3: one thread per 16-element block — its amax -> its E4M3 scale (blocked layout) -> one 8-byte store.
#define BLOCK_QUANT_NVFP4_BODY(LOAD)                                                                       \
    unsigned long long block = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;                   \
    unsigned int blocksPerRow = cols >> 4;                                                                  \
    if (block >= (unsigned long long)rows * blocksPerRow) return;                                           \
    unsigned int row = (unsigned int)(block / blocksPerRow);                                                \
    unsigned int blockCol = (unsigned int)(block % blocksPerRow);                                           \
    unsigned long long base = (unsigned long long)row * cols + (unsigned long long)blockCol * 16u;          \
    float v[16];                                                                                            \
    float amax = 0.0f;                                                                                      \
    for (unsigned int i = 0; i < 16u; i++)                                                                  \
    {                                                                                                       \
        v[i] = LOAD(base + i);                                                                              \
        float a = fabsf(v[i]);                                                                              \
        if (a > amax) amax = a;                                                                             \
    }                                                                                                       \
    float sf = scalars[0];                                                                                  \
    unsigned char scaleByte = f32_to_e4m3(amax / (6.0f * sf));                                              \
    blockScale[swizzled_scale_index(row, blockCol, paddedCols)] = scaleByte;                                \
    float dq = nvfp4_e4m3_decode(scaleByte) * sf;                                                           \
    float inv = dq > 0.0f ? 1.0f / dq : 0.0f;                                                               \
    unsigned long long packed = 0ull;                                                                       \
    for (unsigned int i = 0; i < 16u; i += 2u)                                                              \
        packed |= (unsigned long long)e2m1x2_pack(v[i] * inv, v[i + 1] * inv) << (4u * i);                  \
    *(unsigned long long*)(out + (base >> 1)) = packed;

#define LOAD_F32(I) x[(I)]
#define LOAD_F16(I) h16_to_f32(x[(I)])

extern "C" __global__ void block_quant_nvfp4_f32(
    const float* __restrict__ x, unsigned char* __restrict__ out, unsigned char* __restrict__ blockScale,
    const float* __restrict__ scalars, unsigned int rows, unsigned int cols, unsigned int paddedCols)
{
    BLOCK_QUANT_NVFP4_BODY(LOAD_F32)
}

extern "C" __global__ void block_quant_nvfp4_f16(
    const unsigned short* __restrict__ x, unsigned char* __restrict__ out, unsigned char* __restrict__ blockScale,
    const float* __restrict__ scalars, unsigned int rows, unsigned int cols, unsigned int paddedCols)
{
    BLOCK_QUANT_NVFP4_BODY(LOAD_F16)
}
