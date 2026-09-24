// Fused Q2_K × F32 matrix-vector product for LLM decode (M small): the same warp-per-row scheme as
// mul_mat_vec_q4k_f32.cu, with dequant_q2_k_to_f16.cu's per-element decode inlined. Each lane owns 8
// contiguous elements of every 256-wide super-block, which all share one (half, shift group, 16-run), so a
// lane reads one scale byte and one 8-byte run of quants per super-block.
//
// Q2_K layout (84 bytes): [16 scales: 4-bit scale low nibble, 4-bit min high][64 qs: 2-bit, four per byte][d][dmin].
// x = d · (sc & 0xF) · q − dmin · (sc >> 4), element index → (h = e/128, j = (e%128)/32, half16 = (e%32)/16, l = e%16),
// scale = scales[8h + 2j + half16], byte = qs[32h + 16·half16 + l], shift = 2j.
//
// Launch: blockDim = (32, WARPS_PER_BLOCK); grid = (ceil(N / WARPS_PER_BLOCK), M).

#include <cuda_fp16.h>

#define SUPER_ELEMS 256
#define SUPER_BYTES 84
#define WARP_SIZE 32

// Byte-assembled little-endian word: a K-quant super-block is 84/110 bytes, so no wider load is aligned.
__device__ __forceinline__ unsigned int load_u32(const unsigned char* p)
{
    return (unsigned int)p[0] | ((unsigned int)p[1] << 8) | ((unsigned int)p[2] << 16) | ((unsigned int)p[3] << 24);
}

extern "C" __global__ void mul_mat_vec_q2k_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const unsigned char* __restrict__ weight,
    const float* __restrict__ bias,     // may be nullptr
    int N, int K, int M)
{
    const int lane = threadIdx.x;
    const int n = blockIdx.x * blockDim.y + threadIdx.y;
    const int m = blockIdx.y;
    if (n >= N || m >= M) return;

    const int nsb = K / SUPER_ELEMS;
    const unsigned char* wrow = weight + (size_t)n * nsb * SUPER_BYTES;
    const float* xrow = input + (size_t)m * K;

    const int e0 = lane * 8;                 // first element this lane owns within a super-block
    const int h = e0 >> 7;
    const int j = (e0 & 127) >> 5;
    const int half16 = (e0 & 31) >> 4;
    const int l0 = e0 & 15;                  // 0 or 8
    const int scaleIdx = 8 * h + 2 * j + half16;
    const int qsOff = 32 * h + 16 * half16 + l0;
    const int shift = 2 * j;

    float acc = 0.0f;
    for (int sb = 0; sb < nsb; ++sb) {
        const unsigned char* block = wrow + (size_t)sb * SUPER_BYTES;
        const float d = __half2float(*(const __half*)(block + 80));
        const float dmin = __half2float(*(const __half*)(block + 82));
        const unsigned char sc = block[scaleIdx];
        const float scale = d * (float)(sc & 0x0F);
        const float minv = dmin * (float)(sc >> 4);

        const unsigned char* q = block + 16 + qsOff;
        const unsigned int lo = load_u32(q), hi = load_u32(q + 4);
        const float* xb = xrow + sb * SUPER_ELEMS + e0;
        const float4 xa = *reinterpret_cast<const float4*>(xb);
        const float4 xb2 = *reinterpret_cast<const float4*>(xb + 4);
        const float w0 = scale * (float)((lo >> (shift     )) & 3u) - minv;
        const float w1 = scale * (float)((lo >> (shift +  8)) & 3u) - minv;
        const float w2 = scale * (float)((lo >> (shift + 16)) & 3u) - minv;
        const float w3 = scale * (float)((lo >> (shift + 24)) & 3u) - minv;
        const float w4 = scale * (float)((hi >> (shift     )) & 3u) - minv;
        const float w5 = scale * (float)((hi >> (shift +  8)) & 3u) - minv;
        const float w6 = scale * (float)((hi >> (shift + 16)) & 3u) - minv;
        const float w7 = scale * (float)((hi >> (shift + 24)) & 3u) - minv;
        acc += w0 * xa.x + w1 * xa.y + w2 * xa.z + w3 * xa.w
             + w4 * xb2.x + w5 * xb2.y + w6 * xb2.z + w7 * xb2.w;
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) {
        if (bias != nullptr) acc += bias[n];
        output[(size_t)m * N + n] = acc;
    }
}
