// Fused Q6_K × Q8_1 matrix-vector product for LLM decode, using __dp4a int8 dot products.
//
// Q6_K is the lm_head quant type on K-quant models (the single largest GEMV, run every token) and
// half the ffn_down/attn_v layers in Q4_K_M schemes. Same approach as mul_mat_vec_q4k_q8_1: the
// activation is pre-quantized to int8 (Q8_1, per-32-block scale), the 6-bit weights are unpacked
// four-at-a-time into a SIMD int (values 0..63), shifted to signed via per-byte subtract 32
// (__vsub4), and dotted with __dp4a. Q6_K scales are signed per-16-element int8 (symmetric, no min
// term), and a lane's 4 consecutive elements never span a 16-group, so each dp4a result is scaled
// by exactly one (d * sc):
//
//   dot_group = d * sc[g] * Σ (q[i]-32)·xq[i]
//
// Q6_K super-block: 256 elements / 210 bytes:
//   [128 B ql (low 4 bits)] [64 B qh (high 2 bits, 4/byte)] [16 B int8 scales] [2 B fp16 d]
// Element mapping (canonical ggml, half n∈{0,1}, l∈0..31, quadrant q∈0..3 at elem n*128+q*32+l):
//   ql byte = ql[n*64 + (q&1)*32 + l], nibble = (q>>1) ? high : low; qh bits = qh[n*32+l] >> (2q).
//
// Layout: one WARP per output row; 32 lanes = 2 halves × 2 ql-pairs × 8 u16-pair positions.
// Each lane loads its 4 ql bytes ONCE and processes both of their nibble planes (quadrants p and
// p+2, 2 dp4a / lane / super-block) — a plane-per-lane split would load every ql byte twice.
// 210-byte blocks are only 2-byte aligned → u16 assembly loads.
//
// weight [N,K] Q6_K row-major; xq [M,K] int8; xd [M,K/32] F32.
// Launch: blockDim = (32, WARPS_PER_BLOCK); grid = (ceil(N/WARPS_PER_BLOCK), M). Requires K % 256 == 0.

#include <cuda_fp16.h>

#define SUPER_ELEMS 256
#define SUPER_BYTES 210
#define WARP_SIZE 32

__device__ __forceinline__ int load_int_2aligned(const unsigned char* p)
{
    const unsigned short* p16 = (const unsigned short*)p;
    return (int)((unsigned int)p16[0] | ((unsigned int)p16[1] << 16));
}

// Per-lane partial dot over super-blocks sbStart, sbStart+sbStride, … of one output row.
__device__ __forceinline__ float q6k_q8_1_row_partial(
    const unsigned char* __restrict__ wrow,
    const signed char* __restrict__ xqrow,
    const float* __restrict__ xdrow,
    int nsb, int sbStart, int sbStride, int lane)
{
    const int li = lane & 7;               // u16-pair position (4 consecutive bytes)
    const int p = (lane >> 3) & 1;         // ql pair: quadrants p (lo nibble) and p+2 (hi nibble)
    const int h = lane >> 4;               // half-super-block 0..1
    const int qlOff = h * 64 + p * 32 + li * 4;
    const int qhOff = h * 32 + li * 4;
    const int elemOff0 = h * 128 + p * 32 + li * 4;
    const int elemOff1 = elemOff0 + 64;    // quadrant p+2
    const int scOff0 = h * 8 + p * 2 + (li >> 2);
    const int scOff1 = scOff0 + 4;         // (p+2)*2 − p*2

    float acc = 0.0f;
    for (int sb = sbStart; sb < nsb; sb += sbStride) {
        const unsigned char* block = wrow + (size_t)sb * SUPER_BYTES;
        const unsigned char* qh = block + 128;
        const signed char* scales = (const signed char*)(block + 192);
        const float d = __half2float(*(const __half*)(block + 208));
        const signed char* xb = xqrow + sb * SUPER_ELEMS;
        const float* xdb = xdrow + sb * 8;

        // Issue every load up front (same latency-hiding structure the float Q6_K kernel needed —
        // see mul_mat_vec_q6k_f32.cu), then unpack and accumulate.
        const int qlw = load_int_2aligned(block + qlOff);
        const int qhw = load_int_2aligned(qh + qhOff);
        const signed char sc0 = scales[scOff0];
        const signed char sc1 = scales[scOff1];
        const int xv0 = *reinterpret_cast<const int*>(xb + elemOff0);
        const int xv1 = *reinterpret_cast<const int*>(xb + elemOff1);
        const float xs0 = xdb[elemOff0 >> 5];
        const float xs1 = xdb[elemOff1 >> 5];

        const int packed0 = ((unsigned int)qlw & 0x0F0F0F0F)
                          | ((((unsigned int)qhw >> (2 * p)) & 0x03030303) << 4);
        const int packed1 = (((unsigned int)qlw >> 4) & 0x0F0F0F0F)
                          | ((((unsigned int)qhw >> (2 * p + 4)) & 0x03030303) << 4);
        const int q0 = __vsub4(packed0, 0x20202020);   // 0..63 → signed −32..31 per byte
        const int q1 = __vsub4(packed1, 0x20202020);

        const int idot0 = __dp4a(q0, xv0, 0);
        const int idot1 = __dp4a(q1, xv1, 0);
        acc += d * (float)sc0 * xs0 * (float)idot0;
        acc += d * (float)sc1 * xs1 * (float)idot1;
    }
    return acc;
}

extern "C" __global__ void mul_mat_vec_q6k_q8_1(
    float* __restrict__ output,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    const unsigned char* __restrict__ weight,
    const float* __restrict__ bias,     // may be nullptr
    int N, int K, int M)
{
    const int lane = threadIdx.x;                          // 0..31
    const int n = blockIdx.x * blockDim.y + threadIdx.y;   // output row
    const int m = blockIdx.y;                              // batch row
    if (n >= N || m >= M) return;

    const int nsb = K / SUPER_ELEMS;
    float acc = q6k_q8_1_row_partial(
        weight + (size_t)n * nsb * SUPER_BYTES,
        xq + (size_t)m * K, xd + (size_t)m * (K / 32),
        nsb, 0, 1, lane);

    #pragma unroll
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) {
        if (bias != nullptr) acc += bias[n];
        output[(size_t)m * N + n] = acc;
    }
}

// Block-per-row K-split (long-K/small-N shapes; see mul_mat_vec_q4k_q8_1.cu's ksplit notes).
extern "C" __global__ void mul_mat_vec_q6k_q8_1_ksplit(
    float* __restrict__ output,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    const unsigned char* __restrict__ weight,
    const float* __restrict__ bias,     // may be nullptr
    int N, int K, int M)
{
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int n = blockIdx.x;
    const int m = blockIdx.y;
    if (n >= N || m >= M) return;

    const int nsb = K / SUPER_ELEMS;
    float acc = q6k_q8_1_row_partial(
        weight + (size_t)n * nsb * SUPER_BYTES,
        xq + (size_t)m * K, xd + (size_t)m * (K / 32),
        nsb, warp, blockDim.y, lane);

    #pragma unroll
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);

    __shared__ float partial[16];
    if (lane == 0) partial[warp] = acc;
    __syncthreads();
    if (warp == 0 && lane == 0) {
        float sum = 0.0f;
        for (int w = 0; w < (int)blockDim.y; ++w) sum += partial[w];
        if (bias != nullptr) sum += bias[n];
        output[(size_t)m * N + n] = sum;
    }
}
