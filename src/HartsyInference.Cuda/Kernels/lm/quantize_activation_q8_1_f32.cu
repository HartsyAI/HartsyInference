// Quantize an F32 activation vector to int8 with a per-32-block scale + int-sum (Q8_1-style),
// so the decode GEMV can dot it against quantized weights via __dp4a (int8) instead of per-element
// float dequant. Producing this once per Linear input lets every output row reuse it.
//
//   x   : [M, K] F32
//   xq  : [M, K] int8            (x[i] ≈ xd[block] * xq[i])
//   xd  : [M, K/32] F32          per-32-block scale = amax/127
//   xs  : [M, K/32] F32          sum of the block's int8 values (for the Q4_K min term)
//
// Launch: one warp (32 lanes) per 32-block, 8 warps per CUDA block;
// grid = ceil(M*K/32 / 8), blockDim = (32, 8). Warps are independent (no barriers).

extern "C" __global__ void quantize_activation_q8_1_f32(
    signed char* __restrict__ xq,
    float* __restrict__ xd,
    float* __restrict__ xs,
    const float* __restrict__ x,
    int nblocks)
{
    const int blk = blockIdx.x * blockDim.y + threadIdx.y;
    if (blk >= nblocks) return;
    const int lane = threadIdx.x;               // 0..31

    const float v = x[(size_t)blk * 32 + lane];

    // Warp reduce max |v|.
    float amax = fabsf(v);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));

    const float scale = amax / 127.0f;
    const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
    int q = __float2int_rn(v * inv);
    q = max(-127, min(127, q));
    xq[(size_t)blk * 32 + lane] = (signed char)q;

    // Warp reduce sum of the int8 values.
    int s = q;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);

    if (lane == 0) { xd[blk] = scale; xs[blk] = (float)s; }
}
