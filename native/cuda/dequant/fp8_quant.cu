// Per-tensor dynamic FP8 activation quantization for the native cuBLASLt FP8 GEMM path (Ada+).
// Three kernels: two-pass absmax reduction, then a fused scale-and-quantize F32 -> e4m3fn.
// The dequant scale (amax/448) is written to DEVICE memory and consumed by cublasLtMatmul via
// CUBLASLT_MATMUL_DESC_B_SCALE_POINTER — the whole quantize+GEMM chain stays async (no host sync).
// The e4m3 conversion is hand-rolled bit math (no sm_89-only cvt instructions) so this PTX still
// JITs on Ampere for the unit tests; the GEMM itself is gated on SM 8.9+ elsewhere.

#define REDUCE_THREADS 256u

// Pass 1: grid-strided per-block |max| -> blockMax[blockIdx].
extern "C" __global__ void absmax_f32(
    const float* __restrict__ x, float* __restrict__ blockMax, unsigned int n)
{
    __shared__ float sm[REDUCE_THREADS];
    unsigned int tid = threadIdx.x;
    float m = 0.0f;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + tid; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x)
    {
        float v = fabsf(x[i]);
        if (v > m) m = v;
    }
    sm[tid] = m;
    __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (tid < s && sm[tid + s] > sm[tid]) sm[tid] = sm[tid + s];
        __syncthreads();
    }
    if (tid == 0) blockMax[blockIdx.x] = sm[0];
}

// Pass 2 (single block): blockMax[0..numBlocks) -> scale[0] = amax/448 (the e4m3 DEQUANT scale).
// amax==0 (all-zero tensor) writes scale 1.0 so the GEMM stays a well-defined no-op.
extern "C" __global__ void absmax_finalize_scale(
    const float* __restrict__ blockMax, unsigned int numBlocks, float* __restrict__ scale)
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
    if (tid == 0) scale[0] = sm[0] > 0.0f ? sm[0] / 448.0f : 1.0f;
}

// float -> e4m3fn (bias 7, max normal 448, no inf; 0x7F/0xFF = NaN). Round-half-away-from-zero on
// the mantissa (vs the IEEE ties-to-even a hardware cvt would do) — a <=0.5-ulp difference on exact
// ties only, irrelevant for activation quantization. Values past the 448+16 rounding midpoint clamp
// to +-448 (satfinite semantics: quantization must never emit NaN).
__device__ __forceinline__ unsigned char f32_to_e4m3(float f)
{
    unsigned char sign = (unsigned char)((__float_as_uint(f) >> 24) & 0x80u);
    float a = fabsf(f);
    if (!(a == a)) return (unsigned char)(sign | 0x7Fu);       // NaN propagates as e4m3 NaN
    if (a >= 464.0f) return (unsigned char)(sign | 0x7Eu);     // clamp to max normal 448
    if (a < 0.0009765625f) return sign;                        // < 2^-10 = half of min subnormal -> +-0
    int e;
    float m = frexpf(a, &e);                                   // a = m * 2^e, m in [0.5, 1)
    if (e - 1 >= -6)                                           // normal: value = (2m) * 2^(e-1), 2m in [1,2)
    {
        int q = (int)rintf(m * 16.0f);                         // round mantissa to 4 bits (8..16)
        if (q == 16) { q = 8; e += 1; }                        // mantissa overflow -> bump exponent
        int expField = (e - 1) + 7;                            // e4m3 bias 7
        if (expField >= 16) return (unsigned char)(sign | 0x7Eu);
        return (unsigned char)(sign | (expField << 3) | (q - 8));
    }
    // subnormal: value = q * 2^-9, q in [0, 8); q==8 rolls into the smallest normal (2^-6).
    int q = (int)rintf(a * 512.0f);
    if (q >= 8) return (unsigned char)(sign | (1 << 3));
    return (unsigned char)(sign | q);
}

// out[i] = e4m3(x[i] * 448/amax), reading the dequant scale written by absmax_finalize_scale.
extern "C" __global__ void quant_f32_e4m3(
    const float* __restrict__ x, unsigned char* __restrict__ out,
    const float* __restrict__ scale, unsigned int n)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float rs = 1.0f / scale[0];
    out[i] = f32_to_e4m3(x[i] * rs);
}
