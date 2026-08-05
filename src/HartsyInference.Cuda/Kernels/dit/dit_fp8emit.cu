// Producer kernels that emit an fp8 (e4m3) copy of their output alongside the F32 one, so the
// consuming fp8 Linear can skip its own quantize pass entirely.
//
// Own module, like dit_rope.cu: the committed dit_f32.ptx was built by a different nvcc than the one
// available here, so rebuilding that file would rewrite codegen for its other 40 kernels.
//
// Only legal because the checkpoint ships a STATIC per-Linear activation scale. With the old dynamic
// absmax the scale is not known until a grid-wide reduction over the finished tensor has retired, so
// a producer could not quantize from registers without a second pass or a device sync.
//
// The F32 output path is byte-identical to dit_affine_broadcast_rowindexed_f32 — same expressions in
// the same order — so the unfused result is unchanged and only the sidecar is new.

// VERBATIM COPY of fp8_quant.cu's f32_to_e4m3. It must stay byte-identical: the fused sidecar has to
// encode exactly what the standalone quantize kernel would, or enabling fusion silently changes results.
// If that file's encoder ever changes, this one has to change with it.
__device__ __forceinline__ unsigned char f32_to_e4m3_emit(float f)
{
    unsigned char sign = (unsigned char)((__float_as_uint(f) >> 24) & 0x80u);
    float a = fabsf(f);
    if (!(a == a)) return (unsigned char)(sign | 0x7Fu);
    if (a >= 464.0f) return (unsigned char)(sign | 0x7Eu);
    if (a < 0.0009765625f) return sign;
    int e;
    float m = frexpf(a, &e);
    if (e - 1 >= -6)
    {
        int q = (int)rintf(m * 16.0f);
        if (q == 16) { q = 8; e += 1; }
        int expField = (e - 1) + 7;
        if (expField >= 16) return (unsigned char)(sign | 0x7Eu);
        return (unsigned char)(sign | (expField << 3) | (q - 8));
    }
    int q = (int)rintf(a * 512.0f);
    if (q >= 8) return (unsigned char)(sign | (1 << 3));
    return (unsigned char)(sign | q);
}

// Writes ONLY the e4m3 result. The F32 modulated tensor was read by exactly one consumer -- the fp8
// Linear that immediately follows -- so materializing it cost a 141 MB write plus a 141 MB read per
// site on MiniMax-H3, to hand the quantize kernel something it would compress to 35 MB anyway.
extern "C" __global__ void dit_affine_broadcast_rowindexed_to_fp8_f32(
    unsigned char* __restrict__ outputFp8,
    const float* __restrict__ input,
    const float* __restrict__ scaleTable,
    const float* __restrict__ shiftTable,
    const int* __restrict__ rowIndex,
    const float* __restrict__ inputScale,
    unsigned int dim,
    unsigned long long total)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;

    unsigned int d = (unsigned int)(i % dim);
    unsigned long long row = i / dim;
    size_t tIdx = (size_t)rowIndex[row] * dim + d;

    float v = input[i] * (1.0f + scaleTable[tIdx]);
    if (shiftTable != 0) v += shiftTable[tIdx];
    // `1.0f / scale[0]` computed here, not passed in pre-inverted: quant_f32_e4m3 does exactly this, and
    // inverting on the host could differ in the last bit and desync the two encodings.
    float rs = 1.0f / inputScale[0];
    outputFp8[i] = f32_to_e4m3_emit(v * rs);
}
