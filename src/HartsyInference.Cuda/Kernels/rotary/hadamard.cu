// Walsh-Hadamard transform of each row over the last dimension (Sylvester, natural order), one butterfly stage per launch.
//
// hadamard_stage_f32 handles the pairs (j, j + h) inside each row. src may equal dst: pairs of one stage are disjoint and
// the launches of successive stages are ordered on the stream. The launcher passes scale on the last stage only, with 1
// on the others, so the product of all stages is scale * (input @ H).
// hadamard_scale_f32 covers width 1, which has no stages.

extern "C" __global__ void hadamard_stage_f32(const float* src, float* dst, long long pairs, int width, int h, float scale)
{
    const long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pairs) return;
    const long long half = width >> 1;
    const long long row = p / half;
    const long long q = p - row * half;
    const long long j = (q / h) * (2LL * h) + (q % h);
    const long long base = row * width;
    const float a = src[base + j];
    const float b = src[base + j + h];
    dst[base + j] = __fmul_rn(__fadd_rn(a, b), scale);
    dst[base + j + h] = __fmul_rn(__fsub_rn(a, b), scale);
}

extern "C" __global__ void hadamard_scale_f32(const float* src, float* dst, long long n, float scale)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __fmul_rn(src[i], scale);
}
