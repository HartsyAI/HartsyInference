// RMSNorm, ReLU and a scale in one pass (IBackend.RmsNormRelu), F32: out = max(x * rsqrt(mean(x^2) + eps) * w, 0) * scale
// per row. One warp per row: the sum of squares is a warp-tree reduction, then the row (in L1 from the first read) is
// written once. The default implementation composes RmsNorm, LeakyRelu and Scale.
//
// Build: src/HartsyInference.Cuda/Kernels/dit/build.sh.

__device__ __forceinline__ float rr_warp_sum(float v)
{
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xffffffffu, v, offset);
    return v;
}

// blockDim.x = 32 * rowsPerBlock; warp w of block b takes row b * rowsPerBlock + w.
extern "C" __global__ void rms_norm_relu_f32(float* __restrict__ out, const float* __restrict__ in, const float* __restrict__ weight,
    unsigned int rows, unsigned int width, float eps, float scale)
{
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned long long row = (unsigned long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= rows) return;
    const float* x = in + row * width;
    float* y = out + row * width;
    float sum = 0.0f;
    for (unsigned int i = lane; i < width; i += 32u)
    {
        float v = x[i];
        sum += v * v;
    }
    sum = rr_warp_sum(sum);
    const float inv = rsqrtf(sum / (float)width + eps);
    for (unsigned int i = lane; i < width; i += 32u)
        y[i] = fmaxf(x[i] * inv * weight[i], 0.0f) * scale;
}
