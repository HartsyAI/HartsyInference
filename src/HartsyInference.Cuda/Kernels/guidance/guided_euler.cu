// The guided Euler step (IBackend.GuidedEulerStep): the guided prediction of a GuidanceForm, rescaled by a GuidanceNorm ratio,
// added to the latent in place. Mirrors GuidedEulerReference, including the F64 norm sums.
//
//   guided_euler_f32           z[i] += blend(cond[i], uncond[i]) · delta                      (no norm: one launch)
//   guided_euler_reduce_f32    per-block F64 partial sums of ‖cond‖² and ‖guided‖² over the whole tensor
//   guided_euler_finalize_f32  one ratio from the partials, clamped to [min, max]               (global norm: three launches)
//   guided_euler_scaled_f32    z[i] += (blend · ratio) · delta
//   guided_euler_rows_f32      one block per last-dimension row: its own ratio, then the update (per-row norm: one launch)
//
// blend is g·cond + (1−g)·uncond (form 0) or uncond + g·(cond − uncond) (form 1): the same value in exact arithmetic, a different
// last bit in F32, so a model keeps the form its reference uses. The parameters are by-value structs (mirrored field for field by
// CudaKernels.Guidance.cs), so a launch captures into a graph. F32 only: the latent stays F32 through the denoise loop.
//
// Build: ./build.sh (nvrtc through ../build_common.sh).

struct GuidedEulerArgs
{
    float guidance;
    float delta;
    int form;
    int reserved;
    unsigned long long count;
};

struct GuidanceNormArgs
{
    float min;
    float max;
    float eps;
    unsigned int partialCount;
};

__device__ __forceinline__ float guided_blend(int form, float g, float c, float u)
{
    if (form == 0) return g * c + (1.0f - g) * u;
    return u + g * (c - u);
}

// ‖cond‖ / (‖guided‖ + eps), 0 when that denominator is 0, clamped to [min, max]; formed in F64 like the host reference.
__device__ __forceinline__ float guided_ratio(double condSq, double guidedSq, GuidanceNormArgs n)
{
    double denom = sqrt(guidedSq) + (double)n.eps;
    double ratio = denom > 0.0 ? sqrt(condSq) / denom : 0.0;
    if (ratio < (double)n.min) ratio = (double)n.min;
    else if (ratio > (double)n.max) ratio = (double)n.max;
    return (float)ratio;
}

// Block-wide sum of each thread's two partials into sums[0] of each half of the dynamic shared buffer.
__device__ __forceinline__ void block_reduce_pair(double* condSums, double* guidedSums, double condSq, double guidedSq)
{
    condSums[threadIdx.x] = condSq;
    guidedSums[threadIdx.x] = guidedSq;
    __syncthreads();
    for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (threadIdx.x < offset)
        {
            condSums[threadIdx.x] += condSums[threadIdx.x + offset];
            guidedSums[threadIdx.x] += guidedSums[threadIdx.x + offset];
        }
        __syncthreads();
    }
}

extern "C" __global__ void guided_euler_f32(
    float* __restrict__ z,
    const float* __restrict__ cond,
    const float* __restrict__ uncond,
    GuidedEulerArgs a)
{
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < a.count; i += stride)
    {
        float v = guided_blend(a.form, a.guidance, cond[i], uncond[i]);
        z[i] = z[i] + v * a.delta;
    }
}

extern "C" __global__ void guided_euler_reduce_f32(
    const float* __restrict__ cond,
    const float* __restrict__ uncond,
    double* __restrict__ condPartials,
    double* __restrict__ guidedPartials,
    GuidedEulerArgs a)
{
    extern __shared__ double sums[];
    double* condSums = sums;
    double* guidedSums = sums + blockDim.x;

    double condSq = 0.0;
    double guidedSq = 0.0;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < a.count; i += stride)
    {
        float c = cond[i];
        float guided = guided_blend(a.form, a.guidance, c, uncond[i]);
        condSq += (double)c * (double)c;
        guidedSq += (double)guided * (double)guided;
    }
    block_reduce_pair(condSums, guidedSums, condSq, guidedSq);
    if (threadIdx.x == 0)
    {
        condPartials[blockIdx.x] = condSums[0];
        guidedPartials[blockIdx.x] = guidedSums[0];
    }
}

extern "C" __global__ void guided_euler_finalize_f32(
    float* __restrict__ scale,
    const double* __restrict__ condPartials,
    const double* __restrict__ guidedPartials,
    GuidanceNormArgs n)
{
    extern __shared__ double sums[];
    double* condSums = sums;
    double* guidedSums = sums + blockDim.x;
    double condSq = 0.0;
    double guidedSq = 0.0;
    for (unsigned int i = threadIdx.x; i < n.partialCount; i += blockDim.x)
    {
        condSq += condPartials[i];
        guidedSq += guidedPartials[i];
    }
    block_reduce_pair(condSums, guidedSums, condSq, guidedSq);
    if (threadIdx.x == 0) scale[0] = guided_ratio(condSums[0], guidedSums[0], n);
}

extern "C" __global__ void guided_euler_scaled_f32(
    float* __restrict__ z,
    const float* __restrict__ cond,
    const float* __restrict__ uncond,
    const float* __restrict__ scale,
    GuidedEulerArgs a)
{
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    float s = scale[0];
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < a.count; i += stride)
    {
        float guided = guided_blend(a.form, a.guidance, cond[i], uncond[i]);
        guided *= s;
        z[i] += guided * a.delta;
    }
}

// a.count is the row count; lastDim the row width. The block reduces one row, broadcasts its ratio and applies the update.
extern "C" __global__ void guided_euler_rows_f32(
    float* __restrict__ z,
    const float* __restrict__ cond,
    const float* __restrict__ uncond,
    GuidedEulerArgs a,
    unsigned long long lastDim,
    GuidanceNormArgs n)
{
    extern __shared__ double sums[];
    double* condSums = sums;
    double* guidedSums = sums + blockDim.x;
    __shared__ float rowRatio;

    for (unsigned long long row = blockIdx.x; row < a.count; row += gridDim.x)
    {
        unsigned long long base = row * lastDim;
        double condSq = 0.0;
        double guidedSq = 0.0;
        for (unsigned long long d = threadIdx.x; d < lastDim; d += blockDim.x)
        {
            unsigned long long i = base + d;
            float c = cond[i];
            float guided = guided_blend(a.form, a.guidance, c, uncond[i]);
            condSq += (double)c * (double)c;
            guidedSq += (double)guided * (double)guided;
        }
        block_reduce_pair(condSums, guidedSums, condSq, guidedSq);
        if (threadIdx.x == 0) rowRatio = guided_ratio(condSums[0], guidedSums[0], n);
        __syncthreads();

        float ratio = rowRatio;
        for (unsigned long long d = threadIdx.x; d < lastDim; d += blockDim.x)
        {
            unsigned long long i = base + d;
            float guided = guided_blend(a.form, a.guidance, cond[i], uncond[i]);
            z[i] += a.delta * (guided * ratio);
        }
        __syncthreads();
    }
}
