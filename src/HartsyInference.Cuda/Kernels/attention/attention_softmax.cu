// Materialized attention softmax. Add the public F32 bias without narrowing or clamping it.
// All -infinity rows produce zero probabilities; NaN and +infinity scores propagate NaN. On finite rows the softmax
// itself reproduces the legacy softmax_f32/_f16 arithmetic bit for bit (same block size, reduction order, ex2.approx
// and rcp.approx). A biased F16 row does not reproduce the old add-then-softmax pipeline, which clamped the bias to
// the F16 range and added it in F16.
// One 256-thread block per row; dynamic shared memory is blockDim.x * sizeof(float).
// Build: ./build.sh (pinned NVRTC, sm_80 baseline and sm_75 floor).
#include <cuda_fp16.h>
#include <math_constants.h>

template <typename T>
__device__ void attention_softmax(T* data, const float* bias, unsigned int rowLen, unsigned int rows)
{
    unsigned int row = blockIdx.x;
    unsigned int tid = threadIdx.x;
    if (row >= rows) return;
    size_t base = (size_t)row * rowLen;
    extern __shared__ float scratch[];
    float maximum = -CUDART_INF_F;
    for (unsigned int c = tid; c < rowLen; c += blockDim.x)
    {
        float score = (float)data[base + c] + (bias ? bias[base + c] : 0.0f);
        maximum = isnan(score) || isnan(maximum) ? CUDART_NAN_F : fmaxf(maximum, score);
    }
    scratch[tid] = maximum;
    __syncthreads();
    for (unsigned int step = blockDim.x / 2; step > 0; step /= 2)
    {
        if (tid < step)
        {
            float other = scratch[tid + step];
            scratch[tid] = isnan(scratch[tid]) || isnan(other)
                ? CUDART_NAN_F : fmaxf(scratch[tid], other);
        }
        __syncthreads();
    }
    maximum = scratch[0];
    __syncthreads();
    float sum = 0.0f;
    for (unsigned int c = tid; c < rowLen; c += blockDim.x)
    {
        float score = (float)data[base + c] + (bias ? bias[base + c] : 0.0f);
        float probability = maximum == -CUDART_INF_F ? 0.0f : __expf(score - maximum);
        sum += probability;
        data[base + c] = (T)probability;
    }
    scratch[tid] = sum;
    __syncthreads();
    for (unsigned int step = blockDim.x / 2; step > 0; step /= 2)
    {
        if (tid < step) scratch[tid] += scratch[tid + step];
        __syncthreads();
    }
    sum = scratch[0];
    // rcp.approx, not 1.0f / sum (rcp.rn): the legacy softmax_f32/_f16 PTX normalized with the approximate reciprocal,
    // so finite rows stay bit-identical to it. Only all-excluded and NaN rows differ, as the contract requires.
    float inverse;
    if (sum > 0.0f)
        asm("rcp.approx.f32 %0, %1;" : "=f"(inverse) : "f"(sum));
    else
        inverse = isnan(sum) ? sum : 0.0f;
    for (unsigned int c = tid; c < rowLen; c += blockDim.x)
        data[base + c] = (T)((float)data[base + c] * inverse);
}

extern "C" __global__ void attention_softmax_f32(
    float* data, const float* bias, unsigned int rowLen, unsigned int rows)
{
    attention_softmax(data, bias, rowLen, rows);
}

extern "C" __global__ void attention_softmax_f16(
    __half* data, const float* bias, unsigned int rowLen, unsigned int rows)
{
    attention_softmax(data, bias, rowLen, rows);
}
