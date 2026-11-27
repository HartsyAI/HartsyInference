// The DiT gated residual that joins an F16 branch to an F32 stream — F32 compute.
//
//   gated_residual_f32_f16        out[i] = residual[i] + gate[m,d]·value[i]: an F16 branch added to an F32 stream.
//
// Contract and host reference: IBackend.GatedResidualLastDim.
//
// Build: src/HartsyInference.Cuda/Kernels/dit/build.sh (nvcc, or the nvrtc helper when no toolkit is present).

#include <cuda_fp16.h>

extern "C" {

// The gate row is b = row / seqLen, as in dit_gated_residual_lastdim_f32; the F16 value widens exactly.
__global__ void gated_residual_f32_f16(float* __restrict__ output, const float* __restrict__ residual,
    const __half* __restrict__ value, const float* __restrict__ gate, unsigned int seqLen, unsigned int dim,
    unsigned long long total)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned int d = (unsigned int)(i % dim);
    unsigned long long row = i / dim;
    unsigned int b = (unsigned int)(row / seqLen);
    output[i] = residual[i] + gate[(size_t)b * dim + d] * __half2float(value[i]);
}

}
