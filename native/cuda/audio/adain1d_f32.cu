// Adaptive Instance Normalization 1D — the core style-conditioning primitive in
// Kokoro / StyleTTS 2 (prosody predictor F0/N chains, decoder AdainResBlk1d, and the
// HiFi-GAN MRF resblocks).
//
// input/output: [B, C, T] channels-first. Per (batch, channel) row, normalize over the
// T axis (biased variance, /T), then apply the style-modulated affine:
//   out = (in - mean) * (1 + gamma[idx]) * invStd + beta[idx]
// gamma/beta are [B, C] (perBatch=1, idx = row) or [C] (perBatch=0, idx = row % channels);
// they come from a Linear projection of the speaker/style embedding.
//
// One block per row; shared-mem two-pass reduction for mean then variance. This mirrors
// dit_layernorm_noaffine_f32 and matches the CPU reference in NormKernels.AdaInstanceNorm1d
// exactly (scale = (1+gamma)*invStd, shift = beta - mean*scale).
//
// Build:
//   nvcc --ptx -arch=sm_80 adain1d_f32.cu -o adain1d_f32.ptx
//   (move the .ptx into src/HartsyInference.Cuda/Ptx/)

extern "C" __global__ void audio_adain1d_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    unsigned int dim,         // T
    unsigned int totalRows,   // B * C
    unsigned int channels,    // C
    int perBatch,             // 1 if gamma/beta are [B,C], else 0 ([C], broadcast over batch)
    float eps)
{
    extern __shared__ float sdata[];
    unsigned int row = blockIdx.x;
    if (row >= totalRows) return;

    const float* inRow = input + (size_t)row * dim;
    float* outRow = output + (size_t)row * dim;

    // Pass 1: mean over T.
    float partial = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x)
        partial += inRow[i];
    sdata[threadIdx.x] = partial;
    __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float mean = sdata[0] / (float)dim;
    __syncthreads();

    // Pass 2: biased variance over T.
    float vpart = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x)
    {
        float diff = inRow[i] - mean;
        vpart += diff * diff;
    }
    sdata[threadIdx.x] = vpart;
    __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float invStd = rsqrtf(sdata[0] / (float)dim + eps);

    // Style-modulated affine. Fold (in - mean)*scale into in*scale + shift.
    unsigned int aidx = perBatch ? row : (row % channels);
    float scale = (1.0f + gamma[aidx]) * invStd;
    float shift = beta[aidx] - mean * scale;
    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x)
        outRow[i] = inRow[i] * scale + shift;
}
