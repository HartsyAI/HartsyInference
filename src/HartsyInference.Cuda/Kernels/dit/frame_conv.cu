// Depthwise convolution along the frame axis of token-major [frames, spatial, channels] F32 data
// (IBackend.DepthwiseConvFrames): out[t, s, c] = sum_j w[c, j] * x[t + j - pad, s, c], zero outside the clip. One thread
// per output element; consecutive threads take consecutive channels, so every tap load is coalesced. Host reference:
// DepthwiseConvFramesReference.
//
// Build: src/HartsyInference.Cuda/Kernels/dit/build.sh.

extern "C" __global__ void depthwise_conv_frames_f32(
    float* __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ w,     // [channels, taps]
    unsigned int frames,
    unsigned int spatial,
    unsigned int channels,
    unsigned int taps,
    unsigned int pad)
{
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long plane = (unsigned long long)spatial * channels;
    unsigned long long total = (unsigned long long)frames * plane;
    if (idx >= total) return;
    unsigned int t = (unsigned int)(idx / plane);
    unsigned int c = (unsigned int)(idx % channels);
    float acc = 0.0f;
    for (unsigned int j = 0; j < taps; j++)
    {
        int src = (int)t + (int)j - (int)pad;
        if (src < 0 || src >= (int)frames) continue;
        acc += w[(unsigned long long)c * taps + j] * x[idx + ((long long)src - (long long)t) * (long long)plane];
    }
    out[idx] = acc;
}
