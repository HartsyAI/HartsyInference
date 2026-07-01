// Batched CausalConv3d helpers (FP32) — collapse the per-frame-per-tap loop into `kt` batched Conv2D calls.
// The GPU was idle ~66% during VAE decode due to thousands of tiny ops; these let one Conv2D process ALL frames
// per temporal tap, then a temporal gather-sum assembles the output. One thread per element.

// Build the frame-major padded input [paddedT, cIn, H, W] from input [1,cIn,Tin,H,W] (=[cIn,Tin,HW]) + optional
// cache [1,cIn,cacheLen,H,W]. Frames: [zeroPad zeros] + [cache] + [input] (+ trailing clamp). This is a
// transpose (cIn<->T) + causal zero-pad + cache prepend, in one pass. Wan path only (no first-frame replicate).
extern "C" __global__ void wan_vae_build_padded(
    float* __restrict__ padded, const float* __restrict__ input, const float* __restrict__ cache,
    unsigned int paddedT, unsigned int cIn, unsigned int Tin, unsigned int cacheLen,
    unsigned int zeroPad, unsigned int HW)
{
    unsigned long long total = (unsigned long long)paddedT * cIn * HW;
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    unsigned int hw = (unsigned int)(idx % HW);
    unsigned long long tc = idx / HW;
    unsigned int ci = (unsigned int)(tc % cIn);
    unsigned int t  = (unsigned int)(tc / cIn);
    float v;
    if (t < zeroPad) v = 0.0f;
    else {
        unsigned int az = t - zeroPad;
        if (az < cacheLen) v = cache[((unsigned long long)ci * cacheLen + az) * HW + hw];
        else {
            unsigned int ti = az - cacheLen;
            if (ti >= Tin) ti = Tin - 1;   // trailing clamp (non-causal replicate)
            v = input[((unsigned long long)ci * Tin + ti) * HW + hw];
        }
    }
    padded[idx] = v;
}

// out[1,cOut,tout,H,W] (=[cOut,tout,HW]) := bias[co]  (or 0 when bias==0).
extern "C" __global__ void wan_vae_fill_bias(
    float* __restrict__ out, const float* __restrict__ bias, unsigned int cOut, unsigned int tout, unsigned int HW)
{
    unsigned long long total = (unsigned long long)cOut * tout * HW;
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    unsigned long long ct = idx / HW;
    unsigned int co = (unsigned int)(ct / tout);
    out[idx] = (bias != 0) ? bias[co] : 0.0f;
}

// out[co][to][hw] += convDt[to*strideT+dt][co][hw].  out is [cOut,tout,HW]; convDt is [paddedT,cOut,HW] (frame-major).
extern "C" __global__ void wan_vae_accumulate_tap(
    float* __restrict__ out, const float* __restrict__ convDt, unsigned int dt, unsigned int strideT,
    unsigned int cOut, unsigned int tout, unsigned int HW)
{
    unsigned long long total = (unsigned long long)cOut * tout * HW;
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    unsigned int hw = (unsigned int)(idx % HW);
    unsigned long long ct = idx / HW;
    unsigned int to = (unsigned int)(ct % tout);
    unsigned int co = (unsigned int)(ct / tout);
    unsigned int srcT = to * strideT + dt;
    out[idx] += convDt[((unsigned long long)srcT * cOut + co) * HW + hw];
}
