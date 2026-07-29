// Batched CausalConv3d helpers (FP32) — collapse the per-frame-per-tap loop into `kt` batched Conv2D calls.
// The GPU was idle ~66% during VAE decode due to thousands of tiny ops; these let one Conv2D process ALL frames
// per temporal tap, then a temporal gather-sum assembles the output. One thread per element.

// Build the frame-major padded input [paddedT, cIn, H+2*padH, W+2*padW] from input [1,cIn,Tin,H,W] (=[cIn,Tin,HW])
// + optional cache [1,cIn,cacheLen,H,W]. Frames: [zeroPad pad-frames] + [cache] + [input] (+ trailing clamp).
// Transpose (cIn<->T) + temporal pad + cache prepend + optional spatial edge-replicate pad, in one pass.
// replicateFirst=0: pad frames are zeros (Wan). replicateFirst=1: pad frames replicate the first content frame
// (HunyuanVideo/LTX F.pad mode="replicate"). padH/padW>0 emit spatially edge-clamped frames so the caller runs
// Conv2D with zero padding disabled (replicate spatial pad, F.pad mode="replicate").
extern "C" __global__ void wan_vae_build_padded(
    float* __restrict__ padded, const float* __restrict__ input, const float* __restrict__ cache,
    unsigned int paddedT, unsigned int cIn, unsigned int Tin, unsigned int cacheLen,
    unsigned int zeroPad, unsigned int H, unsigned int W,
    unsigned int padH, unsigned int padW, unsigned int replicateFirst)
{
    unsigned int Hp = H + 2u * padH, Wp = W + 2u * padW;
    unsigned long long HWp = (unsigned long long)Hp * Wp;
    unsigned long long total = (unsigned long long)paddedT * cIn * HWp;
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    unsigned long long hw = idx % HWp;
    unsigned long long tc = idx / HWp;
    unsigned int ci = (unsigned int)(tc % cIn);
    unsigned int t  = (unsigned int)(tc / cIn);
    int sy = (int)(hw / Wp) - (int)padH;
    int sx = (int)(hw % Wp) - (int)padW;
    if (sy < 0) sy = 0; else if (sy >= (int)H) sy = (int)H - 1;
    if (sx < 0) sx = 0; else if (sx >= (int)W) sx = (int)W - 1;
    unsigned long long srcHW = (unsigned long long)sy * W + sx;
    unsigned long long HW = (unsigned long long)H * W;
    float v;
    if (t < zeroPad && replicateFirst == 0u) v = 0.0f;
    else {
        unsigned int az = t < zeroPad ? 0u : t - zeroPad;   // replicate-first: pad frames clamp to the first content frame
        if (az < cacheLen) v = cache[((unsigned long long)ci * cacheLen + az) * HW + srcHW];
        else {
            unsigned int ti = az - cacheLen;
            if (ti >= Tin) ti = Tin - 1;   // trailing clamp (non-causal replicate)
            v = input[((unsigned long long)ci * Tin + ti) * HW + srcHW];
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
