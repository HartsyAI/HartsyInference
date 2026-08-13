// Batched CausalConv3d helpers (FP32) — collapse the per-frame-per-tap loop into `kt` batched Conv2D calls.
// The GPU was idle ~66% during VAE decode due to thousands of tiny ops; these let one Conv2D process ALL frames
// per temporal tap, then a temporal gather-sum assembles the output. One thread per element.

// Build the frame-major padded input [paddedT, cIn, H+2*padH, W+2*padW] from input [1,cIn,Tin,H,W] (=[cIn,Tin,HW])
// + optional cache [1,cIn,cacheLen,H,W]. Frames: [zeroPad pad-frames] + [cache] + [input] (+ trailing clamp).
// Transpose (cIn<->T) + temporal pad + cache prepend + optional spatial edge-replicate pad, in one pass.
// replicateFirst=0: pad frames are zeros (Wan). replicateFirst=1: pad frames replicate the first content frame
// (HunyuanVideo/LTX F.pad mode="replicate"). padH/padW>0 emit spatially padded frames so the caller runs
// Conv2D with zero padding disabled: edge-clamped (F.pad mode="replicate") when reflectSpatial=0, mirrored
// (F.pad mode="reflect", the LTX-2 VAE's default) when reflectSpatial=1.
extern "C" __global__ void wan_vae_build_padded(
    float* __restrict__ padded, const float* __restrict__ input, const float* __restrict__ cache,
    unsigned int paddedT, unsigned int cIn, unsigned int Tin, unsigned int cacheLen,
    unsigned int zeroPad, unsigned int H, unsigned int W,
    unsigned int padH, unsigned int padW, unsigned int replicateFirst, unsigned int reflectSpatial)
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
    if (reflectSpatial != 0u) {
        // PyTorch F.pad(mode="reflect"): the border pixel is NOT repeated (-1 -> 1, H -> H-2).
        if (sy < 0) sy = -sy; else if (sy >= (int)H) sy = 2 * ((int)H - 1) - sy;
        if (sx < 0) sx = -sx; else if (sx >= (int)W) sx = 2 * ((int)W - 1) - sx;
        // Reflection only stays in range while pad <= len-1; clamp so a degenerate axis cannot read OOB.
        if (sy < 0) sy = 0; else if (sy >= (int)H) sy = (int)H - 1;
        if (sx < 0) sx = 0; else if (sx >= (int)W) sx = (int)W - 1;
    } else {
        if (sy < 0) sy = 0; else if (sy >= (int)H) sy = (int)H - 1;
        if (sx < 0) sx = 0; else if (sx >= (int)W) sx = (int)W - 1;
    }
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

// BF16 variants (bias stays FP32; per-element math in FP32). Same indexing as the FP32 kernels above.
#include <cuda_bf16.h>

extern "C" __global__ void wan_vae_build_padded_bf16(
    __nv_bfloat16* __restrict__ padded, const __nv_bfloat16* __restrict__ input, const __nv_bfloat16* __restrict__ cache,
    unsigned int paddedT, unsigned int cIn, unsigned int Tin, unsigned int cacheLen,
    unsigned int zeroPad, unsigned int H, unsigned int W,
    unsigned int padH, unsigned int padW, unsigned int replicateFirst, unsigned int reflectSpatial)
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
    if (reflectSpatial != 0u) {
        // PyTorch F.pad(mode="reflect"): the border pixel is NOT repeated (-1 -> 1, H -> H-2).
        if (sy < 0) sy = -sy; else if (sy >= (int)H) sy = 2 * ((int)H - 1) - sy;
        if (sx < 0) sx = -sx; else if (sx >= (int)W) sx = 2 * ((int)W - 1) - sx;
        // Reflection only stays in range while pad <= len-1; clamp so a degenerate axis cannot read OOB.
        if (sy < 0) sy = 0; else if (sy >= (int)H) sy = (int)H - 1;
        if (sx < 0) sx = 0; else if (sx >= (int)W) sx = (int)W - 1;
    } else {
        if (sy < 0) sy = 0; else if (sy >= (int)H) sy = (int)H - 1;
        if (sx < 0) sx = 0; else if (sx >= (int)W) sx = (int)W - 1;
    }
    unsigned long long srcHW = (unsigned long long)sy * W + sx;
    unsigned long long HW = (unsigned long long)H * W;
    __nv_bfloat16 v;
    if (t < zeroPad && replicateFirst == 0u) v = __float2bfloat16(0.0f);
    else {
        unsigned int az = t < zeroPad ? 0u : t - zeroPad;
        if (az < cacheLen) v = cache[((unsigned long long)ci * cacheLen + az) * HW + srcHW];
        else {
            unsigned int ti = az - cacheLen;
            if (ti >= Tin) ti = Tin - 1;
            v = input[((unsigned long long)ci * Tin + ti) * HW + srcHW];
        }
    }
    padded[idx] = v;
}

extern "C" __global__ void wan_vae_fill_bias_bf16(
    __nv_bfloat16* __restrict__ out, const float* __restrict__ bias, unsigned int cOut, unsigned int tout, unsigned int HW)
{
    unsigned long long total = (unsigned long long)cOut * tout * HW;
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    unsigned long long ct = idx / HW;
    unsigned int co = (unsigned int)(ct / tout);
    out[idx] = __float2bfloat16((bias != 0) ? bias[co] : 0.0f);
}

extern "C" __global__ void wan_vae_accumulate_tap_bf16(
    __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ convDt, unsigned int dt, unsigned int strideT,
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
    float acc = __bfloat162float(out[idx]) + __bfloat162float(convDt[((unsigned long long)srcT * cOut + co) * HW + hw]);
    out[idx] = __float2bfloat16(acc);
}

// SeedVR2 MAGViT channel-to-space shuffle: in [1,cIn,f,h,w] -> out [1,c,fFinal,h*sr,w*sr] with
// c = cIn/(sr*sr*tr), source channel ((xi*sr+yi)*tr+zi)*c+ci, and (temporal only) output frame INDEX 1
// dropped (T -> 2T-1). One thread per OUTPUT element (pure gather; inverts the drop by of>=1 -> of+1).
extern "C" __global__ void seedvr2_pixel_shuffle_f32(
    float* __restrict__ out, const float* __restrict__ in,
    unsigned int c, unsigned int fFinal, unsigned int hOut, unsigned int wOut,
    unsigned int cIn, unsigned int f, unsigned int h, unsigned int w,
    unsigned int sr, unsigned int tr, unsigned int dropDup)
{
    unsigned long long total = (unsigned long long)c * fFinal * hOut * wOut;
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    unsigned int ox = (unsigned int)(idx % wOut);
    unsigned long long r = idx / wOut;
    unsigned int oy = (unsigned int)(r % hOut); r /= hOut;
    unsigned int of = (unsigned int)(r % fFinal);
    unsigned int ci = (unsigned int)(r / fFinal);
    unsigned int outF = (dropDup && of >= 1u) ? of + 1u : of;
    unsigned int fi = outF / tr, zi = outF % tr;
    unsigned int y = oy / sr, xi = oy % sr;
    unsigned int x = ox / sr, yi = ox % sr;
    unsigned int srcC = ((xi * sr + yi) * tr + zi) * c + ci;
    out[idx] = in[(((unsigned long long)srcC * f + fi) * h + y) * w + x];
}

extern "C" __global__ void seedvr2_pixel_shuffle_bf16(
    __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ in,
    unsigned int c, unsigned int fFinal, unsigned int hOut, unsigned int wOut,
    unsigned int cIn, unsigned int f, unsigned int h, unsigned int w,
    unsigned int sr, unsigned int tr, unsigned int dropDup)
{
    unsigned long long total = (unsigned long long)c * fFinal * hOut * wOut;
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    unsigned int ox = (unsigned int)(idx % wOut);
    unsigned long long r = idx / wOut;
    unsigned int oy = (unsigned int)(r % hOut); r /= hOut;
    unsigned int of = (unsigned int)(r % fFinal);
    unsigned int ci = (unsigned int)(r / fFinal);
    unsigned int outF = (dropDup && of >= 1u) ? of + 1u : of;
    unsigned int fi = outF / tr, zi = outF % tr;
    unsigned int y = oy / sr, xi = oy % sr;
    unsigned int x = ox / sr, yi = ox % sr;
    unsigned int srcC = ((xi * sr + yi) * tr + zi) * c + ci;
    out[idx] = in[(((unsigned long long)srcC * f + fi) * h + y) * w + x];
}

// SeedVR2 asymmetric zero pad (right/bottom only, diffusers Downsample2D(padding=0)): [B,C,T,h,w] ->
// [B,C,T,h+1,w+1]. One thread per output element.
extern "C" __global__ void seedvr2_pad_br_f32(
    float* __restrict__ out, const float* __restrict__ in,
    unsigned long long planes, unsigned int h, unsigned int w)
{
    unsigned int hp = h + 1u, wp = w + 1u;
    unsigned long long total = planes * hp * wp;
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    unsigned int x = (unsigned int)(idx % wp);
    unsigned long long r = idx / wp;
    unsigned int y = (unsigned int)(r % hp);
    unsigned long long p = r / hp;
    out[idx] = (y < h && x < w) ? in[(p * h + y) * w + x] : 0.0f;
}

extern "C" __global__ void seedvr2_pad_br_bf16(
    __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ in,
    unsigned long long planes, unsigned int h, unsigned int w)
{
    unsigned int hp = h + 1u, wp = w + 1u;
    unsigned long long total = planes * hp * wp;
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    unsigned int x = (unsigned int)(idx % wp);
    unsigned long long r = idx / wp;
    unsigned int y = (unsigned int)(r % hp);
    unsigned long long p = r / hp;
    out[idx] = (y < h && x < w) ? in[(p * h + y) * w + x] : __float2bfloat16(0.0f);
}
