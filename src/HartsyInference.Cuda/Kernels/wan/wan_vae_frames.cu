// Wan/3D-VAE temporal frame ops (FP32 + BF16) — make CausalConv3d GPU-resident. Both are strided copies between a 5D
// [B,C,T,H,W] tensor's temporal slot and a 4D [B,C,H,W] frame. One thread per (b,c,hw).
#include <cuda_bf16.h>

// out[B,C,H,W] = src[B,C,Tsrc,H,W][:, :, ti, :, :]
extern "C" __global__ void wan_vae_extract_frame(
    float* __restrict__ out, const float* __restrict__ src,
    unsigned int ti, unsigned int B, unsigned int C, unsigned int Tsrc, unsigned int frameHW)
{
    unsigned long long total = (unsigned long long)B * C * frameHW;
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    unsigned int hw = (unsigned int)(idx % frameHW);
    unsigned long long bc = idx / frameHW;
    unsigned int c = (unsigned int)(bc % C);
    unsigned int b = (unsigned int)(bc / C);
    unsigned long long srcOff = (((unsigned long long)b * C + c) * Tsrc + ti) * frameHW + hw;
    out[idx] = src[srcOff];
}

// out[B,C,Tout,H,W][:, :, to, :, :] = acc[B,C,H,W] + bias[c]  (bias may be null)
extern "C" __global__ void wan_vae_write_frame(
    float* __restrict__ out, const float* __restrict__ acc, const float* __restrict__ bias,
    unsigned int to, unsigned int B, unsigned int C, unsigned int Tout, unsigned int frameHW)
{
    unsigned long long total = (unsigned long long)B * C * frameHW;
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    unsigned int hw = (unsigned int)(idx % frameHW);
    unsigned long long bc = idx / frameHW;
    unsigned int c = (unsigned int)(bc % C);
    unsigned int b = (unsigned int)(bc / C);
    float bv = (bias != 0) ? bias[c] : 0.0f;
    unsigned long long outOff = (((unsigned long long)b * C + c) * Tout + to) * frameHW + hw;
    out[outOff] = acc[idx] + bv;
}

// BF16 variants (bias stays FP32; math in FP32 per element). Same indexing as the FP32 kernels above.
extern "C" __global__ void wan_vae_extract_frame_bf16(
    __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ src,
    unsigned int ti, unsigned int B, unsigned int C, unsigned int Tsrc, unsigned int frameHW)
{
    unsigned long long total = (unsigned long long)B * C * frameHW;
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    unsigned int hw = (unsigned int)(idx % frameHW);
    unsigned long long bc = idx / frameHW;
    unsigned int c = (unsigned int)(bc % C);
    unsigned int b = (unsigned int)(bc / C);
    unsigned long long srcOff = (((unsigned long long)b * C + c) * Tsrc + ti) * frameHW + hw;
    out[idx] = src[srcOff];
}

extern "C" __global__ void wan_vae_write_frame_bf16(
    __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ acc, const float* __restrict__ bias,
    unsigned int to, unsigned int B, unsigned int C, unsigned int Tout, unsigned int frameHW)
{
    unsigned long long total = (unsigned long long)B * C * frameHW;
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    unsigned int hw = (unsigned int)(idx % frameHW);
    unsigned long long bc = idx / frameHW;
    unsigned int c = (unsigned int)(bc % C);
    unsigned int b = (unsigned int)(bc / C);
    float bv = (bias != 0) ? bias[c] : 0.0f;
    unsigned long long outOff = (((unsigned long long)b * C + c) * Tout + to) * frameHW + hw;
    out[outOff] = __float2bfloat16(__bfloat162float(acc[idx]) + bv);
}
