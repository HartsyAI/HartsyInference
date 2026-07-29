// Wan-Video interleaved 3D-RoPE, FP32, in-place. Matches WanRope.ApplyRotary (diffusers WanRotaryPosEmbed):
// x is [S, heads*headDim]; for each (s, head, pair i) rotate the adjacent pair (2i, 2i+1) by the shared angle
// stored (duplicated) at cos/sin index 2i. cos/sin are [S, headDim] shared across heads (standard sigma_theta=0
// path). One thread per (s, head, pair). Compiled to PTX via nvrtc (no nvcc frontend on this box).
extern "C" __global__ void wan_rope_interleaved(
    float* __restrict__ x,
    const float* __restrict__ cosT,
    const float* __restrict__ sinT,
    unsigned int S, unsigned int heads, unsigned int headDim)
{
    unsigned int pairs = headDim >> 1;
    unsigned long long total = (unsigned long long)S * heads * pairs;
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    unsigned int i = (unsigned int)(idx % pairs); idx /= pairs;
    unsigned int h = (unsigned int)(idx % heads); idx /= heads;
    unsigned int s = (unsigned int)idx;
    unsigned int i0 = 2u * i;
    unsigned long long xoff = ((unsigned long long)s * heads + h) * headDim + i0;
    unsigned long long coff = (unsigned long long)s * headDim + i0;   // shared across heads
    float re = x[xoff], im = x[xoff + 1];
    float c = cosT[coff], sn = sinT[coff];
    x[xoff]     = re * c - im * sn;
    x[xoff + 1] = re * sn + im * c;
}

// Per-head variant (Matrix-Game 3.0 sigma_theta): cos/sin are [heads, S, headDim] — each head its own theta.
// Angle for (s, head, pair i) at (h*S + s)*headDim + 2i. One thread per (s, head, pair). Ported off the host loop
// that dominated the MG3 backbone (~1.9s of a 2.05s forward).
extern "C" __global__ void wan_rope_interleaved_perhead(
    float* __restrict__ x,
    const float* __restrict__ cosT,
    const float* __restrict__ sinT,
    unsigned int S, unsigned int heads, unsigned int headDim)
{
    unsigned int pairs = headDim >> 1;
    unsigned long long total = (unsigned long long)S * heads * pairs;
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    unsigned int i = (unsigned int)(idx % pairs); idx /= pairs;
    unsigned int h = (unsigned int)(idx % heads); idx /= heads;
    unsigned int s = (unsigned int)idx;
    unsigned int i0 = 2u * i;
    unsigned long long xoff = ((unsigned long long)s * heads + h) * headDim + i0;
    unsigned long long coff = ((unsigned long long)h * S + s) * headDim + i0;   // per-head cos/sin
    float re = x[xoff], im = x[xoff + 1];
    float c = cosT[coff], sn = sinT[coff];
    x[xoff]     = re * c - im * sn;
    x[xoff + 1] = re * sn + im * c;
}
