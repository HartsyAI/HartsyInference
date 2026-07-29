// Matrix-Game 3.0 ActionModule temporal-batched rearranges, FP32. Ports the host DataPointer pointer-loops
// (SplitQkvTemporal / MergeTemporal / ApplyRopeBatched / keyboard K-V expand) that dominated the MG3 forward
// (~1.07 s of a 1.28 s forward across 15 action blocks) onto the device so the mouse/keyboard attention stays
// GPU-resident. Layout: token rows are (f, s) frame-major (token = f*sp + s); the batched attention tensor is
// [sp, heads, tt, headDim] (per spatial position, a tiny frame-length-tt temporal attention). Compiled to PTX via
// nvrtc (no nvcc frontend on this box). One thread per output element.

// qkv[tt*sp, stride*streamDim] slot `part` -> [sp, heads, tt, headDim].  streamDim = heads*headDim.
extern "C" __global__ void mg3_split_qkv_temporal_f32(
    const float* __restrict__ qkv, float* __restrict__ out,
    unsigned int tt, unsigned int sp, unsigned int heads, unsigned int headDim,
    unsigned int part, unsigned int stride)
{
    unsigned long long total = (unsigned long long)sp * heads * tt * headDim;
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    unsigned int streamDim = heads * headDim;
    unsigned int d = (unsigned int)(idx % headDim); idx /= headDim;
    unsigned int f = (unsigned int)(idx % tt);      idx /= tt;
    unsigned int h = (unsigned int)(idx % heads);   idx /= heads;
    unsigned int s = (unsigned int)idx;
    unsigned long long src = (unsigned long long)(f * sp + s) * (stride * streamDim) + part * streamDim + h * headDim + d;
    unsigned long long dst = (((unsigned long long)s * heads + h) * tt + f) * headDim + d;
    out[dst] = qkv[src];
}

// [sp, heads, tt, headDim] -> token rows [tt*sp, streamDim].
extern "C" __global__ void mg3_merge_temporal_f32(
    const float* __restrict__ attn, float* __restrict__ out,
    unsigned int tt, unsigned int sp, unsigned int heads, unsigned int headDim)
{
    unsigned long long total = (unsigned long long)tt * sp * heads * headDim;
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    unsigned int streamDim = heads * headDim;
    unsigned int d = (unsigned int)(idx % headDim); idx /= headDim;
    unsigned int h = (unsigned int)(idx % heads);   idx /= heads;
    unsigned int s = (unsigned int)(idx % sp);      idx /= sp;
    unsigned int f = (unsigned int)idx;
    unsigned long long src = (((unsigned long long)s * heads + h) * tt + f) * headDim + d;
    unsigned long long dst = (unsigned long long)(f * sp + s) * streamDim + h * headDim + d;
    out[dst] = attn[src];
}

// Interleaved rope on [sp, heads, tt, headDim]. cos/sin are [gridRows, headDim] (duplicated pairs). The grid row for
// (s, f) is f when broadcastSpatial else f*gh*gw + s. One thread per (s, h, f, pair).
extern "C" __global__ void mg3_rope_batched_f32(
    float* __restrict__ x, const float* __restrict__ cosT, const float* __restrict__ sinT,
    unsigned int sp, unsigned int heads, unsigned int tt, unsigned int headDim,
    unsigned int gh, unsigned int gw, unsigned int broadcastSpatial)
{
    unsigned int pairs = headDim >> 1;
    unsigned long long total = (unsigned long long)sp * heads * tt * pairs;
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    unsigned int i = (unsigned int)(idx % pairs); idx /= pairs;
    unsigned int f = (unsigned int)(idx % tt);    idx /= tt;
    unsigned int h = (unsigned int)(idx % heads); idx /= heads;
    unsigned int s = (unsigned int)idx;
    unsigned int i0 = 2u * i;
    unsigned long long gridRow = broadcastSpatial ? f : (unsigned long long)f * gh * gw + s;
    unsigned long long cOff = gridRow * headDim + i0;
    unsigned long long xOff = (((unsigned long long)s * heads + h) * tt + f) * headDim + i0;
    float re = x[xOff], im = x[xOff + 1];
    float c = cosT[cOff], sn = sinT[cOff];
    x[xOff]     = re * c - im * sn;
    x[xOff + 1] = re * sn + im * c;
}

// keyboard kv[tt, 2*streamDim] -> k,v [sp, heads, tt, headDim], broadcast across the sp spatial positions.
extern "C" __global__ void mg3_kv_expand_f32(
    const float* __restrict__ kv, float* __restrict__ kOut, float* __restrict__ vOut,
    unsigned int sp, unsigned int heads, unsigned int tt, unsigned int headDim)
{
    unsigned long long total = (unsigned long long)sp * heads * tt * headDim;
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    unsigned int streamDim = heads * headDim;
    unsigned int d = (unsigned int)(idx % headDim); idx /= headDim;
    unsigned int f = (unsigned int)(idx % tt);      idx /= tt;
    unsigned int h = (unsigned int)(idx % heads);   idx /= heads;
    unsigned int s = (unsigned int)idx;
    unsigned long long dst = (((unsigned long long)s * heads + h) * tt + f) * headDim + d;
    unsigned long long kvBase = (unsigned long long)f * 2u * streamDim + h * headDim + d;
    kOut[dst] = kv[kvBase];
    vOut[dst] = kv[kvBase + streamDim];
}

// MG3 mouse stream: build the MLP input [tt*sp, imgDim+winFloats] = [hidden token features | per-frame mouse window]
// (the window is broadcast across the sp spatial positions of a frame). Ports the host concat that read `hidden` D2H
// every mouse block. token = f*sp + s. One thread per output element.
extern "C" __global__ void mg3_mouse_mlp_concat_f32(
    const float* __restrict__ hidden, const float* __restrict__ mouseWin, float* __restrict__ out,
    unsigned int tt, unsigned int sp, unsigned int imgDim, unsigned int winFloats)
{
    unsigned int rowW = imgDim + winFloats;
    unsigned long long total = (unsigned long long)tt * sp * rowW;
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    unsigned int col = (unsigned int)(idx % rowW); unsigned long long token = idx / rowW;
    if (col < imgDim) { out[idx] = hidden[token * imgDim + col]; }
    else { unsigned int f = (unsigned int)(token / sp); out[idx] = mouseWin[(unsigned long long)f * winFloats + (col - imgDim)]; }
}
