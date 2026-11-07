// Matrix-Game 3.0 ActionModule temporal-batched rearranges, FP32. Ports the host DataPointer pointer-loops
// (SplitQkvTemporal / MergeTemporal / ApplyRopeBatched / keyboard K-V expand) that dominated the MG3 forward
// (~1.07 s of a 1.28 s forward across 15 action blocks) onto the device so the mouse/keyboard attention stays
// GPU-resident. Layout: token rows are (f, s) frame-major (token = f*sp + s); the batched attention tensor is
// [sp, heads, tt, headDim] (per spatial position, a tiny frame-length-tt temporal attention). Compiled to PTX via
// nvrtc (no nvcc frontend on this box). One thread per output element.

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
