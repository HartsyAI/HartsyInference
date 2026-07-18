// Multi-scale deformable attention (Deformable DETR) forward, as used by Grounding DINO's encoder
// self-attention / decoder cross-attention and RT-DETR's decoder. One thread per output element
// (query, head, channel): it loops over levels*points, bilinearly samples the multi-scale value maps
// (grid_sample, align_corners=false, zero padding) at each predicted location, and accumulates the
// softmax-weighted taps. Matches the CPU reference in IBackend.DeformableAttention exactly.
//
//   loc(coords==2) = ref[:2] + off / (W,H)
//   loc(coords==4) = ref[:2] + off / points * ref[2:4] * 0.5
//   out[q,h,c] = sum_{l,p} softmax_lp(attn)[q,h,l,p] * bilinear(value_l[head=h,chan=c], loc)
//
// The softmax over (levels*points) is folded in here (per-thread, two passes over the lp logits) so the
// caller keeps everything GPU-resident — no host round-trip for the attention weights. lp is tiny
// (GDINO 4x4=16), so recomputing it per channel-thread is negligible next to the sampling reads.
//
// 64-bit indexing: Nq*heads*hd and Nkv*heads*hd overflow u32 at encoder scale (Nq ~ 17.8k).
//
// Build (no nvcc on this box — use the repo's nvrtc frontend):
//   LD_LIBRARY_PATH=~/.local/lib/cuda13 native/cuda/nvrtc_compile \
//     native/cuda/vision/msda.cu native/cuda/vision/msda.ptx compute_80 ~/.local/lib/cuda13/include
//   cp native/cuda/vision/msda.ptx src/HartsyInference.Cuda/Ptx/

extern "C" {

__global__ void msda_forward_f32(
    float* __restrict__ out,            // [Nq, heads*hd]
    const float* __restrict__ value,    // [Nkv, heads*hd]  (== [Nkv, heads, hd])
    const float* __restrict__ sampOff,  // [Nq, heads, levels, points, 2]
    const float* __restrict__ attn,     // [Nq, heads, levels, points]  raw logits
    const float* __restrict__ refPoints,// [Nq, levels, coords]
    const int* __restrict__ shapes,     // [levels*2]  (h, w) per level
    const int* __restrict__ levelStart, // [levels]
    unsigned int Nq, unsigned int heads, unsigned int hd,
    unsigned int levels, unsigned int points, unsigned int coords,
    unsigned int refQueryStride, unsigned int refLevelStride)
{
    unsigned long long total = (unsigned long long)Nq * heads * hd;
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    unsigned int c = (unsigned int)(idx % hd);
    unsigned long long t = idx / hd;
    unsigned int h = (unsigned int)(t % heads);
    unsigned int q = (unsigned int)(t / heads);

    unsigned int lp = levels * points;
    const float* aBase = attn + (unsigned long long)(q * heads + h) * lp;

    // softmax denominator over lp (pass 1)
    float amax = -1e30f;
    for (unsigned int i = 0; i < lp; i++) { float v = aBase[i]; if (v > amax) amax = v; }
    float asum = 0.0f;
    for (unsigned int i = 0; i < lp; i++) asum += __expf(aBase[i] - amax);
    float invSum = 1.0f / asum;

    unsigned long long refBase = (unsigned long long)q * refQueryStride;
    unsigned int hdHeads = heads * hd;
    float acc = 0.0f;

    for (unsigned int l = 0; l < levels; l++)
    {
        int hL = shapes[l * 2 + 0];
        int wL = shapes[l * 2 + 1];
        int start = levelStart[l];
        unsigned long long refOff = refBase + (unsigned long long)l * refLevelStride;
        float refX = refPoints[refOff + 0];
        float refY = refPoints[refOff + 1];
        float refW = 0.0f, refH = 0.0f;
        if (coords == 4)
        {
            refW = refPoints[refOff + 2];
            refH = refPoints[refOff + 3];
        }
        for (unsigned int p = 0; p < points; p++)
        {
            unsigned long long offIdx = ((((unsigned long long)(q * heads + h) * levels + l) * points + p)) * 2;
            float ox = sampOff[offIdx + 0], oy = sampOff[offIdx + 1];
            float locX, locY;
            if (coords == 2)
            {
                locX = refX + ox / (float)wL;
                locY = refY + oy / (float)hL;
            }
            else
            {
                locX = refX + ox / (float)points * refW * 0.5f;
                locY = refY + oy / (float)points * refH * 0.5f;
            }
            float weight = __expf(aBase[l * points + p] - amax) * invSum;

            // grid_sample bilinear, align_corners=false, zero padding
            float gx = 2.0f * locX - 1.0f, gy = 2.0f * locY - 1.0f;
            float ix = ((gx + 1.0f) * wL - 1.0f) * 0.5f;
            float iy = ((gy + 1.0f) * hL - 1.0f) * 0.5f;
            int x0 = (int)floorf(ix), y0 = (int)floorf(iy);
            int x1 = x0 + 1, y1 = y0 + 1;
            float wx1 = ix - x0, wx0 = 1.0f - wx1, wy1 = iy - y0, wy0 = 1.0f - wy1;

            float sample = 0.0f;
            bool x0ok = x0 >= 0 && x0 < wL, x1ok = x1 >= 0 && x1 < wL;
            bool y0ok = y0 >= 0 && y0 < hL, y1ok = y1 >= 0 && y1 < hL;
            unsigned long long chanOff = (unsigned long long)h * hd + c;
            if (y0ok && x0ok) sample += wy0 * wx0 * value[((unsigned long long)start + (unsigned long long)y0 * wL + x0) * hdHeads + chanOff];
            if (y0ok && x1ok) sample += wy0 * wx1 * value[((unsigned long long)start + (unsigned long long)y0 * wL + x1) * hdHeads + chanOff];
            if (y1ok && x0ok) sample += wy1 * wx0 * value[((unsigned long long)start + (unsigned long long)y1 * wL + x0) * hdHeads + chanOff];
            if (y1ok && x1ok) sample += wy1 * wx1 * value[((unsigned long long)start + (unsigned long long)y1 * wL + x1) * hdHeads + chanOff];
            acc += weight * sample;
        }
    }

    out[(unsigned long long)(q * heads + h) * hd + c] = acc;
}

} // extern "C"
