// Head-major rotary embedding, division-free indexing.
//
// Lives in its own module rather than dit_f32.cu so rebuilding it cannot perturb that file's other 40
// kernels: the committed dit_f32.ptx was produced by a different nvcc than the one available here, so
// regenerating it would rewrite codegen for every model that uses it.
//
// The v1 kernel in dit_f32.cu derives its indices with THREE 64-bit integer div/mod per thread
// (gid/half, vec/(heads*seq), vec%seq). At 56 heads x 6550 tokens x 48 pairs that is 17.6M threads
// each running an emulated 64-bit division sequence, against only ~282 MB of actual traffic. This
// version carries the token in blockIdx.x and the (batch,head) pair in blockIdx.y, leaving one
// block-uniform 32-bit divide for the batch index.
//
// Arithmetic is expression-for-expression identical to v1, so results are bit-exact.

extern "C" __global__ void dit_rope_head_major_v2_f32(
    float* __restrict__ x,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    unsigned int headDim,
    unsigned int half,
    unsigned int seq,
    unsigned int heads,
    unsigned int padShift)
{
    unsigned int padHalf = 1u << padShift;
    unsigned int i = threadIdx.x & (padHalf - 1u);
    unsigned int slot = threadIdx.x >> padShift;
    unsigned int s = blockIdx.x * (blockDim.x >> padShift) + slot;
    if (i >= half || s >= seq) return;

    unsigned int hb = blockIdx.y;
    unsigned int b = hb / heads;
    size_t vec = (size_t)hb * seq + s;
    size_t baseX = vec * headDim;
    size_t baseCs = ((size_t)b * seq + s) * headDim;

    float lower = x[baseX + i];
    float upper = x[baseX + i + half];
    x[baseX + i] = lower * cos[baseCs + i] - upper * sin[baseCs + i];
    x[baseX + i + half] = upper * cos[baseCs + i + half] + lower * sin[baseCs + i + half];
}
