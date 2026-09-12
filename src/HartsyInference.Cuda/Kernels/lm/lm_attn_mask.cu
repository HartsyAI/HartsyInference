// Additive causal attention bias for the fused (cuDNN) prefill path: 0 where a query may attend a key,
// a large negative elsewhere. cuDNN's flash engine takes this as a Bias score-modifier and stays on its
// fused kernel, which is what makes a prefill-shaped attention cheap; the decode-tuned flash kernel costs
// ~6.8x more at 4096 tokens (64.8 ms vs 9.5 ms a layer, D=128, 16q/8kv, RTX 4090).
//
// Building it on the device matters as much as using it: at 8,664 tokens the mask is 75M floats, and the
// existing host-loop builder (TextEncoderTensorHelpers.BuildCausalMask) costs ~0.2-0.3 s to fill — more
// than the attention it would accelerate. Here it is one streaming write, and the caller caches it for the
// whole prefill so all layers share one buffer.
//
//   mask : [1, 1, rows, cols] F32, row-major
//
// Query row i sits at absolute position qOffset + i (nonzero when continuing a prefix), so key j is
// visible when j <= qOffset + i. A sliding window additionally hides keys older than `window` positions,
// matching the half-open convention the host builder uses: visible when j > qOffset + i - window.
// window = 0 means no window (plain causal).
//
// -1e30f rather than -INFINITY: the bias is added in fp32 and the fused softmax subtracts the row max, so
// a finite sentinel keeps a fully-masked row uniform instead of producing NaN. This matches the convention
// the materialized-mask path already uses.
//
// Launch: 1-D over rows*cols, 256 threads per block.

extern "C" __global__ void lm_causal_bias_mask_f32(
    float* __restrict__ mask,
    unsigned int rows,
    unsigned int cols,
    unsigned int qOffset,
    unsigned int window,
    unsigned long long total)
{
    unsigned long long gid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;

    unsigned int j = (unsigned int)(gid % (unsigned long long)cols);
    unsigned long long i = gid / (unsigned long long)cols;
    unsigned long long queryPos = (unsigned long long)qOffset + i;

    bool visible = (unsigned long long)j <= queryPos;
    if (visible && window != 0u)
    {
        // Guard the subtraction: an early query has fewer than `window` keys behind it, and unsigned
        // wraparound would hide the whole row.
        unsigned long long oldest = queryPos >= (unsigned long long)window ? queryPos - (unsigned long long)window : 0ull;
        visible = queryPos < (unsigned long long)window || (unsigned long long)j > oldest;
    }
    mask[gid] = visible ? 0.0f : -1e30f;
}
