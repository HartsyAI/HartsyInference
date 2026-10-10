// Weighted sum of each token's routed-expert rows, plus the optional shared-expert row, in one launch:
//   out[t] = sharedGate(t) * shared[t] + sum_j topkWeight[t, j] * slotOut[t * topk + j]
// sharedGate is sigmoid(sharedGateLogit[t]) when the logits are given, else 1; with no shared row the first term is 0.
// The sum runs in slot order, so a launch is deterministic.
//
// Launch: grid = (tokens, ceil(hidden / 256)), block = 256.

extern "C" __global__ void moe_combine_slots_f32(
    float* __restrict__ out,
    const float* __restrict__ slotOut,
    const float* __restrict__ topkWeight,
    const float* __restrict__ shared,
    const float* __restrict__ sharedGateLogit,
    int hidden, int topk)
{
    const int t = blockIdx.x;
    const int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= hidden) return;

    float acc = 0.0f;
    if (shared != nullptr) {
        float s = shared[(size_t)t * hidden + c];
        if (sharedGateLogit != nullptr) s *= 1.0f / (1.0f + expf(-sharedGateLogit[t]));
        acc = s;
    }
    for (int j = 0; j < topk; ++j)
        acc += topkWeight[(size_t)t * topk + j] * slotOut[((size_t)t * topk + j) * hidden + c];
    out[(size_t)t * hidden + c] = acc;
}
