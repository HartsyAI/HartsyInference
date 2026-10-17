// Guided semantic logits for MiniMax Music 3's AR step, one row per block. The candidate set is the conditional row's top-K:
// every candidate gets u + (c - u) * g with the host's operation order, and every other entry is -inf. The threshold is the
// K-th largest conditional value, so ties at the threshold stay candidates, as in the host BlendGuided. Round-to-nearest
// intrinsics stop the compiler from contracting the blend into an FMA, which would change the last bit.
#define MM3_TOPK_MAX_COUNT 16448   // 16384 codes plus end-of-audio, with room for the rounding of the shared mask
#define MM3_TOPK_BLOCK 256
#define MM3_NEG_INF __int_as_float(0xff800000)   // -inf without the math header, which NVRTC does not provide

extern "C" __global__ void lm_cfg_topk_guided_f32(
    float* __restrict__ output,
    const float* __restrict__ cond,
    const float* __restrict__ uncond,
    int count,
    int topK,
    float guidance)
{
    __shared__ unsigned char taken[MM3_TOPK_MAX_COUNT];
    __shared__ float bestValue[MM3_TOPK_BLOCK];
    __shared__ int bestIndex[MM3_TOPK_BLOCK];
    __shared__ float threshold;

    const int tid = threadIdx.x;
    for (int i = tid; i < count; i += MM3_TOPK_BLOCK) taken[i] = 0;
    __syncthreads();

    // K passes of block-wide argmax over the untaken conditional values. The last pick is the K-th largest value.
    for (int pass = 0; pass < topK; ++pass) {
        float localValue = MM3_NEG_INF;
        int localIndex = -1;
        for (int i = tid; i < count; i += MM3_TOPK_BLOCK) {
            if (taken[i]) continue;
            const float v = cond[i];
            if (localIndex < 0 || v > localValue) { localValue = v; localIndex = i; }
        }
        bestValue[tid] = localValue;
        bestIndex[tid] = localIndex;
        __syncthreads();
        for (int stride = MM3_TOPK_BLOCK / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const int other = tid + stride;
                const bool otherWins = bestIndex[other] >= 0 &&
                    (bestIndex[tid] < 0 || bestValue[other] > bestValue[tid] ||
                     (bestValue[other] == bestValue[tid] && bestIndex[other] < bestIndex[tid]));
                if (otherWins) { bestValue[tid] = bestValue[other]; bestIndex[tid] = bestIndex[other]; }
            }
            __syncthreads();
        }
        if (tid == 0) {
            taken[bestIndex[0]] = 1;
            threshold = bestValue[0];
        }
        __syncthreads();
    }

    for (int i = tid; i < count; i += MM3_TOPK_BLOCK) {
        const float c = cond[i];
        const float u = uncond[i];
        output[i] = c < threshold ? MM3_NEG_INF : __fadd_rn(u, __fmul_rn(__fsub_rn(c, u), guidance));
    }
}
