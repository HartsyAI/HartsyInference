// Language-model (decoder LLM) glue kernels — FP32.
//
// Net-new GPU ops the autoregressive decode loop needs that the DiT glue set does not
// cover. Each kernel keeps its inputs and output GPU-resident so the whole decode loop
// stays on-device (no cuStreamSynchronize + device-to-host copy per token).
//
// Convention: activations are FP32, contiguous, channels-last per the model code.
//
// Build:  ./build.sh   (nvcc --ptx -arch=sm_80 lm_f32.cu -o lm_f32.ptx, installed into Ptx/)

extern "C" {

// ── GQA K/V head repeat (block pattern) ────────────────────────────────────
// Expands grouped-query K or V from [B, Hkv, L, D] to [B, Hkv*group, L, D] so the
// head count matches the query heads before SDPA. Block layout (matches HF repeat_kv and
// Qwen2Attention.RepeatKvHeads): output head qh = h*group + g maps to input head h, i.e.
// the input head for output head qh is qh / group.
// One thread per output element.
__global__ void lm_repeat_kv_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    unsigned int kvHeads,
    unsigned int group,
    unsigned int seqLen,
    unsigned int headDim,
    unsigned long long total)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;

    unsigned int d = (unsigned int)(i % headDim);
    unsigned long long rem = i / headDim;
    unsigned int l = (unsigned int)(rem % seqLen);
    rem /= seqLen;
    unsigned int qHeads = kvHeads * group;
    unsigned int qh = (unsigned int)(rem % qHeads);
    unsigned long long b = rem / qHeads;

    unsigned int inH = qh / group;
    unsigned long long inIdx = (((b * kvHeads + inH) * seqLen) + l) * (unsigned long long)headDim + d;
    output[i] = input[inIdx];
}

} // extern "C"
