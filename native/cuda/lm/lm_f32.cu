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

// ── KV cache in-place append ────────────────────────────────────────────────
// Writes newKv [1, H, tNew, D] into a fixed-capacity buffer [1, H, maxSeq, D] at sequence offset, so a KV
// cache can grow without reallocating (the Concat-grown cache is O(n^2); this is O(tNew) per step).
// One thread per source element.
__global__ void lm_kv_append_f32(
    float* __restrict__ buffer,
    const float* __restrict__ newKv,
    unsigned int heads,
    unsigned int maxSeq,
    unsigned int tNew,
    unsigned int headDim,
    unsigned int offset,
    unsigned long long total,
    const int* __restrict__ dPos)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;

    // Graph-capture decode: the write slot comes from a device buffer {kvLen, qOffset}; qOffset (dPos[1]) is
    // the position to append at. Null → use the host-passed offset (eager path / all other callers).
    if (dPos != nullptr) offset = (unsigned int)dPos[1];

    unsigned int d = (unsigned int)(i % headDim);
    unsigned long long rem = i / headDim;
    unsigned int ti = (unsigned int)(rem % tNew);
    unsigned int h = (unsigned int)(rem / tNew);

    unsigned long long bufIdx = (((unsigned long long)h * maxSeq) + (offset + ti)) * headDim + d;
    buffer[bufIdx] = newKv[i];
}

// ── MoE expert dispatch: gather + weighted scatter-add ──────────────────────
// gather: output[m, j] = input[rowIndices[m], j]   (collect an expert's routed tokens into a contiguous group)
__global__ void lm_gather_rows_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const int* __restrict__ rowIndices,
    unsigned int K,
    unsigned long long total)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned int j = (unsigned int)(i % K);
    unsigned int m = (unsigned int)(i / K);
    output[i] = input[(unsigned long long)rowIndices[m] * K + j];
}

// scatter-add: output[rowIndices[m], j] += scales[m] * input[m, j]   (combine an expert's weighted output back)
// Indices within a call are distinct, so no atomics; call once per expert into the same (accumulating) output.
__global__ void lm_scatter_add_weighted_rows_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const int* __restrict__ rowIndices,
    const float* __restrict__ scales,
    unsigned int K,
    unsigned long long total)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned int j = (unsigned int)(i % K);
    unsigned int m = (unsigned int)(i / K);
    output[(unsigned long long)rowIndices[m] * K + j] += scales[m] * input[i];
}

// ── Interleaved (GPT-J) rotary embedding, in-place ─────────────────────────
// x [B, L, numHeads, headDim]; rotates adjacent pairs (2i, 2i+1) sharing frequency i (cos/sin
// index i, NOT the split-half convention of lm/dit_rope). cos/sin are [B, L, headDim] (only the
// first headDim/2 entries per position are read); a vector's cos/sin row is vec/numHeads. Matches
// IBackend.ApplyRopeInterleaved — the Sesame CSM / HeartMuLa backbone+decoder RoPE. One thread per
// (vector, pair); keeps q/k GPU-resident through RoPE (no host round-trip per layer).
__global__ void lm_rope_interleaved_f32(
    float* __restrict__ x,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    unsigned int numHeads,
    unsigned int headDim,
    unsigned long long totalVecs)
{
    unsigned int half = headDim >> 1;
    unsigned long long gid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = totalVecs * (unsigned long long)half;
    if (gid >= total) return;
    unsigned int i = (unsigned int)(gid % half);
    unsigned long long vec = gid / half;
    unsigned long long row = vec / numHeads;
    size_t baseX = (size_t)vec * headDim + 2u * i;
    size_t baseCs = (size_t)row * headDim + i;
    float xe = x[baseX];
    float xo = x[baseX + 1];
    float c = cos[baseCs];
    float s = sin[baseCs];
    x[baseX]     = xe * c - xo * s;
    x[baseX + 1] = xo * c + xe * s;
}

// ── Per-row argmax over the last dim ────────────────────────────────────────
// indices[r] = argmax_c input[r, c]   (first max on ties, matching the CPU reference).
// One block per row; each thread scans a strided slice, then a tree reduction keeps the
// (value, index) pair with the lowest index on ties. Keeps greedy token sampling GPU-resident
// so only the chosen index (one int per row) leaves the device, not the C-wide logit vector.
// Launch: grid = rows, block = min(C, 256). Shared mem = blockDim.x * (sizeof(float)+sizeof(int)).
__global__ void lm_argmax_lastdim_f32(
    int* __restrict__ indices,
    const float* __restrict__ input,
    unsigned int C)
{
    extern __shared__ unsigned char smem[];
    float* sVal = (float*)smem;
    int* sIdx = (int*)(sVal + blockDim.x);

    unsigned int row = blockIdx.x;
    const float* in = input + (unsigned long long)row * C;

    float bestVal = -3.402823466e38f;   // -FLT_MAX
    int bestIdx = 0;
    for (unsigned int c = threadIdx.x; c < C; c += blockDim.x)
    {
        float v = in[c];
        if (v > bestVal) { bestVal = v; bestIdx = (int)c; }
    }
    sVal[threadIdx.x] = bestVal;
    sIdx[threadIdx.x] = bestIdx;
    __syncthreads();

    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            float ov = sVal[threadIdx.x + stride];
            int oi = sIdx[threadIdx.x + stride];
            float cv = sVal[threadIdx.x];
            int ci = sIdx[threadIdx.x];
            // Take the larger value; on an exact tie take the smaller index (matches the CPU "first max").
            // The lower index is NOT always the left partner after the strided scan, so compare indices explicitly.
            if (ov > cv || (ov == cv && oi < ci))
            {
                sVal[threadIdx.x] = ov;
                sIdx[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) indices[row] = sIdx[0];
}

} // extern "C"
