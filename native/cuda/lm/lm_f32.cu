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

// ── Paged KV: contiguous time-range extraction ──────────────────────────────
// output[1,h,t,d] = input[1,h,start+t,d] for t in [0,len). Mirror of lm_kv_append_f32's per-head
// addressing but reading a sub-range instead of writing one — used by the paged KV cache to (a) split a
// multi-token append across page boundaries and (b) gather a partially-filled page's occupied prefix
// before copying it into the contiguous scratch buffer FlashAttention expects. One thread per output element.
__global__ void lm_kv_slice_time_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    unsigned int heads,
    unsigned int tIn,
    unsigned int headDim,
    unsigned int start,
    unsigned int len,
    unsigned long long total)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;

    unsigned int d = (unsigned int)(i % headDim);
    unsigned long long rem = i / headDim;
    unsigned int t = (unsigned int)(rem % len);
    unsigned int h = (unsigned int)(rem / len);

    unsigned long long inIdx = (((unsigned long long)h * tIn) + (start + t)) * headDim + d;
    output[i] = input[inIdx];
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
// IBackend.ApplyRopeInterleaved — the Sesame CSM / HeartMuLa backbone+decoder RoPE, and (with
// partial rotary) GLM-4. One thread per (vector, pair); keeps q/k GPU-resident through RoPE (no
// host round-trip per layer). rotaryDim: 0 (or >=headDim) rotates every pair (full rotary, prior
// behavior); otherwise only pairs entirely inside [0, rotaryDim) rotate — dims [rotaryDim, headDim)
// pass through untouched, matching HF's `q_rot, q_pass = q[...,:rotary_dim], q[...,rotary_dim:]`
// partial-rotary split (GLM-4's `partial_rotary_factor`). This guard was missing before 2026-07-22:
// the kernel always used headDim/2 pairs regardless of rotaryDim, so any interleaved-RoPE arch with
// partial rotary had its un-rotated tail spuriously rotated with wrong (repeated-block) frequencies
// instead of being left alone — a no-op at position 0 (angle=0) that corrupts every later position.
__global__ void lm_rope_interleaved_f32(
    float* __restrict__ x,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    unsigned int numHeads,
    unsigned int headDim,
    unsigned int rotaryDim,
    unsigned long long totalVecs)
{
    unsigned int half = headDim >> 1;
    unsigned long long gid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = totalVecs * (unsigned long long)half;
    if (gid >= total) return;
    unsigned int i = (unsigned int)(gid % half);
    unsigned int rdim = (rotaryDim == 0u || rotaryDim > headDim) ? headDim : rotaryDim;
    if (2u * i >= rdim) return;
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

// ── Graph-capture decode: device-indexed RoPE ───────────────────────────────
// Applies rotary embedding to a single decode-step Q or K tensor [1, numHeads, 1, headDim] by
// reading the rotation ROW from a precomputed FULL table [maxPos, headDim] (duplicated-half
// layout, built once host-side via RopeFrequencyBuilder — identical math to GenericTransformer's
// BuildRope, just for every position up front instead of one small per-step tensor) at the
// position in devicePos[1] (qOffset — the position of the token this decode step is embedding;
// same buffer/convention as lm_kv_append_f32/lm_flash_attn_f32's dPos). This removes RoPE's
// per-token host Math.Cos/Sin + H2D upload (one of the two host rebuilds — with embed gather —
// that blocked graph capture; see LLM_DECODE_PERF_GRIND.md Phase 6). Two entry points (not a
// runtime branch) mirroring the existing split-half (dit_rope_f32) vs interleaved
// (lm_rope_interleaved_f32) kernels this replaces for the graphed decode path only.
__global__ void lm_rope_decode_splithalf(
    float* __restrict__ x,
    const float* __restrict__ cosTable,
    const float* __restrict__ sinTable,
    unsigned int numHeads,
    unsigned int headDim,
    unsigned int rotaryDim,
    const int* __restrict__ devicePos)
{
    unsigned int rdim = (rotaryDim == 0u || rotaryDim > headDim) ? headDim : rotaryDim;
    unsigned int half = rdim >> 1;
    unsigned long long gid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = (unsigned long long)numHeads * half;   // totalVecs = numHeads (B=1, Tq=1)
    if (gid >= total) return;
    unsigned int i = (unsigned int)(gid % half);
    unsigned long long h = gid / half;
    int pos = devicePos[1];
    size_t baseX = h * headDim;
    size_t baseCs = (size_t)pos * headDim;
    float lower = x[baseX + i];
    float upper = x[baseX + i + half];
    x[baseX + i] = lower * cosTable[baseCs + i] - upper * sinTable[baseCs + i];
    x[baseX + i + half] = upper * cosTable[baseCs + i + half] + lower * sinTable[baseCs + i + half];
}

__global__ void lm_rope_decode_interleaved(
    float* __restrict__ x,
    const float* __restrict__ cosTable,
    const float* __restrict__ sinTable,
    unsigned int numHeads,
    unsigned int headDim,
    const int* __restrict__ devicePos)
{
    unsigned int half = headDim >> 1;
    unsigned long long gid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = (unsigned long long)numHeads * half;
    if (gid >= total) return;
    unsigned int i = (unsigned int)(gid % half);
    unsigned long long h = gid / half;
    int pos = devicePos[1];
    size_t baseX = h * headDim + 2u * i;
    size_t baseCs = (size_t)pos * headDim + i;
    float xe = x[baseX];
    float xo = x[baseX + 1];
    float c = cosTable[baseCs];
    float s = sinTable[baseCs];
    x[baseX]     = xe * c - xo * s;
    x[baseX + 1] = xo * c + xe * s;
}

// ── Graph-capture decode: device-indexed embedding gather ───────────────────
// output[hidden] = embedTable[tokenId, hidden]. tokenId comes from a persistent 1-int device
// buffer (the SAME buffer lm_argmax_lastdim_f32 writes into for the previous step — greedy
// decode chains entirely on-device: step N's argmax output is step N+1's embed input, with no
// host round-trip). embedTable must already be GPU-resident (preloaded weight). One thread per
// hidden channel.
__global__ void lm_embed_gather_decode_f32(
    float* __restrict__ output,
    const float* __restrict__ embedTable,
    const int* __restrict__ tokenId,
    unsigned int hidden)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden) return;
    output[i] = embedTable[(size_t)tokenId[0] * hidden + i];
}

// ── Graph-capture decode: repetition-penalty history tracking ───────────────
// Appends tokenId[0] (the token this step is about to embed — same buffer lm_embed_gather_decode_f32
// reads) into history[*historyCount], then increments the counter. Runs once per graph replay so the
// device history buffer tracks exactly the same token sequence CPU-side eager decode passes to
// RepetitionPenaltyStep (SamplerChain.Next appends the current token to `generated` before sampling
// the next one — see TextGenerationPipeline.GenerateGraphDecode). Single thread: this is a handful of
// int writes per step, not a bandwidth-bound op, and a fixed 1×1×1 launch keeps it graph-replay-safe.
__global__ void lm_history_append(
    int* __restrict__ history,
    int* __restrict__ historyCount,
    const int* __restrict__ tokenId)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int count = *historyCount;
    history[count] = tokenId[0];
    *historyCount = count + 1;
}

// ── Graph-capture decode: on-device repetition penalty ───────────────────────
// Applies HF-convention repetition penalty (divide positive logits, multiply negative) to
// logits[history[i]] for every i in [0, *historyCount), SEQUENTIALLY and single-threaded — this
// exactly replicates RepetitionPenaltyStep.Apply's CPU semantics byte-for-byte, including a token
// that occurs more than once in history being penalized cumulatively on each occurrence (order-
// dependent: the second application divides/multiplies the ALREADY-penalized value). A parallel
// scatter across history entries would race on repeated tokens and diverge from the CPU reference,
// so single-thread is a correctness choice here, not just simplicity — the loop itself (at most
// maxSeqLen iterations of one global read + one global write) is microseconds, nowhere near the
// per-step cost of the transformer forward pass it runs alongside.
__global__ void lm_repetition_penalty_f32(
    float* __restrict__ logits,
    const int* __restrict__ history,
    const int* __restrict__ historyCount,
    float penalty,
    unsigned int vocabSize)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int count = *historyCount;
    for (int i = 0; i < count; ++i)
    {
        int token = history[i];
        if ((unsigned int)token >= vocabSize) continue;
        float logit = logits[token];
        logits[token] = logit > 0.0f ? logit / penalty : logit * penalty;
    }
}

} // extern "C"
