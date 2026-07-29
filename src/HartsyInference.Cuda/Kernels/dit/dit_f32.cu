// DiT (diffusion transformer) glue kernels — FP32.
//
// These replace the per-op CPU pointer loops in the Ideogram 4 (and other DiT) forward
// path that were forcing a cuStreamSynchronize + device-to-host copy on every block. Each
// kernel keeps its inputs and output GPU-resident so the whole denoise loop stays on-device.
//
// Convention: the primary activation tensor is FP32; per-channel parameter vectors
// (rms weight, modulation scale/gate) are also FP32 — that is how the model code allocates
// them. FP32 accumulation throughout.
//
// Build:  ./build.sh   (nvcc --ptx -arch=sm_80 dit_f32.cu -o dit_f32.ptx, installed into Ptx/)

extern "C" {

// ── RMSNorm ──────────────────────────────────────────────────────────────
// out[row, i] = in[row, i] * rsqrt(mean(in[row,:]^2) + eps) * weight[i]
// One block per row; blockDim.x threads stride over normDim. Shared-mem block reduction.
// Also serves per-head QK-RMSNorm: pass rows = B*L*numHeads, normDim = headDim.
__global__ void dit_rmsnorm_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    unsigned int normDim,
    unsigned int totalRows,
    float eps)
{
    extern __shared__ float sdata[];
    unsigned int row = blockIdx.x;
    if (row >= totalRows) return;

    const float* inRow = input + (size_t)row * normDim;
    float* outRow = output + (size_t)row * normDim;

    float partial = 0.0f;
    for (unsigned int i = threadIdx.x; i < normDim; i += blockDim.x)
    {
        float v = inRow[i];
        partial += v * v;
    }
    sdata[threadIdx.x] = partial;
    __syncthreads();

    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    float invRms = rsqrtf(sdata[0] / (float)normDim + eps);
    for (unsigned int i = threadIdx.x; i < normDim; i += blockDim.x)
        outRow[i] = inRow[i] * invRms * weight[i];
}

// ── Broadcast affine over the last dim ─────────────────────────────────────
// input/output are [B, S, D] (row-major). scale and (optional) shift are [B, D],
// broadcast over the S (sequence) axis. With shift == null this is a pure broadcast
// multiply — covers Ideogram 4 ApplyScale (modulation, scale-only).
// out[b,s,d] = in[b,s,d] * scale[b,d] + (shift ? shift[b,d] : 0)
__global__ void dit_affine_broadcast_lastdim_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ scale,
    const float* __restrict__ shift,
    unsigned int seqLen,
    unsigned int dim,
    unsigned long long total)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;

    unsigned int d = (unsigned int)(i % dim);
    unsigned long long row = i / dim;
    unsigned int b = (unsigned int)(row / seqLen);
    size_t pIdx = (size_t)b * dim + d;

    float v = input[i] * scale[pIdx];
    if (shift != 0) v += shift[pIdx];
    output[i] = v;
}

// ── Gated residual over the last dim ───────────────────────────────────────
// out[b,s,d] = residual[b,s,d] + gate[b,d] * value[b,s,d]
// gate is [B, D] broadcast over S. Covers Ideogram 4 ApplyGatedResidual.
__global__ void dit_gated_residual_lastdim_f32(
    float* __restrict__ output,
    const float* __restrict__ residual,
    const float* __restrict__ value,
    const float* __restrict__ gate,
    unsigned int seqLen,
    unsigned int dim,
    unsigned long long total)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;

    unsigned int d = (unsigned int)(i % dim);
    unsigned long long row = i / dim;
    unsigned int b = (unsigned int)(row / seqLen);
    size_t pIdx = (size_t)b * dim + d;

    output[i] = residual[i] + gate[pIdx] * value[i];
}

// ── AdaLN modulation split (scale-only, tanh-gated) ─────────────────────────
// proj is [B, 4*D] = chunk(scale_msa, gate_msa, scale_mlp, gate_mlp). Writes four [B, D]
// tensors applying (1 + x) to scales and tanh(x) to gates, matching Ideogram 4's
// ComputeModulation. One thread per (b, d).
__global__ void dit_modulation4_f32(
    float* __restrict__ scaleMsa,
    float* __restrict__ gateMsa,
    float* __restrict__ scaleMlp,
    float* __restrict__ gateMlp,
    const float* __restrict__ proj,
    unsigned int dim,
    unsigned int batch)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = batch * dim;
    if (idx >= total) return;

    unsigned int b = idx / dim;
    unsigned int d = idx % dim;
    size_t src = (size_t)b * 4u * dim;

    scaleMsa[idx] = 1.0f + proj[src + d];
    gateMsa[idx] = tanhf(proj[src + dim + d]);
    scaleMlp[idx] = 1.0f + proj[src + 2u * dim + d];
    gateMlp[idx] = tanhf(proj[src + 3u * dim + d]);
}

// ── CFG combine + Euler step (in-place on z) ────────────────────────────────
// v = guidance * pos + (1 - guidance) * neg ;  z += v * delta
// Operates on flat [N]. z is the latent carried across denoise steps (kept resident).
__global__ void dit_cfg_euler_f32(
    float* __restrict__ z,
    const float* __restrict__ pos,
    const float* __restrict__ neg,
    float guidance,
    float delta,
    unsigned int count)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float v = guidance * pos[i] + (1.0f - guidance) * neg[i];
    z[i] = z[i] + v * delta;
}

// ── Tanh (general elementwise) ──────────────────────────────────────────────
__global__ void dit_tanh_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    unsigned int count)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    output[i] = tanhf(input[i]);
}

// ── Rotary position embedding (in-place, rotate_half convention) ────────────
// x is [B, L, numHeads, headDim] flattened to [totalVecs, headDim] (totalVecs = B*L*numHeads).
// cos/sin are [B, L, headDim] broadcast over heads (cos row = vec / numHeads).
// out[i]      = lower*cos[i]      - upper*sin[i]        (i in [0, half))
// out[i+half] = upper*cos[i+half] + lower*sin[i+half]
// where lower = x[i], upper = x[i+half] (originals). Each thread owns one i and writes both
// halves from originals — race-free, no snapshot needed. Matches Ideogram4Mrope.ApplyOne.
// rotaryDim: number of leading dims of each head to rotate (NEOX pairing (i, i+rotaryDim/2)); the rest pass
// through. 0 (or >= headDim) means full rotary — the default, identical to the original kernel (DiT path).
__global__ void dit_rope_f32(
    float* __restrict__ x,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    unsigned int numHeads,
    unsigned int headDim,
    unsigned long long totalVecs,
    unsigned int rotaryDim)
{
    unsigned int rdim = (rotaryDim == 0u || rotaryDim > headDim) ? headDim : rotaryDim;
    unsigned int half = rdim >> 1;   // rotate pairs (i, i+half) for i < half; cos/sin stride stays headDim
    unsigned long long gid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = totalVecs * (unsigned long long)half;
    if (gid >= total) return;

    unsigned int i = (unsigned int)(gid % half);
    unsigned long long vec = gid / half;
    unsigned long long row = vec / numHeads;
    size_t baseX = (size_t)vec * headDim;
    size_t baseCs = (size_t)row * headDim;

    float lower = x[baseX + i];
    float upper = x[baseX + i + half];
    x[baseX + i] = lower * cos[baseCs + i] - upper * sin[baseCs + i];
    x[baseX + i + half] = upper * cos[baseCs + i + half] + lower * sin[baseCs + i + half];
}

// ── Per-row scalar multiply (token masking) ────────────────────────────────
// out[row, c] = in[row, c] * rowScale[row]. rowScale has one value per row (rows = total/channels).
// Used to zero non-role token positions in Ideogram 4's masked text/image embedding.
__global__ void dit_row_scale_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ rowScale,
    unsigned int channels,
    unsigned long long total)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned long long row = i / channels;
    output[i] = input[i] * rowScale[row];
}

// ── Add scalar (out = in + c) ───────────────────────────────────────────────
__global__ void dit_add_scalar_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    float c,
    unsigned int count)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    output[i] = input[i] + c;
}

// ── Non-affine LayerNorm ────────────────────────────────────────────────────
// Per row: normalize to zero mean, unit variance (biased var, /dim). No scale/bias.
// One block per row; shared-mem reduction for mean and variance.
__global__ void dit_layernorm_noaffine_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    unsigned int dim,
    unsigned int totalRows,
    float eps)
{
    extern __shared__ float sdata[];
    unsigned int row = blockIdx.x;
    if (row >= totalRows) return;

    const float* inRow = input + (size_t)row * dim;
    float* outRow = output + (size_t)row * dim;

    float partial = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x)
        partial += inRow[i];
    sdata[threadIdx.x] = partial;
    __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float mean = sdata[0] / (float)dim;
    __syncthreads();

    float vpart = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x)
    {
        float diff = inRow[i] - mean;
        vpart += diff * diff;
    }
    sdata[threadIdx.x] = vpart;
    __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float invStd = rsqrtf(sdata[0] / (float)dim + eps);
    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x)
        outRow[i] = (inRow[i] - mean) * invStd;
}

// ── Index-add of embedding rows (in-place) ──────────────────────────────────
// h[row, d] += table[indices[row], d]. Keeps the indicator embedding a GPU-resident weight
// (no host gather), one of two rows selected per token. total = totalRows * dim.
__global__ void dit_index_add_f32(
    float* __restrict__ h,
    const float* __restrict__ table,
    const int* __restrict__ indices,
    unsigned int dim,
    unsigned long long total)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned int d = (unsigned int)(i % dim);
    unsigned long long row = i / dim;
    h[i] += table[(size_t)indices[row] * dim + d];
}

// ── Scatter rows after a zeroed head block ──────────────────────────────────
// output = [ zeros(headRows), input ] along the row axis. out[row,d] = row < headRows ? 0
// : input[(row-headRows)*dim + d]. Builds Ideogram 4's conditional latent (text rows zeroed,
// image rows = current latent z).
__global__ void dit_scatter_rows_after_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    unsigned int headRows,
    unsigned int dim,
    unsigned long long total)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned long long row = i / dim;
    if (row < headRows) { output[i] = 0.0f; return; }
    unsigned int d = (unsigned int)(i % dim);
    output[i] = input[(size_t)(row - headRows) * dim + d];
}

// ── Slice a contiguous row block (by element offset) ────────────────────────
// output[i] = input[elemOffset + i]. Extracts image-token velocity rows from a full sequence.
__global__ void dit_slice_rows_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    unsigned long long elemOffset,
    unsigned long long total)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    output[i] = input[elemOffset + i];
}

// ── Slice / gather over the last dim ───────────────────────────────────────
// out[row, d] = in[row, offset + d], for d in [0, outDim). in row stride = inDim.
// Splits a fused [.., inDim] tensor (e.g. QKV [B,L,3H]) into a contiguous [.., outDim] chunk.
__global__ void dit_slice_lastdim_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    unsigned int outDim,
    unsigned int inDim,
    unsigned int offset,
    unsigned long long total)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned int d = (unsigned int)(i % outDim);
    unsigned long long row = i / outDim;
    output[i] = input[(size_t)row * inDim + offset + d];
}

// ── LTX-2 "split" rotary (rotate-half, per-head cos) ───────────────────────
// LTX-2.3 (rope_type=split): x is [S, dim] = [S, numHeads, headDim]; cos/sin are [S, dim/2] = [S, numHeads, r]
// with r=headDim/2 (per-head, ONE angle shared by both elements of a pair). For pair i in head h at token s:
//   out[i]   = a*c - b*sn ; out[i+r] = b*c + a*sn ,  a=x[i], b=x[i+r], c=cos[i], sn=sin[i]
// Distinct from dit_rope_f32 (per-position cos, dual-angle) and wan_rope_interleaved (adjacent-pair). Keeps LTX-2.3
// Q/K GPU-resident through RoPE — the host loop D2H'd + re-uploaded [S,dim] Q/K per attention, and on this
// block-swap-bound 22B model those re-uploads fight the 19 GB/forward weight stream on PCIe.
__global__ void ltx2_split_rope_f32(
    float* __restrict__ x,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    unsigned int seqLen,
    unsigned int numHeads,
    unsigned int headDim)
{
    unsigned int r = headDim >> 1;
    unsigned int dim = numHeads * headDim;
    unsigned int cosWidth = dim >> 1;            // = numHeads * r
    unsigned long long total = (unsigned long long)seqLen * numHeads * r;
    unsigned long long gid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    unsigned int i = (unsigned int)(gid % r);
    unsigned long long t = gid / r;
    unsigned int h = (unsigned int)(t % numHeads);
    unsigned int s = (unsigned int)(t / numHeads);
    size_t headBase = (size_t)s * dim + (size_t)h * headDim;
    size_t cosBase = (size_t)s * cosWidth + (size_t)h * r;
    float a = x[headBase + i];
    float b = x[headBase + i + r];
    float c = cos[cosBase + i], sn = sin[cosBase + i];
    x[headBase + i]     = a * c - b * sn;
    x[headBase + i + r] = b * c + a * sn;
}


// ── Unpatchify: token grid [hP·wP, C·p²] → pixel latent [C, H, W] (B=1) ─────
// innerChannelFastest=1 → token inner order (ph, pw, c) (Z-Image / Lumina2 patchify);
// 0 → channel-outer (c, ph, pw) (Krea2 / diffusers view+permute+reshape).
__global__ void dit_unpatchify_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    unsigned int channels,
    unsigned int hPacked,
    unsigned int wPacked,
    unsigned int patch,
    unsigned int innerChannelFastest,
    unsigned long long total)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned int W = wPacked * patch;
    unsigned int H = hPacked * patch;
    unsigned int x = (unsigned int)(i % W);
    unsigned long long t = i / W;
    unsigned int y = (unsigned int)(t % H);
    unsigned int c = (unsigned int)(t / H);
    unsigned int hp = y / patch, ph = y % patch;
    unsigned int wp = x / patch, pw = x % patch;
    unsigned long long seq = (unsigned long long)hp * wPacked + wp;
    unsigned int patchVol = channels * patch * patch;
    unsigned long long inner = innerChannelFastest
        ? ((unsigned long long)(ph * patch + pw) * channels + c)
        : (((unsigned long long)c * patch + ph) * patch + pw);
    output[i] = input[seq * patchVol + inner];
}


// ── MoE top-k gate weights (HiDream MoEGate) ────────────────────────────────
// Per token: softmax(logits over E experts), select top-k by prob, renormalize the selected
// weights to sum to 1 (norm_topk_prob: w_k / (sum_k w_k + 1e-20)). Dense [tokens, E] output with
// zeros at non-selected slots so a dense per-expert accumulation reproduces sparse dispatch
// exactly. One thread per token; E is small (4 for HiDream) so registers suffice (E <= 16).
__global__ void dit_moe_topk_gate_f32(
    float* __restrict__ weights,
    const float* __restrict__ logits,
    unsigned int numExperts,
    unsigned int topK,
    unsigned long long totalTokens)
{
    unsigned long long t = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= totalTokens) return;
    const float* lp = logits + t * numExperts;
    float* wp = weights + t * numExperts;

    float probs[16];
    float maxLogit = lp[0];
    for (unsigned int e = 1; e < numExperts; e++)
        maxLogit = fmaxf(maxLogit, lp[e]);
    float sum = 0.0f;
    for (unsigned int e = 0; e < numExperts; e++)
    {
        float ex = expf(lp[e] - maxLogit);
        probs[e] = ex;
        sum += ex;
    }
    for (unsigned int e = 0; e < numExperts; e++)
    {
        probs[e] /= sum;
        wp[e] = 0.0f;
    }

    float denom = 0.0f;
    unsigned int k = topK < numExperts ? topK : numExperts;
    for (unsigned int kk = 0; kk < k; kk++)
    {
        int best = -1;
        float bestVal = -1.0f;
        for (unsigned int e = 0; e < numExperts; e++)
        {
            if (wp[e] != 0.0f) continue;
            if (probs[e] > bestVal) { bestVal = probs[e]; best = (int)e; }
        }
        wp[best] = bestVal;
        denom += bestVal;
    }
    denom += 1e-20f;
    for (unsigned int e = 0; e < numExperts; e++)
        if (wp[e] != 0.0f) wp[e] /= denom;
}

// ── Per-row (per-token) gated accumulate (MoE expert combine) ───────────────
// out[b,s,:] += gate[(b,s), expertIdx] * val[b,s,:]. gate is the dense [tokens, E] weight table from
// dit_moe_topk_gate_f32; non-selected tokens carry weight 0 and contribute nothing. In-place on out.
__global__ void dit_row_gated_accum_f32(
    float* __restrict__ out,
    const float* __restrict__ val,
    const float* __restrict__ gate,
    unsigned int numExperts,
    unsigned int expertIdx,
    unsigned int dim,
    unsigned long long total)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned long long row = i / dim;
    float g = gate[row * numExperts + expertIdx];
    if (g != 0.0f)
        out[i] += g * val[i];
}

// ── Oasis spatio-temporal attention layout (device ports of the host loops) ──
// Frame-major tokens qkv[token, 3*dim] (token = f*sp + i) → out[b, heads, seq, headDim].
// spatial (temporal=0): b = frame f, s = spatial i, seq = sp
// temporal(temporal=1): b = spatial i, s = frame f, seq = frames
// One thread per output element (token, h, d) over the sliced `part` (q=0/k=1/v=2).
__global__ void oasis_split_heads_f32(
    float* __restrict__ out,
    const float* __restrict__ qkv,
    unsigned int frames, unsigned int sp, unsigned int heads, unsigned int headDim,
    unsigned int part, unsigned int temporal)
{
    unsigned int dim = heads * headDim;
    unsigned long long total = (unsigned long long)frames * sp * dim;
    unsigned long long gid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    unsigned int d = (unsigned int)(gid % headDim);
    unsigned long long tmp = gid / headDim;
    unsigned int h = (unsigned int)(tmp % heads);
    unsigned long long token = tmp / heads;
    unsigned int f = (unsigned int)(token / sp);
    unsigned int i = (unsigned int)(token % sp);
    unsigned int b = temporal ? i : f;
    unsigned int s = temporal ? f : i;
    unsigned int seq = temporal ? frames : sp;
    unsigned long long srcIdx = token * 3ULL * dim + (unsigned long long)part * dim + (unsigned long long)h * headDim + d;
    unsigned long long dstIdx = (((unsigned long long)b * heads + h) * seq + s) * headDim + d;
    out[dstIdx] = qkv[srcIdx];
}

// Interleaved (2i, 2i+1) partial rotary on x[b, heads, seq, headDim]; cos/sin are [seq, rotDim].
// Rotates only the first `rotDim` dims of each head (rest pass through). One thread per (b,h,s,pair).
__global__ void oasis_rope_interleaved_f32(
    float* __restrict__ x,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    unsigned int batch, unsigned int heads, unsigned int seq, unsigned int headDim, unsigned int rotDim)
{
    unsigned int pairs = rotDim >> 1;
    unsigned long long total = (unsigned long long)batch * heads * seq * pairs;
    unsigned long long gid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    unsigned int p = (unsigned int)(gid % pairs);
    unsigned long long tmp = gid / pairs;
    unsigned int s = (unsigned int)(tmp % seq);
    unsigned long long bh = tmp / seq;              // b*heads + h
    unsigned int i0 = 2u * p;
    unsigned long long xOff = bh * seq * headDim + (unsigned long long)s * headDim;
    unsigned long long cOff = (unsigned long long)s * rotDim;
    float re = x[xOff + i0], im = x[xOff + i0 + 1];
    float c = cos[cOff + i0], sn = sin[cOff + i0];
    x[xOff + i0]     = re * c - im * sn;
    x[xOff + i0 + 1] = re * sn + im * c;
}

// Inverse of split: attn[b, heads, seq, headDim] → out[token, dim] (token = f*sp + i).
__global__ void oasis_merge_heads_f32(
    float* __restrict__ out,
    const float* __restrict__ attn,
    unsigned int frames, unsigned int sp, unsigned int heads, unsigned int headDim,
    unsigned int temporal)
{
    unsigned int dim = heads * headDim;
    unsigned long long total = (unsigned long long)frames * sp * dim;
    unsigned long long gid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    unsigned int d = (unsigned int)(gid % headDim);
    unsigned long long tmp = gid / headDim;
    unsigned int h = (unsigned int)(tmp % heads);
    unsigned long long token = tmp / heads;
    unsigned int f = (unsigned int)(token / sp);
    unsigned int i = (unsigned int)(token % sp);
    unsigned int b = temporal ? i : f;
    unsigned int s = temporal ? f : i;
    unsigned int seq = temporal ? frames : sp;
    unsigned long long srcIdx = (((unsigned long long)b * heads + h) * seq + s) * headDim + d;
    unsigned long long dstIdx = token * dim + (unsigned long long)h * headDim + d;
    out[dstIdx] = attn[srcIdx];
}

// Fused Oasis adaLN: out[r,d] = LayerNorm(x[r])·(1+scale[f,d]) + shift[f,d], f = r/sp. scale/shift are sliced
// from mod[f, modStride] at scaleOff/shiftOff. One block per row; replaces LayerNorm + slice×2 + addscalar +
// affine-broadcast (5 kernels → 1). blockDim must be a power of two (tree reduction).
__global__ void oasis_adaln_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ mod,
    unsigned int dim, unsigned int sp, unsigned int totalRows,
    unsigned int modStride, unsigned int shiftOff, unsigned int scaleOff, float eps)
{
    extern __shared__ float sdata[];
    unsigned int row = blockIdx.x;
    if (row >= totalRows) return;
    const float* inRow = input + (size_t)row * dim;
    float* outRow = output + (size_t)row * dim;
    unsigned int f = row / sp;
    const float* scale = mod + (size_t)f * modStride + scaleOff;
    const float* shift = mod + (size_t)f * modStride + shiftOff;

    float partial = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x) partial += inRow[i];
    sdata[threadIdx.x] = partial; __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) { if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s]; __syncthreads(); }
    float mean = sdata[0] / (float)dim; __syncthreads();

    float vpart = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x) { float d = inRow[i] - mean; vpart += d * d; }
    sdata[threadIdx.x] = vpart; __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) { if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s]; __syncthreads(); }
    float invStd = rsqrtf(sdata[0] / (float)dim + eps);

    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x)
        outRow[i] = (inRow[i] - mean) * invStd * (1.0f + scale[i]) + shift[i];
}

// Oasis per-frame unpatchify: proj[t*sp, c*p*p] (token = f*sp + hp*gw+wp) → out[t, c, H, W], with the
// out-vector laid [py, px, ci] (channel innermost), matching the reference einsum nhwpqc->nchpwq.
// One thread per output element out[f, ci, y, x].
__global__ void oasis_unpatchify_f32(
    float* __restrict__ out,
    const float* __restrict__ proj,
    unsigned int frames, unsigned int channels, unsigned int gh, unsigned int gw, unsigned int patch)
{
    unsigned int W = gw * patch, H = gh * patch;
    unsigned long long total = (unsigned long long)frames * channels * H * W;
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned int x = (unsigned int)(i % W);
    unsigned long long r = i / W;
    unsigned int y = (unsigned int)(r % H);
    r /= H;
    unsigned int ci = (unsigned int)(r % channels);
    unsigned int f = (unsigned int)(r / channels);
    unsigned int ty = y / patch, py = y % patch, tx = x / patch, px = x % patch;
    unsigned int outVec = channels * patch * patch;
    unsigned long long token = (unsigned long long)f * gh * gw + (unsigned long long)ty * gw + tx;
    unsigned long long src = token * outVec + (unsigned long long)(py * patch + px) * channels + ci;
    out[i] = proj[src];
}

// Pixel quantize to 256 levels (DIAMOND): out = floor((clamp(v,-1,1)+1)*127.5)/127.5 - 1.
// `(int)` truncation of a non-negative value == floor. Enables a drain-free EDM step for graph capture.
__global__ void dit_pixel_quantize_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    float unused,
    unsigned int count)
{
    (void)unused;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float v = input[i];
    v = v > 1.0f ? 1.0f : (v < -1.0f ? -1.0f : v);
    int b = (int)((v + 1.0f) * 0.5f * 255.0f);
    output[i] = (float)b / 255.0f * 2.0f - 1.0f;
}

// TripoSR triplane grid-sample: for each of `count` points, bilinearly sample the three orthogonal
// feature planes (align_corners=False, zeros pad — matches GridSampler.GridSamplePlane) and write the
// concatenated [3*C] feature vector at outF[point*3C + plane*C + c]. Coords are read from coords[point*3]
// (xyz in [-R,R]) when gridRes==0, else generated in-kernel from the ij-order grid index (chunkStart+tid,
// z innermost) — the density-grid hot path, no host coord buffer. Replaces the per-point host loop.
__device__ __forceinline__ void tri_sample_plane(
    const float* __restrict__ plane, unsigned int channels, unsigned int H, unsigned int W,
    float ga, float gb, float* __restrict__ outC)  // ga -> width, gb -> height
{
    float fx = ((ga + 1.0f) * W - 1.0f) * 0.5f;
    float fy = ((gb + 1.0f) * H - 1.0f) * 0.5f;
    int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
    int x1 = x0 + 1, y1 = y0 + 1;
    float tx = fx - x0, ty = fy - y0;
    float w00 = (1.0f - tx) * (1.0f - ty), w10 = tx * (1.0f - ty);
    float w01 = (1.0f - tx) * ty,          w11 = tx * ty;
    bool x0ok = x0 >= 0 && x0 < (int)W, x1ok = x1 >= 0 && x1 < (int)W;
    bool y0ok = y0 >= 0 && y0 < (int)H, y1ok = y1 >= 0 && y1 < (int)H;
    unsigned long long plane2d = (unsigned long long)H * W;
    for (unsigned int c = 0; c < channels; c++)
    {
        const float* b = plane + c * plane2d;
        float acc = 0.0f;
        if (y0ok && x0ok) acc += w00 * b[y0 * (int)W + x0];
        if (y0ok && x1ok) acc += w10 * b[y0 * (int)W + x1];
        if (y1ok && x0ok) acc += w01 * b[y1 * (int)W + x0];
        if (y1ok && x1ok) acc += w11 * b[y1 * (int)W + x1];
        outC[c] = acc;
    }
}

__global__ void triplane_grid_sample_f32(
    float* __restrict__ outF,
    const float* __restrict__ planes,
    const float* __restrict__ coords,
    unsigned long long chunkStart,
    unsigned int count,
    unsigned int channels, unsigned int planeH, unsigned int planeW,
    float radius, unsigned int gridRes)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    float x, y, z;
    if (gridRes > 0)
    {
        unsigned long long lin = chunkStart + tid;
        unsigned int res = gridRes;
        unsigned int iz = (unsigned int)(lin % res);
        unsigned int iy = (unsigned int)((lin / res) % res);
        unsigned int ix = (unsigned int)(lin / ((unsigned long long)res * res));
        float inv = res > 1 ? 1.0f / (float)(res - 1) : 0.0f;
        float span = 2.0f * radius;
        x = -radius + ix * inv * span;
        y = -radius + iy * inv * span;
        z = -radius + iz * inv * span;
    }
    else
    {
        unsigned long long ci = (unsigned long long)tid * 3;
        x = coords[ci]; y = coords[ci + 1]; z = coords[ci + 2];
    }
    float gx = x / radius, gy = y / radius, gz = z / radius;
    unsigned int C = channels;
    unsigned long long base = (unsigned long long)tid * 3 * C;
    unsigned long long planeSz = (unsigned long long)C * planeH * planeW;
    tri_sample_plane(planes,                 C, planeH, planeW, gx, gy, outF + base);
    tri_sample_plane(planes + planeSz,       C, planeH, planeW, gx, gz, outF + base + C);
    tri_sample_plane(planes + 2 * planeSz,   C, planeH, planeW, gy, gz, outF + base + 2 * C);
}

// GEGLU with exact (erf) GELU gate: out[row,i] = proj[row,i] * gelu_erf(proj[row, inner+i]), where
// proj is [rows, 2*inner], out is [rows, inner], gelu_erf(x) = 0.5*x*(1+erf(x/sqrt2)). Fuses the split +
// erf-gelu + gate multiply into one device pass — replaces the per-block host loop whose proj.DataPointer
// read drained the compute stream every block (serializing the whole backbone).
__global__ void geglu_erf_f32(
    float* __restrict__ out,
    const float* __restrict__ proj,
    unsigned long long rows, unsigned int inner)
{
    unsigned long long total = rows * inner;
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    unsigned long long row = idx / inner;
    unsigned int i = (unsigned int)(idx % inner);
    const float* base = proj + row * 2ULL * inner;
    float h = base[i];
    float g = base[inner + i];
    out[idx] = h * (0.5f * g * (1.0f + erff(g * 0.70710678118654752440f)));
}

// Exact (erf) GELU, elementwise: out[i] = 0.5*x*(1+erf(x/sqrt2)) — PyTorch's default nn.GELU().
// Distinct from the tanh approximation gelu: the ~3e-3 pointwise gap compounds across deep
// backbones (DINOv2's MLPs are exact-erf; Depth-Anything parity needs it).
__global__ void gelu_erf_f32(
    float* __restrict__ out,
    const float* __restrict__ in,
    unsigned long long count)
{
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    float x = in[idx];
    out[idx] = 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
}

// 2D transposed convolution (gather form — one thread per output element, no atomics). Weight is
// [Cin, Cout, kH, kW] (PyTorch ConvTranspose2d). out[b,co,oy,ox] = bias[co] + Σ_{ci,ky,kx} in[b,ci,iy,ix]·W,
// where iy = (oy+pH-ky)/sH (integer, in-range). Replaces the CPU scatter-add default (was ~1.5 s for TripoSR's
// tiny 32²→64² upsample; also used by ClipSeg/YOLO/Demucs/RVC/ResembleEnhance).
__global__ void conv_transpose2d_f32(
    float* __restrict__ out, const float* __restrict__ in,
    const float* __restrict__ weight, const float* __restrict__ bias,
    int N, int Cin, int Cout, int iH, int iW, int oH, int oW,
    int kH, int kW, int sH, int sW, int pH, int pW)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)N * Cout * oH * oW;
    if (idx >= total) return;
    int ox = (int)(idx % oW);
    long r = idx / oW;
    int oy = (int)(r % oH);
    r /= oH;
    int co = (int)(r % Cout);
    int b = (int)(r / Cout);
    float acc = bias ? bias[co] : 0.0f;
    for (int ky = 0; ky < kH; ky++)
    {
        int ty = oy + pH - ky;
        if (ty % sH != 0) continue;
        int iy = ty / sH;
        if (iy < 0 || iy >= iH) continue;
        for (int kx = 0; kx < kW; kx++)
        {
            int tx = ox + pW - kx;
            if (tx % sW != 0) continue;
            int ix = tx / sW;
            if (ix < 0 || ix >= iW) continue;
            const float* inRow = in + (((long)b * Cin) * iH + iy) * iW + ix;
            const float* wRow = weight + ((long)co * kH + ky) * kW + kx;
            long inStride = (long)iH * iW, wStride = (long)Cout * kH * kW;
            for (int ci = 0; ci < Cin; ci++)
                acc += inRow[ci * inStride] * wRow[ci * wStride];
        }
    }
    out[idx] = acc;
}

// Two-input concat along an arbitrary dim, in ONE launch (one thread per output element). Replaces the
// host-side per-outer-slice cuMemcpyDtoDAsync loop in CudaBackend.Concat, which issued `outer` × 2 async
// memcpys — catastrophic for last-dim / per-token concats (the Hunyuan3D single-block cat(attn,mlp) at
// outer=seqLen=4442 issued ~8900 memcpys/concat → 8.4 ms/call, ~280k graph nodes/forward). Logical layout is
// [outer, aDim|bDim, inner] with the concat on the middle axis: dim=1 concat → inner = product of trailing
// dims; last-dim concat → inner = 1, outer = product of leading dims. Covers every 2-input concat.
__global__ void dit_concat2_f32(
    float* __restrict__ out,
    const float* __restrict__ a,
    const float* __restrict__ b,
    unsigned int outer, unsigned int aDim, unsigned int bDim, unsigned int inner)
{
    unsigned int outDim = aDim + bDim;
    unsigned long long total = (unsigned long long)outer * outDim * inner;
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    unsigned int innerI = (unsigned int)(idx % inner);
    unsigned long long tmp = idx / inner;
    unsigned int dimIdx = (unsigned int)(tmp % outDim);
    unsigned long long o = tmp / outDim;
    if (dimIdx < aDim)
        out[idx] = a[((o * aDim + dimIdx) * (unsigned long long)inner) + innerI];
    else
        out[idx] = b[((o * bDim + (dimIdx - aDim)) * (unsigned long long)inner) + innerI];
}

// Fused adaLN modulation: out[r,d] = (1 + scale[b,d])·LayerNormNoAffine(in[r]) + shift[b,d], b = r/seqLen.
// One block per row (tree reduction over dim). Replaces LayerNormNoAffine + AddScalar(+1) + AffineBroadcast (3 → 1)
// — the DiT NormModulate, ~96×/forward. scale/shift are [B,dim] (per-channel, broadcast over seq).
__global__ void dit_layernorm_modulate_f32(
    float* __restrict__ output, const float* __restrict__ input,
    const float* __restrict__ scale, const float* __restrict__ shift,
    unsigned int dim, unsigned int seqLen, unsigned int totalRows, float eps)
{
    extern __shared__ float sdata[];
    unsigned int row = blockIdx.x;
    if (row >= totalRows) return;
    const float* inRow = input + (size_t)row * dim;
    float* outRow = output + (size_t)row * dim;
    unsigned int b = row / seqLen;
    const float* sc = scale + (size_t)b * dim;
    const float* sh = shift + (size_t)b * dim;

    float partial = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x) partial += inRow[i];
    sdata[threadIdx.x] = partial; __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) { if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s]; __syncthreads(); }
    float mean = sdata[0] / (float)dim; __syncthreads();

    float vpart = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x) { float d = inRow[i] - mean; vpart += d * d; }
    sdata[threadIdx.x] = vpart; __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) { if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s]; __syncthreads(); }
    float invStd = rsqrtf(sdata[0] / (float)dim + eps);

    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x)
        outRow[i] = (inRow[i] - mean) * invStd * (1.0f + sc[i]) + sh[i];
}

// Fused QKV split + per-head QK-RMSNorm: from a fused qkv[token, 3·W] (W = heads·headDim), writes q,k,v each
// [token, W] laid [token, head, d], applying RMSNorm(·)·weight over each head's headDim slice to q and k (v copied).
// Replaces SliceLastDim×3 + RmsNorm×2 (5 → 1) per attention stream. One block per (token, head); blockDim = headDim
// (power of two — tree reduction). Activation follows the qkv dtype; the norm weights stay F32.
__global__ void dit_qkv_split_norm_f32(
    float* __restrict__ q, float* __restrict__ k, float* __restrict__ v,
    const float* __restrict__ qkv, const float* __restrict__ qW, const float* __restrict__ kW,
    unsigned int tokens, unsigned int heads, unsigned int headDim, float eps)
{
    extern __shared__ float sred[];
    unsigned long long g = (unsigned long long)blockIdx.x;   // group = token*heads + head
    if (g >= (unsigned long long)tokens * heads) return;
    unsigned int h = (unsigned int)(g % heads);
    unsigned long long token = g / heads;
    unsigned int W = heads * headDim, d = threadIdx.x;
    const float* base = qkv + token * 3ULL * W;
    unsigned long long outOff = token * W + (unsigned long long)h * headDim;

    float qv = base[(unsigned long long)h * headDim + d];
    sred[d] = qv * qv; __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) { if (d < s) sred[d] += sred[d + s]; __syncthreads(); }
    float qInv = rsqrtf(sred[0] / (float)headDim + eps); __syncthreads();

    float kv = base[W + (unsigned long long)h * headDim + d];
    sred[d] = kv * kv; __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) { if (d < s) sred[d] += sred[d + s]; __syncthreads(); }
    float kInv = rsqrtf(sred[0] / (float)headDim + eps);

    q[outOff + d] = qv * qInv * qW[d];
    k[outOff + d] = kv * kInv * kW[d];
    v[outOff + d] = base[2u * W + (unsigned long long)h * headDim + d];
}

// FourierEmbedder (num_freqs bands, freqs 2^i, include_input=true, include_pi=false): for each point i and coord
// c∈{0,1,2}: out[i, c] = x; out[i, 3 + c·bands + band] = sin(x·2^band); out[i, 3 + 3·bands + c·bands + band] =
// cos(x·2^band). One thread per (point, coord). Replaces the host trig loop in the ShapeVAE geo-decoder query
// (kept feat on device → no per-chunk H2D of the Fourier features). dim = 3·(2·bands + 1).
__global__ void fourier_embed_f32(
    float* __restrict__ dst, const float* __restrict__ coords,
    unsigned int count, unsigned int bands, unsigned int dim)
{
    unsigned long long gid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (unsigned long long)count * 3u) return;
    unsigned int c = (unsigned int)(gid % 3u);
    unsigned long long i = gid / 3u;
    float x = coords[i * 3ULL + c];
    unsigned long long o = i * dim;
    unsigned int sinBase = 3u, cosBase = 3u + 3u * bands;
    dst[o + c] = x;
    for (unsigned int band = 0; band < bands; band++)
    {
        float a = x * (float)(1u << band);
        dst[o + sinBase + c * bands + band] = sinf(a);
        dst[o + cosBase + c * bands + band] = cosf(a);
    }
}

// Dense 3D convolution (gather form — one thread per output element, no atomics). Weight is [Cout,Cin,kD,kH,kW]
// (PyTorch Conv3d, groups=1). out[b,co,od,oh,ow] = bias[co] + Σ_{ci,kd,kh,kw} in[b,ci,id,ih,iw]·W, id = od·sD−pD+kd.
// For the TRELLIS sparse-structure VAE decoder (16³ latent → 64³ occupancy).
__global__ void conv3d_f32(
    float* __restrict__ out, const float* __restrict__ in,
    const float* __restrict__ weight, const float* __restrict__ bias,
    int N, int Cin, int Cout, int iD, int iH, int iW, int oD, int oH, int oW,
    int kD, int kH, int kW, int sD, int sH, int sW, int pD, int pH, int pW)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)N * Cout * oD * oH * oW;
    if (idx >= total) return;
    int ow = (int)(idx % oW); long r = idx / oW;
    int oh = (int)(r % oH); r /= oH;
    int od = (int)(r % oD); r /= oD;
    int co = (int)(r % Cout); int b = (int)(r / Cout);
    float acc = bias ? bias[co] : 0.0f;
    int id0 = od * sD - pD, ih0 = oh * sH - pH, iw0 = ow * sW - pW;
    for (int ci = 0; ci < Cin; ci++)
    {
        const float* src = in + (((long)b * Cin + ci) * iD) * iH * iW;
        long wOff = (((long)co * Cin + ci) * kD) * kH * kW;
        for (int kd = 0; kd < kD; kd++)
        {
            int id = id0 + kd; if (id < 0 || id >= iD) continue;
            for (int kh = 0; kh < kH; kh++)
            {
                int ih = ih0 + kh; if (ih < 0 || ih >= iH) continue;
                for (int kw = 0; kw < kW; kw++)
                {
                    int iw = iw0 + kw; if (iw < 0 || iw >= iW) continue;
                    acc += src[(id * iH + ih) * iW + iw] * weight[wOff + (kd * kH + kh) * kW + kw];
                }
            }
        }
    }
    out[idx] = acc;
}

// Sparse submanifold-conv scatter: write active-voxel features onto a dense (pre-zeroed) grid on-device. Avoids the
// host scatter loop + the multi-GB grid H2D that dominated the TRELLIS SLAT flow. grid is [1,C,R,R,R]; coords [N,4]
// (b,x,y,z). One thread per (voxel, channel). cell = x·R² + y·R + z.
__global__ void sparse_scatter_to_grid_f32(
    float* __restrict__ grid, const float* __restrict__ feats, const int* __restrict__ coords,
    int n, int c, int r)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)n * c;
    if (idx >= total) return;
    int ch = (int)(idx % c);
    long v = idx / c;
    long r3 = (long)r * r * r;
    long cell = (long)coords[v * 4 + 1] * r * r + (long)coords[v * 4 + 2] * r + coords[v * 4 + 3];
    grid[(long)ch * r3 + cell] = feats[v * (long)c + ch];
}

// Inverse of the scatter: gather the conv output at the active voxels back into a feature matrix [N,C].
__global__ void sparse_gather_from_grid_f32(
    float* __restrict__ feats, const float* __restrict__ grid, const int* __restrict__ coords,
    int n, int c, int r)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)n * c;
    if (idx >= total) return;
    int ch = (int)(idx % c);
    long v = idx / c;
    long r3 = (long)r * r * r;
    long cell = (long)coords[v * 4 + 1] * r * r + (long)coords[v * 4 + 2] * r + coords[v * 4 + 3];
    feats[v * (long)c + ch] = grid[(long)ch * r3 + cell];
}

// Row gather: out[j] = in[indices[j]] (one thread per output element). For the sparse-conv rulebook (gather the
// active neighbours for one kernel offset).
__global__ void row_gather_f32(
    float* __restrict__ out, const float* __restrict__ in, const int* __restrict__ indices, int m, int c)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)m * c) return;
    int ch = (int)(idx % c);
    long j = idx / c;
    out[idx] = in[(long)indices[j] * c + ch];
}

// Row scatter-add: out[indices[j]] += in[j] (indices unique within a call → no atomics needed). Accumulates a
// kernel-offset's GEMM contribution into the sparse-conv output.
__global__ void row_scatter_add_f32(
    float* __restrict__ out, const float* __restrict__ in, const int* __restrict__ indices, int m, int c)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)m * c) return;
    int ch = (int)(idx % c);
    long j = idx / c;
    out[(long)indices[j] * c + ch] += in[idx];
}

} // extern "C"
