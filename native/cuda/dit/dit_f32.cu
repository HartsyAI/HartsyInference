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

} // extern "C"
