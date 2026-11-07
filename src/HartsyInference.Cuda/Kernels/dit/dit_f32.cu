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

// ── Row-indexed gated residual (gather fused in) ───────────────────────────
// out[r,d] = residual[r,d] + gateTable[rowIndex[r], d] * value[r,d]
__global__ void dit_gated_residual_rowindexed_f32(
    float* __restrict__ output,
    const float* __restrict__ residual,
    const float* __restrict__ value,
    const float* __restrict__ gateTable,
    const int* __restrict__ rowIndex,
    unsigned int dim,
    unsigned long long total)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;

    unsigned int d = (unsigned int)(i % dim);
    unsigned long long row = i / dim;
    size_t tIdx = (size_t)rowIndex[row] * dim + d;

    output[i] = residual[i] + gateTable[tIdx] * value[i];
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

// ── Generic affine and mask-broadcast affine mixes ─────────────────────────
// These are the shared image-pipeline glue primitives: scheduler AddNoise is a plain affine mix, while masked
// inpaint replaces the preserved region with the source's affine noise trajectory. All sizes are 64-bit and all
// grid dimensions are bounded by the launcher; grid-stride loops cover the remainder.
__global__ void dit_affine_mix_f32(
    float* __restrict__ output,
    const float* __restrict__ x,
    const float* __restrict__ y,
    float xScale,
    float yScale,
    unsigned long long count)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (; i < count; i += stride)
        output[i] = xScale * x[i] + yScale * y[i];
}

__device__ __forceinline__ float dit_affine_replacement(
    const float* __restrict__ source,
    const float* __restrict__ noise,
    unsigned long long i,
    float sourceScale,
    float noiseScale)
{
    float replacement = sourceScale * source[i];
    if (noise != nullptr)
        replacement += noiseScale * noise[i];
    return replacement;
}

// NCHW target/source/noise [B,C,H,W], mask [B,1,H,W]. The launch maps x to spatial tiles and y to
// batches, so each thread walks channels with coalesced accesses and reuses one mask load without integer div/mod.
__global__ void dit_masked_affine_mix_dense_f32(
    float* __restrict__ target,
    const float* __restrict__ source,
    const float* __restrict__ noise,
    const float* __restrict__ mask,
    float sourceScale,
    float noiseScale,
    unsigned long long batch,
    unsigned long long channels,
    unsigned long long spatial)
{
    unsigned long long spatialStart = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long spatialStride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long b = blockIdx.y; b < batch; b += gridDim.y)
    {
        unsigned long long maskBase = b * spatial;
        unsigned long long targetBatchBase = b * channels * spatial;
        for (unsigned long long p = spatialStart; p < spatial; p += spatialStride)
        {
            float maskValue = mask[maskBase + p];
            float inverseMask = 1.0f - maskValue;
            for (unsigned long long c = 0; c < channels; c++)
            {
                unsigned long long i = targetBatchBase + c * spatial + p;
                float replacement = dit_affine_replacement(source, noise, i, sourceScale, noiseScale);
                target[i] = target[i] * maskValue + replacement * inverseMask;
            }
        }
    }
}

__device__ __forceinline__ void dit_masked_affine_mix_packed(
    float* __restrict__ target,
    const float* __restrict__ source,
    const float* __restrict__ noise,
    const float* __restrict__ mask,
    float sourceScale,
    float noiseScale,
    unsigned long long tokens,
    unsigned long long featureDimension,
    unsigned long long patchArea,
    bool channelInner)
{
    unsigned long long channels = featureDimension / patchArea;
    for (unsigned long long token = blockIdx.x; token < tokens; token += gridDim.x)
    {
        unsigned long long featureBase = token * featureDimension;
        unsigned long long maskBase = token * patchArea;
        for (unsigned long long feature = threadIdx.x; feature < featureDimension; feature += blockDim.x)
        {
            unsigned long long patchIndex = channelInner ? feature / channels : feature % patchArea;
            unsigned long long i = featureBase + feature;
            float maskValue = mask[maskBase + patchIndex];
            float replacement = dit_affine_replacement(source, noise, i, sourceScale, noiseScale);
            target[i] = target[i] * maskValue + replacement * (1.0f - maskValue);
        }
    }
}

// Packed channel-outer feature order f=c*P+p (Flux/Chroma/Flux.2).
__global__ void dit_masked_affine_mix_packed_outer_f32(
    float* __restrict__ target,
    const float* __restrict__ source,
    const float* __restrict__ noise,
    const float* __restrict__ mask,
    float sourceScale,
    float noiseScale,
    unsigned long long tokens,
    unsigned long long featureDimension,
    unsigned long long patchArea)
{
    dit_masked_affine_mix_packed(
        target, source, noise, mask, sourceScale, noiseScale, tokens, featureDimension, patchArea, false);
}

// Packed channel-inner feature order f=p*C+c (Ideogram 4).
__global__ void dit_masked_affine_mix_packed_inner_f32(
    float* __restrict__ target,
    const float* __restrict__ source,
    const float* __restrict__ noise,
    const float* __restrict__ mask,
    float sourceScale,
    float noiseScale,
    unsigned long long tokens,
    unsigned long long featureDimension,
    unsigned long long patchArea)
{
    dit_masked_affine_mix_packed(
        target, source, noise, mask, sourceScale, noiseScale, tokens, featureDimension, patchArea, true);
}

// ── Global-renormalized CFG + Euler step ───────────────────────────────────
// Stage 1 computes block partials for ||cond||² and ||uncond + guidance*(cond-uncond)||².
// Double accumulation matches the host reference's global norm without sacrificing the parallel update.
__global__ void dit_cfg_renorm_reduce_f32(
    const float* __restrict__ cond,
    const float* __restrict__ uncond,
    double* __restrict__ condPartials,
    double* __restrict__ guidedPartials,
    float guidance,
    unsigned long long count)
{
    extern __shared__ double sums[];
    double* condSums = sums;
    double* guidedSums = sums + blockDim.x;

    double condSq = 0.0;
    double guidedSq = 0.0;
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (; i < count; i += stride)
    {
        float c = cond[i];
        float u = uncond[i];
        float guided = u + guidance * (c - u);
        condSq += (double)c * (double)c;
        guidedSq += (double)guided * (double)guided;
    }

    condSums[threadIdx.x] = condSq;
    guidedSums[threadIdx.x] = guidedSq;
    __syncthreads();
    for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (threadIdx.x < offset)
        {
            condSums[threadIdx.x] += condSums[threadIdx.x + offset];
            guidedSums[threadIdx.x] += guidedSums[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        condPartials[blockIdx.x] = condSums[0];
        guidedPartials[blockIdx.x] = guidedSums[0];
    }
}

// Stage 2 reduces the small partial arrays and writes one device scalar. Branch clamps intentionally preserve
// NaN, matching Math.Clamp: fmin/fmax would silently replace NaN with a clamp endpoint.
__global__ void dit_cfg_renorm_finalize_f32(
    float* __restrict__ scale,
    const double* __restrict__ condPartials,
    const double* __restrict__ guidedPartials,
    unsigned int partialCount,
    float renormMin)
{
    extern __shared__ double sums[];
    double* condSums = sums;
    double* guidedSums = sums + blockDim.x;
    double condSq = 0.0;
    double guidedSq = 0.0;
    for (unsigned int i = threadIdx.x; i < partialCount; i += blockDim.x)
    {
        condSq += condPartials[i];
        guidedSq += guidedPartials[i];
    }
    condSums[threadIdx.x] = condSq;
    guidedSums[threadIdx.x] = guidedSq;
    __syncthreads();
    for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (threadIdx.x < offset)
        {
            condSums[threadIdx.x] += condSums[threadIdx.x + offset];
            guidedSums[threadIdx.x] += guidedSums[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        double ratio = sqrt(condSums[0]) / (sqrt(guidedSums[0]) + 1.0e-8);
        if (ratio < (double)renormMin) ratio = (double)renormMin;
        else if (ratio > 1.0) ratio = 1.0;
        scale[0] = (float)ratio;
    }
}

// Stage 3 recomputes the cheap guided value and folds scale + Euler into the latent write. Inputs are read-only.
__global__ void dit_cfg_renorm_euler_f32(
    float* __restrict__ z,
    const float* __restrict__ cond,
    const float* __restrict__ uncond,
    const float* __restrict__ scale,
    float guidance,
    float delta,
    unsigned long long count)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    float s = scale[0];
    for (; i < count; i += stride)
    {
        float guided = uncond[i] + guidance * (cond[i] - uncond[i]);
        guided *= s;
        z[i] += guided * delta;
    }
}

// ── Last-dimension-normalized CFG + Euler step ───────────────────────────
// One cooperative block handles each logical row (all dimensions except the last are flattened).
// The grid-stride row loop keeps launch geometry bounded while still covering arbitrarily many rows.
// Double accumulation mirrors CfgHelper.ApplyCfgNormalized's host reference; guided values and the
// broadcast ratio remain float, matching the tensor's F32 storage and the subsequent Euler update.
__global__ void dit_cfg_normalized_euler_f32(
    float* __restrict__ z,
    const float* __restrict__ cond,
    const float* __restrict__ uncond,
    float guidance,
    float delta,
    float eps,
    unsigned long long rows,
    unsigned long long lastDim)
{
    extern __shared__ double sums[];
    double* condSums = sums;
    double* guidedSums = sums + blockDim.x;
    __shared__ float rowRatio;

    for (unsigned long long row = blockIdx.x; row < rows; row += gridDim.x)
    {
        unsigned long long base = row * lastDim;
        double condSq = 0.0;
        double guidedSq = 0.0;
        for (unsigned long long d = threadIdx.x; d < lastDim; d += blockDim.x)
        {
            unsigned long long i = base + d;
            float c = cond[i];
            float u = uncond[i];
            float guided = u + guidance * (c - u);
            condSq += (double)c * (double)c;
            guidedSq += (double)guided * (double)guided;
        }

        condSums[threadIdx.x] = condSq;
        guidedSums[threadIdx.x] = guidedSq;
        __syncthreads();
        for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
        {
            if (threadIdx.x < offset)
            {
                condSums[threadIdx.x] += condSums[threadIdx.x + offset];
                guidedSums[threadIdx.x] += guidedSums[threadIdx.x + offset];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0)
        {
            // Zero denominator (eps=0, all-zero guided row) contributes nothing — ratio=0 avoids 0/0 -> NaN.
            double denom = sqrt(guidedSums[0]) + (double)eps;
            rowRatio = denom > 0.0 ? (float)(sqrt(condSums[0]) / denom) : 0.0f;
        }
        __syncthreads();

        float ratio = rowRatio;
        for (unsigned long long d = threadIdx.x; d < lastDim; d += blockDim.x)
        {
            unsigned long long i = base + d;
            float guided = uncond[i] + guidance * (cond[i] - uncond[i]);
            z[i] += delta * (guided * ratio);
        }
        __syncthreads();
    }
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

} // extern "C" — templates cannot have C linkage

// Copy one arbitrary-axis split chunk from logical [outer,inputDim,inner] storage. Each logical CUDA block
// owns a contiguous part of one outer slice, so only one 64-bit division is paid per block (not per element),
// while all payload loads/stores remain coalesced. The capped physical grid strides over logical blocks, which
// keeps valid 64-bit tensor geometry independent of CUDA's gridDim.x launch limit. Integer words preserve F32,
// F16, and BF16 payload bits exactly, including NaN payloads and signed zero.
template <typename Word>
__device__ __forceinline__ void dit_split_slice_words(
    Word* __restrict__ output,
    const Word* __restrict__ input,
    unsigned long long inputStride,
    unsigned long long outputStride,
    unsigned long long sourceOffset,
    unsigned long long blocksPerOuter,
    unsigned long long totalBlocks)
{
    unsigned long long logicalBlock = (unsigned long long)blockIdx.x;
    unsigned long long blockStride = (unsigned long long)gridDim.x;
    for (; logicalBlock < totalBlocks; logicalBlock += blockStride)
    {
        unsigned long long outer = logicalBlock / blocksPerOuter;
        unsigned long long blockWithinOuter = logicalBlock - outer * blocksPerOuter;
        unsigned long long withinOuter = blockWithinOuter * blockDim.x + threadIdx.x;
        if (withinOuter < outputStride)
            output[outer * outputStride + withinOuter] =
                input[outer * inputStride + sourceOffset + withinOuter];
    }
}

extern "C" __global__ void dit_split_slice_u32(
    unsigned int* output, const unsigned int* input,
    unsigned long long inputStride, unsigned long long outputStride,
    unsigned long long sourceOffset, unsigned long long blocksPerOuter,
    unsigned long long totalBlocks)
{
    dit_split_slice_words(
        output, input, inputStride, outputStride, sourceOffset, blocksPerOuter, totalBlocks);
}

extern "C" __global__ void dit_split_slice_u16(
    unsigned short* output, const unsigned short* input,
    unsigned long long inputStride, unsigned long long outputStride,
    unsigned long long sourceOffset, unsigned long long blocksPerOuter,
    unsigned long long totalBlocks)
{
    dit_split_slice_words(
        output, input, inputStride, outputStride, sourceOffset, blocksPerOuter, totalBlocks);
}

extern "C" {

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

// Fused QKV split + per-head QK-RMSNorm: from a fused qkv[token, 3·W] (W = heads·headDim), writes q,k,v each
// [token, W] laid [token, head, d], applying RMSNorm(·)·weight over each head's headDim slice to q and k (v copied).
// Replaces SliceLastDim×3 + RmsNorm×2 (5 → 1) per attention stream. One power-of-two block per
// (token, head); threads stride across headDim, so non-power-of-two and large head dimensions are exact.
// Activation follows the qkv dtype; the norm weights stay F32.
__global__ void dit_qkv_split_norm_f32(
    float* __restrict__ q, float* __restrict__ k, float* __restrict__ v,
    const float* __restrict__ qkv, const float* __restrict__ qW, const float* __restrict__ kW,
    unsigned int tokens, unsigned int heads, unsigned int headDim, float eps)
{
    extern __shared__ float sred[];
    float* qred = sred;
    float* kred = sred + blockDim.x;
    unsigned long long g = (unsigned long long)blockIdx.x;   // group = token*heads + head
    if (g >= (unsigned long long)tokens * heads) return;
    unsigned int h = (unsigned int)(g % heads);
    unsigned long long token = g / heads;
    unsigned int W = heads * headDim;
    unsigned int d = threadIdx.x;
    const float* base = qkv + token * 3ULL * W;
    unsigned long long outOff = token * W + (unsigned long long)h * headDim;

    float qv = 0.0f;
    float kv = 0.0f;
    if (d < headDim)
    {
        qv = base[(unsigned long long)h * headDim + d];
        kv = base[W + (unsigned long long)h * headDim + d];
    }
    float qsum = qv * qv;
    float ksum = kv * kv;
    for (unsigned int i = d + blockDim.x; i < headDim; i += blockDim.x)
    {
        float qi = base[(unsigned long long)h * headDim + i];
        float ki = base[W + (unsigned long long)h * headDim + i];
        qsum += qi * qi;
        ksum += ki * ki;
    }
    qred[d] = qsum;
    kred[d] = ksum;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        if (d < stride)
        {
            qred[d] += qred[d + stride];
            kred[d] += kred[d + stride];
        }
        __syncthreads();
    }
    if (d == 0)
    {
        float invDim = 1.0f / (float)headDim;
        qred[0] = rsqrtf(qred[0] * invDim + eps);
        kred[0] = rsqrtf(kred[0] * invDim + eps);
    }
    __syncthreads();

    if (d < headDim)
    {
        q[outOff + d] = qv * qred[0] * qW[d];
        k[outOff + d] = kv * kred[0] * kW[d];
        v[outOff + d] = base[2u * W + (unsigned long long)h * headDim + d];
    }
    for (unsigned int i = d + blockDim.x; i < headDim; i += blockDim.x)
    {
        q[outOff + i] = base[(unsigned long long)h * headDim + i] * qred[0] * qW[i];
        k[outOff + i] = base[W + (unsigned long long)h * headDim + i] * kred[0] * kW[i];
        v[outOff + i] = base[2u * W + (unsigned long long)h * headDim + i];
    }
}

// Head-major twin of dit_qkv_split_norm_f32: identical math, writes q/k/v as [batch, heads, seq, headDim]
// instead of [token, head, d], so the attention caller feeds SDPA with no Permute0213. Same grid mapping
// (one block per (token, head)) — only the output address changes.
__global__ void dit_qkv_split_norm_head_major_f32(
    float* __restrict__ q, float* __restrict__ k, float* __restrict__ v,
    const float* __restrict__ qkv, const float* __restrict__ qW, const float* __restrict__ kW,
    unsigned int tokens, unsigned int heads, unsigned int headDim, unsigned int seq, float eps)
{
    extern __shared__ float sred[];
    float* qred = sred;
    float* kred = sred + blockDim.x;
    unsigned long long g = (unsigned long long)blockIdx.x;   // group = token*heads + head
    if (g >= (unsigned long long)tokens * heads) return;
    unsigned int h = (unsigned int)(g % heads);
    unsigned long long token = g / heads;
    unsigned int W = heads * headDim;
    unsigned int d = threadIdx.x;
    const float* base = qkv + token * 3ULL * W;
    unsigned long long b = token / seq, s = token % seq;
    unsigned long long outOff = ((b * heads + h) * seq + s) * headDim;

    float qv = 0.0f;
    float kv = 0.0f;
    if (d < headDim)
    {
        qv = base[(unsigned long long)h * headDim + d];
        kv = base[W + (unsigned long long)h * headDim + d];
    }
    float qsum = qv * qv;
    float ksum = kv * kv;
    for (unsigned int i = d + blockDim.x; i < headDim; i += blockDim.x)
    {
        float qi = base[(unsigned long long)h * headDim + i];
        float ki = base[W + (unsigned long long)h * headDim + i];
        qsum += qi * qi;
        ksum += ki * ki;
    }
    qred[d] = qsum;
    kred[d] = ksum;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        if (d < stride)
        {
            qred[d] += qred[d + stride];
            kred[d] += kred[d + stride];
        }
        __syncthreads();
    }
    if (d == 0)
    {
        float invDim = 1.0f / (float)headDim;
        qred[0] = rsqrtf(qred[0] * invDim + eps);
        kred[0] = rsqrtf(kred[0] * invDim + eps);
    }
    __syncthreads();

    if (d < headDim)
    {
        q[outOff + d] = qv * qred[0] * qW[d];
        k[outOff + d] = kv * kred[0] * kW[d];
        v[outOff + d] = base[2u * W + (unsigned long long)h * headDim + d];
    }
    for (unsigned int i = d + blockDim.x; i < headDim; i += blockDim.x)
    {
        q[outOff + i] = base[(unsigned long long)h * headDim + i] * qred[0] * qW[i];
        k[outOff + i] = base[W + (unsigned long long)h * headDim + i] * kred[0] * kW[i];
        v[outOff + i] = base[2u * W + (unsigned long long)h * headDim + i];
    }
}

// FourierEmbedder (num_freqs bands, freqs 2^i, include_input=true, include_pi=false): for each point i and coord
// c∈{0,1,2}: out[i, c] = x; out[i, 3 + c·bands + band] = sin(x·2^band); out[i, 3 + 3·bands + c·bands + band] =
// cos(x·2^band). One thread per (point, coord). Replaces the host trig loop in the ShapeVAE geo-decoder query
// (kept feat on device → no per-chunk H2D of the Fourier features). dim = 3·(2·bands + 1).

// Subset-emitting twin of dit_qkv_split_norm_head_major_f32. SEPARATE kernel on purpose: the fused call above must stay
// byte-for-byte the shipped kernel, and folding slot guards into it changed its codegen enough to move
// real generation output. packStride is how many W-wide segments the packed source holds; qSlot/kSlot/
// vSlot say which segment each output reads, negative meaning "not produced" (that pointer may be null).
// MiniMaxH3Transformer's chunked attention uses it to project k+v in pass 1 and q in pass 2, so a
// full-sequence q never stays resident across the pass boundary.
__global__ void dit_qkv_split_norm_head_major_subset_f32(
    float* __restrict__ q, float* __restrict__ k, float* __restrict__ v,
    const float* __restrict__ qkv, const float* __restrict__ qW, const float* __restrict__ kW,
    unsigned int tokens, unsigned int heads, unsigned int headDim, unsigned int seq, float eps,
    unsigned int packStride, int qSlot, int kSlot, int vSlot)
{
    extern __shared__ float sred[];
    float* qred = sred;
    float* kred = sred + blockDim.x;
    unsigned long long g = (unsigned long long)blockIdx.x;   // group = token*heads + head
    if (g >= (unsigned long long)tokens * heads) return;
    unsigned int h = (unsigned int)(g % heads);
    unsigned long long token = g / heads;
    unsigned int W = heads * headDim;
    unsigned int d = threadIdx.x;
    const float* base = qkv + token * (unsigned long long)packStride * W;
    unsigned long long b = token / seq, s = token % seq;
    unsigned long long outOff = ((b * heads + h) * seq + s) * headDim;
    unsigned long long qBase = (unsigned long long)(qSlot < 0 ? 0 : qSlot) * W + (unsigned long long)h * headDim;
    unsigned long long kBase = (unsigned long long)(kSlot < 0 ? 0 : kSlot) * W + (unsigned long long)h * headDim;
    unsigned long long vBase = (unsigned long long)(vSlot < 0 ? 0 : vSlot) * W + (unsigned long long)h * headDim;

    float qv = 0.0f;
    float kv = 0.0f;
    if (d < headDim)
    {
        if (qSlot >= 0) qv = base[qBase + d];
        if (kSlot >= 0) kv = base[kBase + d];
    }
    float qsum = qv * qv;
    float ksum = kv * kv;
    for (unsigned int i = d + blockDim.x; i < headDim; i += blockDim.x)
    {
        float qi = qSlot >= 0 ? base[qBase + i] : 0.0f;
        float ki = kSlot >= 0 ? base[kBase + i] : 0.0f;
        qsum += qi * qi;
        ksum += ki * ki;
    }
    qred[d] = qsum;
    kred[d] = ksum;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        if (d < stride)
        {
            qred[d] += qred[d + stride];
            kred[d] += kred[d + stride];
        }
        __syncthreads();
    }
    if (d == 0)
    {
        float invDim = 1.0f / (float)headDim;
        qred[0] = rsqrtf(qred[0] * invDim + eps);
        kred[0] = rsqrtf(kred[0] * invDim + eps);
    }
    __syncthreads();

    if (d < headDim)
    {
        if (qSlot >= 0) q[outOff + d] = qv * qred[0] * qW[d];
        if (kSlot >= 0) k[outOff + d] = kv * kred[0] * kW[d];
        if (vSlot >= 0) v[outOff + d] = base[vBase + d];
    }
    for (unsigned int i = d + blockDim.x; i < headDim; i += blockDim.x)
    {
        if (qSlot >= 0) q[outOff + i] = base[qBase + i] * qred[0] * qW[i];
        if (kSlot >= 0) k[outOff + i] = base[kBase + i] * kred[0] * kW[i];
        if (vSlot >= 0) v[outOff + i] = base[vBase + i];
    }
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

// Row scatter-add: out[indices[j]] += in[j] without atomics, so indices must be unique within a call (a repeated row
// races and loses updates). Accumulates a sparse-conv kernel offset's GEMM output and SeedVR2's window video rows.
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

// ── LTX-2 fused QK-norm + split-RoPE + head-major emit (F32 twin) ───────────
// F32 counterpart of ltx2_qk_norm_rope_headmajor_f16 in dit_f16.cu — see that file for the layout
// contract. Exists so HARTSY_DIT_F16=0 keeps the same op graph (and stays on-device) instead of
// falling back to the three-pass sequence.
extern "C" __global__ void ltx2_qk_norm_rope_headmajor_f32(
    float* __restrict__ out,
    const float* __restrict__ in,
    const float* __restrict__ normW,
    const float* __restrict__ cosT,
    const float* __restrict__ sinT,
    unsigned int seq, unsigned int heads, unsigned int headDim, float eps)
{
    // See the f16 twin in dit_f16.cu for the profiled note on why this stays shared-memory-staged.
    extern __shared__ float smem32[];
    float* sred = smem32;
    float* srow = smem32 + blockDim.x;
    unsigned int W = heads * headDim;
    unsigned int s = blockIdx.x;
    if (s >= seq) return;
    const float* inRow = in + (size_t)s * W;

    float partial = 0.0f;
    for (unsigned int i = threadIdx.x; i < W; i += blockDim.x)
    {
        float v = inRow[i];
        srow[i] = v;
        partial += v * v;
    }
    sred[threadIdx.x] = partial;
    __syncthreads();
    for (unsigned int st = blockDim.x >> 1; st > 0; st >>= 1)
    {
        if (threadIdx.x < st) sred[threadIdx.x] += sred[threadIdx.x + st];
        __syncthreads();
    }
    float inv = rsqrtf(sred[0] / (float)W + eps);
    for (unsigned int i = threadIdx.x; i < W; i += blockDim.x)
        srow[i] = srow[i] * inv * (normW ? normW[i] : 1.0f);
    __syncthreads();

    unsigned int r = headDim >> 1;
    unsigned int halfW = W >> 1;
    for (unsigned int i = threadIdx.x; i < W; i += blockDim.x)
    {
        unsigned int h = i / headDim;
        unsigned int d = i - h * headDim;
        float val;
        if (cosT != 0)
        {
            unsigned int lane = h * r + (d < r ? d : d - r);
            float c = cosT[(size_t)s * halfW + lane];
            float sn = sinT[(size_t)s * halfW + lane];
            if (d < r) { float a = srow[i], b = srow[i + r]; val = a * c - b * sn; }
            else       { float a = srow[i - r], b = srow[i]; val = b * c + a * sn; }
        }
        else val = srow[i];
        out[((size_t)h * seq + s) * headDim + d] = val;
    }
}

// ── LTX-2 fused QK-norm + split-RoPE, TOKEN-MAJOR emit (F32 twin) ──────────
// See the f16 twin in dit_f16.cu: identical math, contiguous store instead of the scattered head-major one.
extern "C" __global__ void ltx2_qk_norm_rope_tokenmajor_f32(
    float* __restrict__ out,
    const float* __restrict__ in,
    const float* __restrict__ normW,
    const float* __restrict__ cosT,
    const float* __restrict__ sinT,
    unsigned int seq, unsigned int heads, unsigned int headDim, float eps)
{
    extern __shared__ float smemTm32[];
    float* sred = smemTm32;
    float* srow = smemTm32 + blockDim.x;
    unsigned int W = heads * headDim;
    unsigned int s = blockIdx.x;
    if (s >= seq) return;
    const float* inRow = in + (size_t)s * W;
    float* outRow = out + (size_t)s * W;

    float partial = 0.0f;
    for (unsigned int i = threadIdx.x; i < W; i += blockDim.x)
    {
        float v = inRow[i];
        srow[i] = v;
        partial += v * v;
    }
    sred[threadIdx.x] = partial;
    __syncthreads();
    for (unsigned int st = blockDim.x >> 1; st > 0; st >>= 1)
    {
        if (threadIdx.x < st) sred[threadIdx.x] += sred[threadIdx.x + st];
        __syncthreads();
    }
    float inv = rsqrtf(sred[0] / (float)W + eps);
    for (unsigned int i = threadIdx.x; i < W; i += blockDim.x)
        srow[i] = srow[i] * inv * (normW ? normW[i] : 1.0f);
    __syncthreads();

    unsigned int r = headDim >> 1;
    unsigned int halfW = W >> 1;
    for (unsigned int i = threadIdx.x; i < W; i += blockDim.x)
    {
        unsigned int h = i / headDim;
        unsigned int d = i - h * headDim;
        float val;
        if (cosT != 0)
        {
            unsigned int lane = h * r + (d < r ? d : d - r);
            float c = cosT[(size_t)s * halfW + lane];
            float sn = sinT[(size_t)s * halfW + lane];
            if (d < r) { float a = srow[i], b = srow[i + r]; val = a * c - b * sn; }
            else       { float a = srow[i - r], b = srow[i]; val = b * c + a * sn; }
        }
        else val = srow[i];
        outRow[i] = val;
    }
}
