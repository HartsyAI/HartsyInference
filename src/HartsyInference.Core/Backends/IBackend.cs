using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>Backend interface that all model code programs against. Implementations provide the actual compute (CPU via SIMD, CUDA via PTX/cuBLAS). All operations are eager — they execute immediately and return when complete.</summary>
public interface IBackend : IDisposable
{
    /// <summary>The device this backend targets.</summary>
    DeviceKind Device { get; }

    /// <summary>Count of lazy device-to-host syncs since the last <see cref="ResetD2hSyncCount"/>. A residency
    /// metric: a fully GPU-resident denoise loop fires ~0. Returns 0 for backends with no device sync (CPU).</summary>
    long GetD2hSyncCount() => 0;

    /// <summary>Resets the D2H sync counter. No-op on backends without a device sync.</summary>
    void ResetD2hSyncCount() { }

    /// <summary>Device memory (free, total) in bytes; (0,0) when not applicable (CPU backend).</summary>
    (long FreeBytes, long TotalBytes) GetVramInfo() => (0, 0);

    /// <summary>Capabilities of this backend.</summary>
    BackendCapabilities Capabilities { get; }

    // ── Linear Algebra ──────────────────────────────────────────────────

    /// <summary>Matrix multiply: output = a @ b</summary>
    void MatMul(Tensor output, Tensor a, Tensor b);

    /// <summary>Batched matrix multiply: output[i] = a[i] @ b[i]</summary>
    void BatchedMatMul(Tensor output, Tensor a, Tensor b);

    /// <summary>Linear layer: output = input × weight^T + bias. Input [M, K], weight [N, K], bias [N] (optional), output [M, N]. Also works with leading batch dims: input [B, S, K] → output [B, S, N].</summary>
    void Linear(Tensor output, Tensor input, Tensor weight, Tensor? bias);

    // ── Convolution ─────────────────────────────────────────────────────

    /// <summary>2D convolution: output = conv2d(input, weight, bias, stride, padding)</summary>
    void Conv2D(Tensor output, Tensor input, Tensor weight, Tensor? bias, int strideH, int strideW, int padH, int padW);

    // ── Normalization ───────────────────────────────────────────────────

    /// <summary>Group normalization.</summary>
    void GroupNorm(Tensor output, Tensor input, Tensor weight, Tensor bias, int groups, float eps);

    /// <summary>Layer normalization.</summary>
    void LayerNorm(Tensor output, Tensor input, Tensor weight, Tensor bias, float eps);

    /// <summary>RMS normalization.</summary>
    void RmsNorm(Tensor output, Tensor input, Tensor weight, float eps);

    /// <summary>Adaptive Instance Normalization 1D. Input <c>[B, C, T]</c> is normalized
    /// per-(batch, channel) across the <c>T</c> axis, then affinely scaled by
    /// <c>(1 + gamma[c])</c> and shifted by <c>beta[c]</c>. <paramref name="gamma"/> and
    /// <paramref name="beta"/> are <c>[B, C]</c> (or <c>[C]</c>, broadcast across batch);
    /// they typically come from a Linear projection of a style / speaker embedding —
    /// AdaIN1d is the core style-conditioning primitive in Kokoro and StyleTTS 2.</summary>
    void AdaInstanceNorm1d(Tensor output, Tensor input, Tensor gamma, Tensor beta, float eps);

    // ── DiT transformer glue ────────────────────────────────────────────
    // These keep the per-block modulation math GPU-resident in diffusion transformers
    // (Ideogram 4, etc.). Default implementations are scalar-F32 CPU loops that also serve
    // as the numerical reference; CudaBackend overrides them with PTX kernels.

    /// <summary>Broadcast affine over the last dim: <c>out[b,s,d] = in[b,s,d] * scale[b,d] + (shift?[b,d] ?? 0)</c>.
    /// Input/output are <c>[B, S, D]</c>; <paramref name="scale"/> and <paramref name="shift"/> are <c>[B, D]</c>
    /// broadcast over the sequence axis. With <paramref name="shift"/> null this is a pure broadcast multiply
    /// (the scale-only modulation used by Ideogram 4's adaLN blocks).</summary>
    unsafe void AffineBroadcastLastDim(Tensor output, Tensor input, Tensor scale, Tensor? shift)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32 || scale.DType != DType.F32 || (shift is not null && shift.DType != DType.F32))
            throw new NotSupportedException("AffineBroadcastLastDim default fallback only supports F32.");
        int rank = input.Shape.Rank;
        int dim = (int)input.Shape[rank - 1];
        int seqLen = rank >= 2 ? (int)input.Shape[rank - 2] : 1;
        long total = input.ElementCount;
        float* pIn = (float*)input.DataPointer;
        float* pOut = (float*)output.DataPointer;
        float* pScale = (float*)scale.DataPointer;
        float* pShift = shift is null ? null : (float*)shift.DataPointer;
        for (long i = 0; i < total; i++)
        {
            int d = (int)(i % dim);
            long row = i / dim;
            int b = (int)(row / seqLen);
            long pIdx = (long)b * dim + d;
            float v = pIn[i] * pScale[pIdx];
            if (pShift != null) v += pShift[pIdx];
            pOut[i] = v;
        }
    }

    /// <summary>Gated residual over the last dim: <c>out[b,s,d] = residual[b,s,d] + gate[b,d] * value[b,s,d]</c>.
    /// <paramref name="gate"/> is <c>[B, D]</c> broadcast over the sequence axis.</summary>
    unsafe void GatedResidualLastDim(Tensor output, Tensor residual, Tensor value, Tensor gate)
    {
        if (output.DType != DType.F32 || residual.DType != DType.F32 || value.DType != DType.F32 || gate.DType != DType.F32)
            throw new NotSupportedException("GatedResidualLastDim default fallback only supports F32.");
        int rank = value.Shape.Rank;
        int dim = (int)value.Shape[rank - 1];
        int seqLen = rank >= 2 ? (int)value.Shape[rank - 2] : 1;
        long total = value.ElementCount;
        float* pRes = (float*)residual.DataPointer;
        float* pVal = (float*)value.DataPointer;
        float* pGate = (float*)gate.DataPointer;
        float* pOut = (float*)output.DataPointer;
        for (long i = 0; i < total; i++)
        {
            int d = (int)(i % dim);
            long row = i / dim;
            int b = (int)(row / seqLen);
            long pIdx = (long)b * dim + d;
            pOut[i] = pRes[i] + pGate[pIdx] * pVal[i];
        }
    }

    /// <summary>AdaLN modulation split (scale-only, tanh-gated). <paramref name="proj"/> is <c>[B, 4*D]</c> =
    /// chunk(scale_msa, gate_msa, scale_mlp, gate_mlp); writes four <c>[B, D]</c> tensors applying
    /// <c>1 + x</c> to scales and <c>tanh(x)</c> to gates (Ideogram 4's ComputeModulation).</summary>
    unsafe void ModulationSplit4(Tensor scaleMsa, Tensor gateMsa, Tensor scaleMlp, Tensor gateMlp, Tensor proj)
    {
        if (proj.DType != DType.F32)
            throw new NotSupportedException("ModulationSplit4 default fallback only supports F32.");
        int dim = (int)scaleMsa.Shape[scaleMsa.Shape.Rank - 1];
        int batch = (int)(scaleMsa.ElementCount / dim);
        float* pProj = (float*)proj.DataPointer;
        float* sMsa = (float*)scaleMsa.DataPointer;
        float* gMsa = (float*)gateMsa.DataPointer;
        float* sMlp = (float*)scaleMlp.DataPointer;
        float* gMlp = (float*)gateMlp.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            long src = (long)b * 4 * dim;
            long dst = (long)b * dim;
            for (int d = 0; d < dim; d++)
            {
                sMsa[dst + d] = 1.0f + pProj[src + d];
                gMsa[dst + d] = MathF.Tanh(pProj[src + dim + d]);
                sMlp[dst + d] = 1.0f + pProj[src + 2 * dim + d];
                gMlp[dst + d] = MathF.Tanh(pProj[src + 3 * dim + d]);
            }
        }
    }

    /// <summary>MoE top-k gate weights (the HiDream <c>MoEGate</c>): per token, softmax
    /// <paramref name="logits"/> <c>[B, S, E]</c> over the expert axis, select the top-<paramref name="topK"/>
    /// experts, renormalize the selected weights to sum to 1 (<c>norm_topk_prob</c>: <c>w/(Σw + 1e-20)</c>).
    /// Writes a dense <c>[B, S, E]</c> table with the renormalized weight at selected slots and 0 elsewhere,
    /// so a dense per-expert accumulation reproduces the sparse top-k dispatch exactly.</summary>
    unsafe void MoeTopKGate(Tensor weights, Tensor logits, int topK)
    {
        if (logits.DType != DType.F32 || weights.DType != DType.F32)
            throw new NotSupportedException("MoeTopKGate default fallback only supports F32.");
        int numExperts = (int)logits.Shape[logits.Shape.Rank - 1];
        long tokens = logits.ElementCount / numExperts;
        float* lp = (float*)logits.DataPointer;
        float* wp = (float*)weights.DataPointer;
        Span<float> probs = stackalloc float[numExperts];
        int k = Math.Min(topK, numExperts);
        for (long t = 0; t < tokens; t++)
        {
            long off = t * numExperts;
            float maxLogit = lp[off];
            for (int e = 1; e < numExperts; e++)
                if (lp[off + e] > maxLogit) maxLogit = lp[off + e];
            float sum = 0f;
            for (int e = 0; e < numExperts; e++)
            {
                float ex = MathF.Exp(lp[off + e] - maxLogit);
                probs[e] = ex;
                sum += ex;
            }
            for (int e = 0; e < numExperts; e++)
            {
                probs[e] /= sum;
                wp[off + e] = 0f;
            }
            float denom = 0f;
            for (int kk = 0; kk < k; kk++)
            {
                int best = -1;
                float bestVal = float.NegativeInfinity;
                for (int e = 0; e < numExperts; e++)
                {
                    if (wp[off + e] != 0f) continue;
                    if (probs[e] > bestVal) { bestVal = probs[e]; best = e; }
                }
                wp[off + best] = bestVal;
                denom += bestVal;
            }
            denom += 1e-20f;
            for (int e = 0; e < numExperts; e++)
                if (wp[off + e] != 0f) wp[off + e] /= denom;
        }
    }

    /// <summary>In-place per-token gated accumulate (the MoE expert combine):
    /// <c>inout[b,s,:] += gate[(b,s), expertIndex] * value[b,s,:]</c>, where <paramref name="gate"/> is the
    /// dense <c>[B, S, E]</c> table from <see cref="MoeTopKGate"/> (zero for non-selected tokens).</summary>
    unsafe void RowGatedAccumulate(Tensor inout, Tensor value, Tensor gate, int numExperts, int expertIndex)
    {
        if (inout.DType != DType.F32 || value.DType != DType.F32 || gate.DType != DType.F32)
            throw new NotSupportedException("RowGatedAccumulate default fallback only supports F32.");
        int dim = (int)inout.Shape[inout.Shape.Rank - 1];
        long rows = inout.ElementCount / dim;
        float* op = (float*)inout.DataPointer;
        float* vp = (float*)value.DataPointer;
        float* gp = (float*)gate.DataPointer;
        for (long r = 0; r < rows; r++)
        {
            float g = gp[r * numExperts + expertIndex];
            if (g == 0f) continue;
            long off = r * dim;
            for (int d = 0; d < dim; d++)
                op[off + d] += g * vp[off + d];
        }
    }

    /// <summary>Classifier-free-guidance combine + Euler step, in-place on <paramref name="z"/>:
    /// <c>v = guidance*pos + (1-guidance)*neg; z += v*delta</c>. Flat element-wise over the latent.</summary>
    unsafe void CfgEulerStep(Tensor z, Tensor pos, Tensor neg, float guidance, float delta)
    {
        if (z.DType != DType.F32 || pos.DType != DType.F32 || neg.DType != DType.F32)
            throw new NotSupportedException("CfgEulerStep default fallback only supports F32.");
        long count = z.ElementCount;
        float* pZ = (float*)z.DataPointer;
        float* pPos = (float*)pos.DataPointer;
        float* pNeg = (float*)neg.DataPointer;
        for (long i = 0; i < count; i++)
        {
            float v = guidance * pPos[i] + (1.0f - guidance) * pNeg[i];
            pZ[i] = pZ[i] + v * delta;
        }
    }

    /// <summary>In-place rotary position embedding on <paramref name="q"/> and <paramref name="k"/>, both
    /// <c>[B, L, numHeads, headDim]</c>. <paramref name="cos"/>/<paramref name="sin"/> are <c>[B, L, headDim]</c>
    /// broadcast over heads. <c>out[i] = x[i]*cos[i] + rotate_half(x)[i]*sin[i]</c> with
    /// <c>rotate_half(x) = cat(-x[half:], x[:half])</c> (Ideogram 4's ApplyRotary).</summary>
    unsafe void ApplyRope(Tensor q, Tensor k, Tensor cos, Tensor sin)
    {
        if (q.DType != DType.F32 || k.DType != DType.F32 || cos.DType != DType.F32 || sin.DType != DType.F32)
            throw new NotSupportedException("ApplyRope default fallback only supports F32.");
        int batch = (int)q.Shape[0];
        int seqLen = (int)q.Shape[1];
        int numHeads = (int)q.Shape[2];
        int headDim = (int)q.Shape[3];
        int half = headDim / 2;
        float* qPtr = (float*)q.DataPointer;
        float* kPtr = (float*)k.DataPointer;
        float* cosPtr = (float*)cos.DataPointer;
        float* sinPtr = (float*)sin.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                long freqBase = ((long)b * seqLen + s) * headDim;
                for (int h = 0; h < numHeads; h++)
                {
                    long vecOff = (((long)b * seqLen + s) * numHeads + h) * headDim;
                    RotateHalfInPlace(qPtr + vecOff, cosPtr + freqBase, sinPtr + freqBase, half);
                    RotateHalfInPlace(kPtr + vecOff, cosPtr + freqBase, sinPtr + freqBase, half);
                }
            }
        }

        static void RotateHalfInPlace(float* vec, float* cos, float* sin, int half)
        {
            for (int i = 0; i < half; i++)
            {
                float lower = vec[i];
                float upper = vec[i + half];
                vec[i] = lower * cos[i] - upper * sin[i];
                vec[i + half] = upper * cos[i + half] + lower * sin[i + half];
            }
        }
    }

    /// <summary>In-place <b>interleaved (GPT-J)</b> rotary embedding on a single tensor <paramref name="x"/> of
    /// shape <c>[B, L, numHeads, headDim]</c>: adjacent dims are rotated in pairs <c>(2i, 2i+1)</c> sharing
    /// frequency <c>i</c>. <paramref name="cos"/>/<paramref name="sin"/> are <c>[B, L, headDim]</c> (only the
    /// first <c>headDim/2</c> entries per position are read). Used by the Qwen3-TTS audio backbone / Moonshine;
    /// NOT interchangeable with the split-half <see cref="ApplyRopeSingle"/>.</summary>
    unsafe void ApplyRopeInterleaved(Tensor x, Tensor cos, Tensor sin)
    {
        if (x.DType != DType.F32 || cos.DType != DType.F32 || sin.DType != DType.F32)
            throw new NotSupportedException("ApplyRopeInterleaved default fallback only supports F32.");
        int batch = (int)x.Shape[0];
        int seqLen = (int)x.Shape[1];
        int numHeads = (int)x.Shape[2];
        int headDim = (int)x.Shape[3];
        int half = headDim / 2;
        float* xPtr = (float*)x.DataPointer;
        float* cosPtr = (float*)cos.DataPointer;
        float* sinPtr = (float*)sin.DataPointer;
        for (int b = 0; b < batch; b++)
            for (int s = 0; s < seqLen; s++)
            {
                long freqBase = ((long)b * seqLen + s) * headDim;
                for (int h = 0; h < numHeads; h++)
                {
                    long vecOff = (((long)b * seqLen + s) * numHeads + h) * headDim;
                    float* vec = xPtr + vecOff;
                    float* c = cosPtr + freqBase;
                    float* si = sinPtr + freqBase;
                    for (int i = 0; i < half; i++)
                    {
                        float xe = vec[2 * i];
                        float xo = vec[2 * i + 1];
                        vec[2 * i] = xe * c[i] - xo * si[i];
                        vec[2 * i + 1] = xo * c[i] + xe * si[i];
                    }
                }
            }
    }

    /// <summary>Wan-Video interleaved in-place RoPE (shared cos/sin path): <paramref name="x"/> is
    /// <c>[S, heads·headDim]</c>; for each <c>(s, head, pair i)</c> the adjacent pair <c>(2i, 2i+1)</c> is rotated by
    /// the angle at cos/sin index <c>2i</c> (Wan's duplicated-pair layout), with <paramref name="cos"/>/<paramref
    /// name="sin"/> <c>[S, headDim]</c> shared across heads. Matches <c>WanRope.ApplyRotary</c>. CUDA overrides with a
    /// kernel so the attention chain stays GPU-resident; the default is the CPU reference.</summary>
    unsafe void WanRopeInterleaved(Tensor x, Tensor cos, Tensor sin, int seqLen, int heads, int headDim)
    {
        float* xp = (float*)x.DataPointer, cp = (float*)cos.DataPointer, sp = (float*)sin.DataPointer;
        int pairs = headDim / 2;
        for (int s = 0; s < seqLen; s++)
            for (int h = 0; h < heads; h++)
            {
                long xoff = ((long)s * heads + h) * headDim;
                long coff = (long)s * headDim;
                for (int i = 0; i < pairs; i++)
                {
                    int i0 = 2 * i;
                    float re = xp[xoff + i0], im = xp[xoff + i0 + 1];
                    float c = cp[coff + i0], sn = sp[coff + i0];
                    xp[xoff + i0] = re * c - im * sn;
                    xp[xoff + i0 + 1] = re * sn + im * c;
                }
            }
    }

    /// <summary>LTX-2 "split" rotary (rotate-half, per-head cos), in-place on <paramref name="x"/>
    /// <c>[seqLen, numHeads*headDim]</c>. <paramref name="cos"/>/<paramref name="sin"/> are <c>[seqLen, dim/2]</c>
    /// laid out per-head (width <c>headDim/2</c>), one angle shared by both elements of a pair
    /// <c>(i, i+headDim/2)</c>. Matches <c>LtxVideo2Rope</c>'s split branch. CUDA overrides with a kernel so the
    /// attention chain stays GPU-resident; the default is the CPU reference.</summary>
    unsafe void Ltx2SplitRope(Tensor x, Tensor cos, Tensor sin, int seqLen, int numHeads, int headDim)
    {
        float* xp = (float*)x.DataPointer, cp = (float*)cos.DataPointer, sp = (float*)sin.DataPointer;
        int r = headDim / 2;
        int dim = numHeads * headDim;
        int cosWidth = dim / 2;
        for (int s = 0; s < seqLen; s++)
        {
            long xoff = (long)s * dim;
            long coff = (long)s * cosWidth;
            for (int h = 0; h < numHeads; h++)
            {
                long headBase = xoff + (long)h * headDim;
                long cosBase = coff + (long)h * r;
                for (int i = 0; i < r; i++)
                {
                    float a = xp[headBase + i];
                    float b = xp[headBase + i + r];
                    float c = cp[cosBase + i], sn = sp[cosBase + i];
                    xp[headBase + i] = a * c - b * sn;
                    xp[headBase + i + r] = b * c + a * sn;
                }
            }
        }
    }

    /// <summary>Extracts temporal frame <paramref name="ti"/> of a 5D <c>[B,C,Tsrc,H,W]</c> source into the 4D
    /// <c>[B,C,H,W]</c> <paramref name="output"/> (a strided temporal slice). Keeps 3D-VAE conv frame ops on-device;
    /// CUDA overrides, default is a host copy.</summary>
    unsafe void ExtractVaeFrame(Tensor output, Tensor src, int ti)
    {
        int b = (int)output.Shape[0], c = (int)output.Shape[1];
        long frameHW = output.ElementCount / ((long)b * c);
        int tsrc = (int)src.Shape[2];
        float* op = (float*)output.DataPointer, sp = (float*)src.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (int ci = 0; ci < c; ci++)
                Buffer.MemoryCopy(sp + ((((long)bi * c + ci) * tsrc + ti) * frameHW), op + (((long)bi * c + ci) * frameHW), frameHW * 4, frameHW * 4);
    }

    /// <summary>Writes the 4D <c>[B,C,H,W]</c> frame <paramref name="acc"/> (plus optional per-channel
    /// <paramref name="bias"/>) into temporal slot <paramref name="to"/> of the 5D <c>[B,C,Tout,H,W]</c>
    /// <paramref name="output"/>, in place. CUDA overrides; default is a host copy.</summary>
    unsafe void WriteVaeFrame(Tensor output, Tensor acc, Tensor? bias, int to)
    {
        int b = (int)acc.Shape[0], c = (int)acc.Shape[1];
        long frameHW = acc.ElementCount / ((long)b * c);
        int tout = (int)output.Shape[2];
        float* op = (float*)output.DataPointer, ap = (float*)acc.DataPointer;
        float* bp = bias is null ? null : (float*)bias.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (int ci = 0; ci < c; ci++)
            {
                float bv = bp is null ? 0f : bp[ci];
                long srcOff = ((long)bi * c + ci) * frameHW, dstOff = (((long)bi * c + ci) * tout + to) * frameHW;
                for (long i = 0; i < frameHW; i++) op[dstOff + i] = ap[srcOff + i] + bv;
            }
    }

    /// <summary>Builds the frame-major padded input <c>[paddedT, cIn, H+2·padH, W+2·padW]</c> for BATCHED
    /// CausalConv3d: transpose (cIn↔T) + temporal pad (first <paramref name="zeroPad"/> frames: zeros, or the first
    /// content frame when <paramref name="replicateFirst"/>) + cache prepend + optional spatial edge-replicate pad
    /// (<paramref name="padH"/>/<paramref name="padW"/> &gt; 0 → source pixel clamped, so the caller convolves with
    /// zero padding disabled), from <paramref name="input"/> <c>[1,cIn,Tin,H,W]</c> and optional
    /// <paramref name="cache"/> <c>[1,cIn,cacheLen,H,W]</c>. CUDA overrides.</summary>
    unsafe void BuildPaddedFrames(Tensor padded, Tensor input, Tensor? cache, int zeroPad,
        bool replicateFirst = false, int padH = 0, int padW = 0)
    {
        int paddedT = (int)padded.Shape[0], cIn = (int)padded.Shape[1];
        int H = (int)input.Shape[3], W = (int)input.Shape[4];
        int Hp = H + 2 * padH, Wp = W + 2 * padW;
        long HW = (long)H * W;
        int Tin = (int)input.Shape[2];
        int cacheLen = cache is null ? 0 : (int)cache.Shape[2];
        float* pp = (float*)padded.DataPointer, ip = (float*)input.DataPointer;
        float* cp = cache is null ? null : (float*)cache.DataPointer;
        for (int t = 0; t < paddedT; t++)
            for (int ci = 0; ci < cIn; ci++)
            {
                long dst = ((long)t * cIn + ci) * Hp * Wp;
                if (t < zeroPad && !replicateFirst)
                {
                    for (long i = 0; i < (long)Hp * Wp; i++) pp[dst + i] = 0f;
                    continue;
                }
                int az = t < zeroPad ? 0 : t - zeroPad;   // replicate-first: pad frames clamp to the first content frame
                float* src;
                if (az < cacheLen) src = cp! + ((long)ci * cacheLen + az) * HW;
                else
                {
                    int ti = az - cacheLen;
                    if (ti >= Tin) ti = Tin - 1;          // trailing clamp (non-causal replicate)
                    src = ip + ((long)ci * Tin + ti) * HW;
                }
                for (int y = 0; y < Hp; y++)
                {
                    int sy = Math.Clamp(y - padH, 0, H - 1);
                    for (int x = 0; x < Wp; x++)
                    {
                        int sx = Math.Clamp(x - padW, 0, W - 1);
                        pp[dst + (long)y * Wp + x] = src[(long)sy * W + sx];
                    }
                }
            }
    }

    /// <summary>Fills <paramref name="output"/> <c>[1,cOut,tout,H,W]</c> with per-channel <paramref name="bias"/>
    /// (or 0 when null). CUDA overrides.</summary>
    unsafe void FillBias(Tensor output, Tensor? bias)
    {
        int cOut = (int)output.Shape[1], tout = (int)output.Shape[2];
        long HW = output.ElementCount / ((long)cOut * tout);
        float* op = (float*)output.DataPointer; float* bp = bias is null ? null : (float*)bias.DataPointer;
        for (int co = 0; co < cOut; co++)
        {
            float bv = bp is null ? 0f : bp[co];
            for (int to = 0; to < tout; to++) { long o = ((long)co * tout + to) * HW; for (long i = 0; i < HW; i++) op[o + i] = bv; }
        }
    }

    /// <summary>Temporal gather-sum for batched CausalConv3d (in place):
    /// <c>output[co][to][hw] += convDt[to·strideT+dt][co][hw]</c>. <paramref name="output"/> is
    /// <c>[1,cOut,tout,H,W]</c>, <paramref name="convDt"/> is <c>[paddedT,cOut,H,W]</c>. CUDA overrides.</summary>
    unsafe void AccumulateTap(Tensor output, Tensor convDt, int dt, int strideT)
    {
        int cOut = (int)output.Shape[1], tout = (int)output.Shape[2];
        long HW = output.ElementCount / ((long)cOut * tout);
        float* op = (float*)output.DataPointer, cp = (float*)convDt.DataPointer;
        for (int co = 0; co < cOut; co++)
            for (int to = 0; to < tout; to++)
            {
                int srcT = to * strideT + dt;
                long o = ((long)co * tout + to) * HW, s = ((long)srcT * cOut + co) * HW;
                for (long i = 0; i < HW; i++) op[o + i] += cp[s + i];
            }
    }

    /// <summary>Wan2.2 VAE channel-wise RMS norm (<c>vae.py</c> <c>RMS_norm</c>): for each position over the flattened
    /// <c>[B, C, spatial]</c>, <c>scale = sqrt(C)/max(L2_over_C, eps)</c> and <c>out[c] = x[c]·scale·gamma[c]</c>
    /// (equivalently x/rms over the channel axis). <paramref name="gamma"/> may be null. Default = host reference loop.</summary>
    unsafe void WanRmsNormChannel(Tensor output, Tensor input, Tensor? gamma, float eps)
    {
        int b = (int)input.Shape[0], c = (int)input.Shape[1];
        long spatial = input.ElementCount / ((long)b * c);
        float sqrtC = MathF.Sqrt(c);
        float* xp = (float*)input.DataPointer, op = (float*)output.DataPointer;
        float* g = gamma is null ? null : (float*)gamma.DataPointer;
        for (int bi = 0; bi < b; bi++)
        {
            long baseB = (long)bi * c * spatial;
            for (long s = 0; s < spatial; s++)
            {
                double sumSq = 0;
                for (int ci = 0; ci < c; ci++) { float v = xp[baseB + (long)ci * spatial + s]; sumSq += (double)v * v; }
                float denom = MathF.Max((float)Math.Sqrt(sumSq), eps);
                float scale = sqrtC / denom;
                for (int ci = 0; ci < c; ci++)
                {
                    long off = baseB + (long)ci * spatial + s;
                    op[off] = xp[off] * scale * (g is null ? 1f : g[ci]);
                }
            }
        }
    }

    // ── Step-graph capture (CUDA-graph replay of a fixed per-step op sequence) ─────────────────────────────
    // A pipeline whose denoise step issues an IDENTICAL op sequence every step can capture it once and replay
    // with a single graph launch, collapsing thousands of per-op host calls. Contract: between Begin and
    // EndAndLaunch only async device work is issued (no DataPointer reads, no sync allocs); per-step-varying
    // inputs are refreshed into FIXED device buffers via CopyInto before each Launch. Defaults: unsupported.

    /// <summary>True when the backend can capture/replay a step graph (CUDA only).</summary>
    bool StepGraphSupported => false;

    /// <summary>True when a captured step graph is ready to <see cref="StepGraphLaunch"/>.</summary>
    bool StepGraphReady => false;

    /// <summary>Begins capturing subsequent backend ops into the step graph.</summary>
    void StepGraphBegin() { }

    /// <summary>Ends capture, instantiates the graph, and launches it once (capture records without executing).</summary>
    void StepGraphEndAndLaunch() { }

    /// <summary>Replays the captured step graph (one launch call).</summary>
    void StepGraphLaunch() { }

    /// <summary>Drops the captured graph (shape/prompt change → recapture) and disables an in-flight capture.</summary>
    void StepGraphReset() { }

    /// <summary>Identity of the model instance whose capture currently occupies the (single) step-graph slot.
    /// The slot is SHARED across models: a transformer must verify it still owns the slot before
    /// <see cref="StepGraphLaunch"/> — when models alternate, replaying a stale "ready" graph would silently run
    /// ANOTHER model's step. Set it when starting a capture cycle; owner mismatch → reset + recapture.</summary>
    object? StepGraphOwner { get => null; set { } }

    /// <summary>Copies <paramref name="src"/>'s data into <paramref name="dst"/>'s EXISTING storage, preserving
    /// dst's buffer address (unlike normal ops, which allocate fresh output buffers). Used to refresh a captured
    /// graph's fixed input buffers between launches. Shapes/dtypes must match. Default = host memcpy.</summary>
    unsafe void CopyInto(Tensor dst, Tensor src)
    {
        long bytes = dst.DType.ComputeByteCount(dst.ElementCount);
        Buffer.MemoryCopy((void*)src.DataPointer, (void*)dst.DataPointer, bytes, bytes);
    }

    // ── Device-side decode position (autoregressive step-graph replay) ──────────────────────────────────────
    // A captured decode graph must NOT re-bake the per-step KV position into its kernel params, or every replay
    // would append/attend at the capture-time position. Instead the self-attention KV-append and flash kernels
    // read the position from a small device buffer {kvLen, qOffset}, refreshed (outside capture) each step. This
    // lets ONE captured graph replay at every step. Defaults: unsupported (eager backends use the host ints).

    /// <summary>True when the backend supports device-position decode-graph replay (CUDA only).</summary>
    bool GraphDecodeSupported => false;

    /// <summary>Allocates a persistent 2-int device buffer holding {kvLen, qOffset}; returns an opaque handle
    /// (0 = unsupported). Free with <see cref="FreeDevicePos"/>.</summary>
    ulong AllocDevicePos() => 0;

    /// <summary>Refreshes the device position buffer (outside any capture region) for the next step/launch.</summary>
    void WriteDevicePos(ulong handle, int kvLen, int qOffset) { }

    /// <summary>Frees a buffer from <see cref="AllocDevicePos"/>.</summary>
    void FreeDevicePos(ulong handle) { }

    /// <summary>KV-cache append whose write slot comes from the device position buffer when <paramref name="devicePos"/>
    /// is non-zero (graph-replayable); otherwise identical to <see cref="KvCacheAppend"/> at host <paramref name="offset"/>.</summary>
    void KvCacheAppendDev(Tensor buffer, Tensor newKv, int offset, ulong devicePos) => KvCacheAppend(buffer, newKv, offset);

    /// <summary>Self-attention FlashAttention whose kvLen/qOffset come from the device position buffer when
    /// <paramref name="devicePos"/> is non-zero (graph-replayable, forces the fixed-grid path); otherwise identical
    /// to <see cref="FlashAttention"/> at the host <paramref name="kvLen"/>/<paramref name="qOffset"/>.</summary>
    void FlashAttentionDev(Tensor output, Tensor query, Tensor key, Tensor value, int kvLen, int kvGroup, bool causal, int qOffset, float scale, ulong devicePos)
        => FlashAttention(output, query, key, value, kvLen, kvGroup, causal, qOffset, scale);

    /// <summary>Copies a RESIDENT device tensor's data to <paramref name="dst"/> WITHOUT freeing/detaching its
    /// device buffer (unlike a normal <c>DataPointer</c> read, which syncs then frees it). Required for a captured
    /// graph's fixed OUTPUT buffers that are read back on the host every step but must keep their baked device
    /// address across replays. Default = host read.</summary>
    unsafe void ReadResidentInto(Tensor src, float[] dst)
    {
        new Span<float>((void*)src.DataPointer, dst.Length).CopyTo(dst);
    }

    // ── LLM decode-graph: RoPE / embed / argmax reading & writing device state ──────────────────────────────
    // The remaining per-token host work a captured graph can't tolerate: RoPE's cos/sin table build
    // (Math.Cos/Sin + H2D every step) and the embedding lookup (host gather from the token id). Both convert
    // to a one-time device-resident precompute (RoPE table) / preload (embed table) plus a fixed-grid kernel
    // indexed by device state — the SAME devicePos buffer above for RoPE's position, and a persistent 1-int
    // "current token id" buffer for embed (see AllocDeviceTokenId) that greedy decode chains entirely
    // on-device via ArgMaxInto (this step's sampled token becomes next step's embed input, no D2H between).

    /// <summary>Allocates a persistent 1-int device buffer for the "token id to embed next" (0 = unsupported).</summary>
    ulong AllocDeviceTokenId() => 0;

    /// <summary>Writes the token id into a buffer from <see cref="AllocDeviceTokenId"/> (outside capture — the
    /// prefill's sampled first token, or any host-driven override; graph replay chains via <see cref="ArgMaxInto"/> instead).</summary>
    void WriteDeviceTokenId(ulong handle, int tokenId) { }

    /// <summary>Frees a buffer from <see cref="AllocDeviceTokenId"/>.</summary>
    void FreeDeviceTokenId(ulong handle) { }

    /// <summary>Reads back the token id from a buffer from <see cref="AllocDeviceTokenId"/> (a 1-int D2H sync —
    /// the ONE per-step host readback a graph-decode loop still needs, for streaming/detokenization/stop-checking;
    /// everything else about the step stays device-resident). Throws when unsupported.</summary>
    int ReadDeviceTokenId(ulong handle) => throw new NotSupportedException("ReadDeviceTokenId requires GraphDecodeSupported.");

    /// <summary>Precomputes the full RoPE cos/sin table <c>[maxPos, headDim]</c> (duplicated-half layout,
    /// identical math to the per-step table a transformer builds each call) as two device-resident tensors, so
    /// a decode-step RoPE apply only needs to index a row, not rebuild it. Returns (cos, sin); dispose both
    /// when the graphed session ends.</summary>
    (Tensor cos, Tensor sin) BuildRopeTableDevice(int maxPos, int headDim, int rotaryDim, float theta, HartsyInference.Core.Rope.RopeScaling scaling)
        => throw new NotSupportedException("BuildRopeTableDevice requires GraphDecodeSupported.");

    /// <summary>RoPE on a single decode-step Q/K tensor <c>[1,numHeads,1,headDim]</c>, reading the rotation row
    /// from <paramref name="cosTable"/>/<paramref name="sinTable"/> (from <see cref="BuildRopeTableDevice"/>) at
    /// the position in <paramref name="devicePos"/> (its qOffset slot) — graph-replayable. No-op default.</summary>
    void RopeApplyDecodeStep(Tensor x, Tensor cosTable, Tensor sinTable, int rotaryDim, bool interleaved, ulong devicePos) { }

    /// <summary>Gathers one row (the id in the persistent buffer <paramref name="tokenId"/>) from a GPU-resident
    /// embedding table into <paramref name="output"/> <c>[1,1,hidden]</c> — graph-replayable. No-op default.</summary>
    void EmbedGatherDecodeStep(Tensor output, Tensor embedTable, ulong tokenId) { }

    /// <summary>Argmax over the last dim, writing directly into a persistent device buffer (e.g. from
    /// <see cref="AllocDeviceTokenId"/>) instead of allocating a fresh output — so the SAME buffer a captured
    /// graph's embed-gather reads is the one this step's argmax writes, chaining greedy decode on-device with
    /// no D2H between steps. No-op default.</summary>
    void ArgMaxInto(ulong outputTokenId, Tensor input) { }

    // ── Graph-capture decode: repetition penalty ────────────────────────────────────────────────────────────
    // Greedy graph decode previously skipped the sampler chain entirely (raw argmax on unpenalized logits),
    // silently ignoring a nonzero repetition penalty — repetition penalty is the ONLY sampler stage that can
    // change greedy's picked token (temperature is monotonic, top-k/top-p/min-p never remove the argmax
    // survivor), so it's also the only one graph decode needs to replicate. History (the model's own generated
    // tokens, not the prompt) lives in a fixed-capacity device buffer appended to once per replay.

    /// <summary>Allocates a persistent int[<paramref name="capacity"/>] device buffer for graph-decode
    /// repetition-penalty history (0 = unsupported). Free with <see cref="FreeDeviceHistory"/>.</summary>
    ulong AllocDeviceHistory(int capacity) => 0;

    /// <summary>Frees a buffer from <see cref="AllocDeviceHistory"/>.</summary>
    void FreeDeviceHistory(ulong handle) { }

    /// <summary>Allocates a persistent 1-int device counter (0 = unsupported). Free with
    /// <see cref="FreeDeviceCounter"/>.</summary>
    ulong AllocDeviceCounter() => 0;

    /// <summary>Frees a buffer from <see cref="AllocDeviceCounter"/>.</summary>
    void FreeDeviceCounter(ulong handle) { }

    /// <summary>Writes a device counter's value (outside capture — e.g. resetting to 0 at session start).</summary>
    void WriteDeviceCounter(ulong handle, int value) { }

    /// <summary>Appends the current value of <paramref name="tokenId"/> (from <see cref="AllocDeviceTokenId"/>)
    /// into <paramref name="history"/> at the index in <paramref name="historyCount"/>, then increments the
    /// counter — graph-replayable (fixed 1-thread launch). No-op default.</summary>
    void AppendTokenHistoryStep(ulong history, ulong historyCount, ulong tokenId) { }

    /// <summary>Applies HF-convention repetition penalty (divide positive logits, multiply negative) to every
    /// token id in <paramref name="history"/>[0, *<paramref name="historyCount"/>), sequentially — matches
    /// <c>RepetitionPenaltyStep</c>'s CPU semantics exactly, including a token occurring more than once being
    /// penalized cumulatively. Graph-replayable (fixed 1-thread launch). No-op default.</summary>
    void ApplyRepetitionPenaltyStep(Tensor logits, ulong history, ulong historyCount, float penalty) { }

    // ── Graph capture/replay (backend-agnostic handle) ──────────────────────────────────────────────────────
    // Callers outside HartsyInference.Cuda (e.g. TextGenerationPipeline, which must stay backend-agnostic —
    // CPU builds link against IBackend only) capture/replay through this opaque-handle indirection instead of
    // referencing CudaGraph directly. Default: unsupported: CaptureGraph returns null, meaning "not captured,
    // call recordWork() yourself every time" — callers must treat a null handle as the eager fallback, not an error.

    /// <summary>Captures the device work <paramref name="recordWork"/> issues into a graph and instantiates it;
    /// returns an opaque handle for <see cref="LaunchGraph"/>/<see cref="DisposeGraph"/>, or null when the
    /// backend doesn't support graph capture (caller falls back to calling <paramref name="recordWork"/>
    /// directly on every step instead of ever calling LaunchGraph).</summary>
    object? CaptureGraph(Action recordWork) => null;

    /// <summary>Replays a graph captured by <see cref="CaptureGraph"/> with a single launch.</summary>
    void LaunchGraph(object graphHandle) => throw new NotSupportedException("LaunchGraph requires a handle from a backend whose CaptureGraph returned non-null.");

    /// <summary>Disposes a graph handle from <see cref="CaptureGraph"/>.</summary>
    void DisposeGraph(object graphHandle) { }

    /// <summary>Image-output conversion: CHW F32 in [-1,1] → interleaved HWC u8 in [0,255] (round-half-up, clamped).
    /// <paramref name="input"/> is <c>[B(=1), 3, H, W]</c> F32; <paramref name="output"/> is <c>[H, W, 3]</c> U8.
    /// Device backends convert on-GPU so only the 3-byte/pixel result crosses PCIe. Default = host reference loop.</summary>
    unsafe void ChwF32ToHwcU8(Tensor output, Tensor input)
    {
        int height = (int)input.Shape[2], width = (int)input.Shape[3];
        long hw = (long)height * width;
        float* ip = (float*)input.DataPointer;
        byte* op = (byte*)output.DataPointer;
        for (long i = 0; i < hw; i++)
        {
            for (int c = 0; c < 3; c++)
            {
                float v = (ip[c * hw + i] + 1.0f) * 0.5f;
                v = Math.Clamp(v, 0.0f, 1.0f);
                op[i * 3 + c] = (byte)(v * 255.0f + 0.5f);
            }
        }
    }

    /// <summary>DiT token-grid unpatchify: <c>[1, hP·wP, C·p²]</c> → <c>[1, C, hP·p, wP·p]</c> (F32, B=1).
    /// <paramref name="innerChannelFastest"/> selects the token inner order: true = (ph, pw, c) (Z-Image/Lumina2),
    /// false = channel-outer (c, ph, pw) (Krea2/diffusers). Keeps the end-of-loop latent → VAE chain device-resident
    /// (the host unpatchify loop D2H-drained the tokens and re-uploaded for the decode). Default = host loop.</summary>
    unsafe void UnpatchifyTokens(Tensor output, Tensor tokens, int channels, int hPacked, int wPacked, int patch,
        bool innerChannelFastest)
    {
        int H = hPacked * patch, W = wPacked * patch;
        int patchVol = channels * patch * patch;
        float* src = (float*)tokens.DataPointer;
        float* dst = (float*)output.DataPointer;
        long hw = (long)H * W;
        for (int c = 0; c < channels; c++)
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                {
                    int hp = y / patch, ph = y % patch;
                    int wp = x / patch, pw = x % patch;
                    long seq = (long)hp * wPacked + wp;
                    long inner = innerChannelFastest
                        ? ((long)(ph * patch + pw) * channels + c)
                        : (((long)c * patch + ph) * patch + pw);
                    dst[c * hw + (long)y * W + x] = src[seq * patchVol + inner];
                }
    }

    /// <summary>Wan2.2 VAE unpatchify / pixel-shuffle: <c>[b, c·p², t, h, w] → [b, c, t, h·p, w·p]</c> with channel
    /// unpack <c>oc = ci·p² + r·p + q</c> at out spatial <c>(hh·p+q, ww·p+r)</c>. Default = host reference loop.</summary>
    unsafe void UnpatchifyVae(Tensor output, Tensor input, int patchSize)
    {
        int b = (int)input.Shape[0], packedC = (int)input.Shape[1], t = (int)input.Shape[2], h = (int)input.Shape[3], w = (int)input.Shape[4];
        int p = patchSize, c = packedC / (p * p), outH = h * p, outW = w * p;
        float* src = (float*)input.DataPointer, dst = (float*)output.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (int ci = 0; ci < c; ci++)
                for (int ti = 0; ti < t; ti++)
                    for (int hh = 0; hh < h; hh++)
                        for (int ww = 0; ww < w; ww++)
                            for (int q = 0; q < p; q++)
                                for (int r = 0; r < p; r++)
                                {
                                    int oc = ci * p * p + r * p + q;
                                    long srcOff = ((((long)bi * packedC + oc) * t + ti) * h + hh) * w + ww;
                                    long dstOff = ((((long)bi * c + ci) * t + ti) * outH + (hh * p + q)) * outW + (ww * p + r);
                                    dst[dstOff] = src[srcOff];
                                }
    }

    /// <summary>Wan2.2 VAE attention qkv split: <c>src [bt, 3c, h, w] → q,k,v each [bt, 1, hw, c]</c>. Default = host loop.</summary>
    unsafe void SplitVaeQkv(Tensor q, Tensor k, Tensor v, Tensor qkv, int bt, int c, int hw)
    {
        float* src = (float*)qkv.DataPointer, qp = (float*)q.DataPointer, kp = (float*)k.DataPointer, vp = (float*)v.DataPointer;
        long frame = hw;
        for (int i = 0; i < bt; i++)
            for (int ci = 0; ci < c; ci++)
                for (int token = 0; token < hw; token++)
                {
                    long srcBase = ((long)i * 3 * c) * frame + token;
                    long dstOff = ((long)i * hw + token) * c + ci;
                    qp[dstOff] = src[srcBase + (long)ci * frame];
                    kp[dstOff] = src[srcBase + (long)(c + ci) * frame];
                    vp[dstOff] = src[srcBase + (long)(2 * c + ci) * frame];
                }
    }

    /// <summary>Wan2.2 VAE attention output un-transpose: <c>attn [bt, 1, hw, c] → out [bt, c, h, w]</c>. Default = host loop.</summary>
    unsafe void VaeTokensToFrame(Tensor output, Tensor attn, int bt, int c, int hw)
    {
        float* a = (float*)attn.DataPointer, o = (float*)output.DataPointer;
        long frame = hw;
        for (int i = 0; i < bt; i++)
            for (int ci = 0; ci < c; ci++)
                for (int token = 0; token < hw; token++)
                    o[((long)i * c + ci) * frame + token] = a[((long)i * hw + token) * c + ci];
    }

    /// <summary>In-place rotary position embedding on a single tensor <paramref name="x"/> of shape
    /// <c>[B, L, numHeads, headDim]</c>; <paramref name="cos"/>/<paramref name="sin"/> are <c>[B, L, headDim]</c>
    /// broadcast over heads. Same rotate-half math as <see cref="ApplyRope"/> but for one tensor — required for
    /// grouped-query attention where Q and K have different head counts (the paired overload would mis-stride K).</summary>
    unsafe void ApplyRopeSingle(Tensor x, Tensor cos, Tensor sin, int rotaryDim = 0)
    {
        if (x.DType != DType.F32 || cos.DType != DType.F32 || sin.DType != DType.F32)
            throw new NotSupportedException("ApplyRopeSingle default fallback only supports F32.");
        int batch = (int)x.Shape[0];
        int seqLen = (int)x.Shape[1];
        int numHeads = (int)x.Shape[2];
        int headDim = (int)x.Shape[3];
        // Partial rotary (Phi-4 / StableLM): rotate only the first rotaryDim dims of each head (NEOX pairing
        // (i, i+half), half = rotaryDim/2), leaving the rest unchanged. 0 / full = the whole head. cos/sin keep
        // headDim stride (only their first rotaryDim entries are read).
        int rdim = rotaryDim <= 0 || rotaryDim > headDim ? headDim : rotaryDim;
        int half = rdim / 2;
        float* xPtr = (float*)x.DataPointer;
        float* cosPtr = (float*)cos.DataPointer;
        float* sinPtr = (float*)sin.DataPointer;
        for (int b = 0; b < batch; b++)
            for (int s = 0; s < seqLen; s++)
            {
                long freqBase = ((long)b * seqLen + s) * headDim;
                for (int h = 0; h < numHeads; h++)
                {
                    long vecOff = (((long)b * seqLen + s) * numHeads + h) * headDim;
                    float* vec = xPtr + vecOff;
                    float* c = cosPtr + freqBase;
                    float* si = sinPtr + freqBase;
                    for (int i = 0; i < half; i++)
                    {
                        float lower = vec[i];
                        float upper = vec[i + half];
                        vec[i] = lower * c[i] - upper * si[i];
                        vec[i + half] = upper * c[i + half] + lower * si[i + half];
                    }
                }
            }
    }

    /// <summary>Copies a contiguous last-dim slice: <c>out[row, d] = in[row, offset + d]</c> for
    /// <c>d in [0, outDim)</c>, where <c>outDim</c> is the output's last dim and the input's last dim
    /// is the row stride. Splits a fused tensor (e.g. QKV <c>[B,L,3H]</c>) into a contiguous chunk.</summary>
    unsafe void SliceLastDim(Tensor output, Tensor input, int offset)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32)
            throw new NotSupportedException("SliceLastDim default fallback only supports F32.");
        // outDim is the slice width over the row, derived from row count (not the output's last dim,
        // which may be split into [.., heads, headDim]). rows = input rows = output rows.
        int inDim = (int)input.Shape[input.Shape.Rank - 1];
        long rows = input.ElementCount / inDim;
        int outDim = (int)(output.ElementCount / rows);
        long total = output.ElementCount;
        float* pOut = (float*)output.DataPointer;
        float* pIn = (float*)input.DataPointer;
        for (long i = 0; i < total; i++)
        {
            int d = (int)(i % outDim);
            long row = i / outDim;
            pOut[i] = pIn[row * inDim + offset + d];
        }
    }

    /// <summary>Extracts a contiguous TIME-dimension sub-range from a KV-shaped tensor:
    /// <c>output[1, h, t, d] = input[1, h, start + t, d]</c> for <c>t in [0, len)</c>. Unlike
    /// <see cref="SliceLastDim"/> (which slices the last/head-dim axis), this slices dim 2 (the sequence/time
    /// axis) — used by paged KV caches to (a) split a multi-token append across page boundaries and (b) gather
    /// a partially-filled page's occupied prefix before copying it into a contiguous scratch buffer for
    /// attention. <paramref name="input"/> is <c>[1, H, Tin, D]</c>, <paramref name="output"/> is
    /// <c>[1, H, len, D]</c>.</summary>
    unsafe void SliceTimeRange(Tensor output, Tensor input, int start, int len)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32)
            throw new NotSupportedException("SliceTimeRange default fallback only supports F32.");
        int h = (int)input.Shape[1];
        int tIn = (int)input.Shape[2];
        int d = (int)input.Shape[3];
        float* pOut = (float*)output.DataPointer;
        float* pIn = (float*)input.DataPointer;
        for (int hi = 0; hi < h; hi++)
        {
            long srcBase = ((long)hi * tIn + start) * d;
            long dstBase = (long)hi * len * d;
            Buffer.MemoryCopy(pIn + srcBase, pOut + dstBase, (long)len * d * 4, (long)len * d * 4);
        }
    }

    /// <summary>Per-row scalar multiply: <c>out[row, c] = in[row, c] * rowMask[row]</c>. <paramref name="rowMask"/>
    /// has one value per row (rows = input element count / last dim). Token masking for DiT embeddings.</summary>
    unsafe void MaskRows(Tensor output, Tensor input, Tensor rowMask)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32 || rowMask.DType != DType.F32)
            throw new NotSupportedException("MaskRows default fallback only supports F32.");
        int channels = (int)input.Shape[input.Shape.Rank - 1];
        long total = input.ElementCount;
        float* pOut = (float*)output.DataPointer;
        float* pIn = (float*)input.DataPointer;
        float* pMask = (float*)rowMask.DataPointer;
        for (long i = 0; i < total; i++)
            pOut[i] = pIn[i] * pMask[i / channels];
    }

    /// <summary>Element-wise add scalar: <c>out[i] = in[i] + c</c>.</summary>
    unsafe void AddScalar(Tensor output, Tensor input, float scalar)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32)
            throw new NotSupportedException("AddScalar default fallback only supports F32.");
        long n = input.ElementCount;
        float* pOut = (float*)output.DataPointer;
        float* pIn = (float*)input.DataPointer;
        for (long i = 0; i < n; i++) pOut[i] = pIn[i] + scalar;
    }

    /// <summary>Non-affine LayerNorm over the last dim: per row, zero mean and unit variance (biased var),
    /// no learned scale/bias.</summary>
    unsafe void LayerNormNoAffine(Tensor output, Tensor input, float eps)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32)
            throw new NotSupportedException("LayerNormNoAffine default fallback only supports F32.");
        int dim = (int)input.Shape[input.Shape.Rank - 1];
        long rows = input.ElementCount / dim;
        float* pIn = (float*)input.DataPointer;
        float* pOut = (float*)output.DataPointer;
        for (long r = 0; r < rows; r++)
        {
            long off = r * dim;
            float mean = 0f;
            for (int d = 0; d < dim; d++) mean += pIn[off + d];
            mean /= dim;
            float var = 0f;
            for (int d = 0; d < dim; d++) { float diff = pIn[off + d] - mean; var += diff * diff; }
            var /= dim;
            float invStd = 1.0f / MathF.Sqrt(var + eps);
            for (int d = 0; d < dim; d++) pOut[off + d] = (pIn[off + d] - mean) * invStd;
        }
    }

    /// <summary>In-place index-add of embedding rows: <c>h[row, d] += table[indices[row], d]</c>.
    /// <paramref name="indices"/> is an I32 tensor of length rows; <paramref name="table"/> is <c>[numRows, dim]</c>.
    /// Keeps the indicator embedding GPU-resident (no host gather of the weight).</summary>
    unsafe void IndexAddRows(Tensor h, Tensor table, Tensor indices)
    {
        if (h.DType != DType.F32 || table.DType != DType.F32)
            throw new NotSupportedException("IndexAddRows default fallback only supports F32.");
        if (indices.DType != DType.I32)
            throw new NotSupportedException("IndexAddRows requires I32 indices.");
        int dim = (int)h.Shape[h.Shape.Rank - 1];
        long rows = h.ElementCount / dim;
        float* pH = (float*)h.DataPointer;
        float* pTable = (float*)table.DataPointer;
        int* pIdx = (int*)indices.DataPointer;
        for (long r = 0; r < rows; r++)
        {
            long off = r * dim;
            long tOff = (long)pIdx[r] * dim;
            for (int d = 0; d < dim; d++) pH[off + d] += pTable[tOff + d];
        }
    }

    /// <summary>Per-row argmax over the last dim: <c>indices[r] = argmax_c input[r, c]</c> (first max on ties).
    /// <paramref name="input"/> is <c>[.., C]</c> (rows = product of the leading dims); <paramref name="indices"/>
    /// is an I32 tensor of length rows. Keeps greedy sampling GPU-resident: the winning token is reduced on-device
    /// so only the row's index (one int), not the full C-wide logit vector, needs to cross to the host.</summary>
    unsafe void ArgMaxLastDim(Tensor indices, Tensor input)
    {
        if (input.DType != DType.F32)
            throw new NotSupportedException("ArgMaxLastDim default fallback only supports F32.");
        if (indices.DType != DType.I32)
            throw new NotSupportedException("ArgMaxLastDim requires I32 indices.");
        int c = (int)input.Shape[input.Shape.Rank - 1];
        long rows = input.ElementCount / c;
        if (indices.ElementCount < rows)
            throw new ArgumentException($"ArgMaxLastDim indices length {indices.ElementCount} < rows {rows}.");
        float* pIn = (float*)input.DataPointer;
        int* pIdx = (int*)indices.DataPointer;
        for (long r = 0; r < rows; r++)
        {
            float* row = pIn + r * c;
            int best = 0; float bv = row[0];
            for (int i = 1; i < c; i++) if (row[i] > bv) { bv = row[i]; best = i; }
            pIdx[r] = best;
        }
    }

    /// <summary>Scatter rows after a zeroed head block: <c>output = [zeros(headRows), input]</c> along the row
    /// axis. Builds the conditional latent (text rows zeroed, image rows = current latent).</summary>
    unsafe void ScatterRowsAfter(Tensor output, Tensor input, int headRows)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32)
            throw new NotSupportedException("ScatterRowsAfter default fallback only supports F32.");
        int dim = (int)input.Shape[input.Shape.Rank - 1];
        long total = output.ElementCount;
        long headElems = (long)headRows * dim;
        float* pOut = (float*)output.DataPointer;
        float* pIn = (float*)input.DataPointer;
        for (long i = 0; i < total; i++)
            pOut[i] = i < headElems ? 0f : pIn[i - headElems];
    }

    /// <summary>Contiguous row-block slice: copies <c>output.ElementCount</c> elements starting at row
    /// <paramref name="rowOffset"/> of <paramref name="input"/> into <paramref name="output"/>.</summary>
    unsafe void SliceRows(Tensor output, Tensor input, int rowOffset)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32)
            throw new NotSupportedException("SliceRows default fallback only supports F32.");
        int dim = (int)output.Shape[output.Shape.Rank - 1];
        long elemOffset = (long)rowOffset * dim;
        long total = output.ElementCount;
        float* pOut = (float*)output.DataPointer;
        float* pIn = (float*)input.DataPointer;
        for (long i = 0; i < total; i++) pOut[i] = pIn[elemOffset + i];
    }

    /// <summary>Fused quantized matmul: <c>output[M,N] = input[M,K] @ quantWeight[N,K]^T (+ bias[N])</c>, where
    /// <paramref name="quantWeight"/> is a GGUF-quantized weight (Q8_0/Q4_K/Q5_K/Q6_K). The weight stays in its
    /// compressed form — dequantized block-by-block inside the kernel and never materialized as F16/F32 — so
    /// big models keep their on-device footprint at the quant size. <paramref name="input"/>/<paramref name="output"/>
    /// are F32. This is the true low-VRAM path; the CUDA backend overrides it. The default throws (CPU/Vulkan
    /// have no in-kernel dequant; callers must pre-dequantize the weight for those backends).</summary>
    void QuantizedMatMul(Tensor output, Tensor input, Tensor quantWeight, Tensor? bias)
        => throw new NotSupportedException(
            "QuantizedMatMul is implemented on the CUDA backend only; dequantize the weight to F16/F32 for CPU/Vulkan.");

    // ── Attention ───────────────────────────────────────────────────────

    /// <summary>Scaled dot-product attention: output = softmax(Q @ K^T / sqrt(d)) @ V</summary>
    /// <param name="allowF16">When true, a backend MAY run the attention in F16 for speed (halves score-matrix
    /// traffic + F16 tensor cores). Callers pass true ONLY when pre-softmax scores are bounded — e.g. Q/K are
    /// RMS/L2-normalized (Wan) — since unbounded scores (some fp8 archs) overflow F16 and produce black output.</param>
    void ScaledDotProductAttention(Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask, float scale, bool allowF16 = false);

    /// <summary>FlashAttention: fused online-softmax attention that never materializes the score matrix and is
    /// grouped-query aware (no need to replicate K/V). <paramref name="query"/> is <c>[B, Hq, Tq, D]</c>;
    /// <paramref name="key"/>/<paramref name="value"/> are <c>[B, Hkv, Lk, D]</c>; <paramref name="output"/> is
    /// <c>[B, Hq, Tq, D]</c>. Query head <c>h</c> reads kv head <c>h / kvGroup</c>. When <paramref name="causal"/>,
    /// query row <c>r</c> (absolute position <c>qOffset + r</c>) attends keys <c>0 .. qOffset+r</c>; otherwise all
    /// <c>Lk</c> keys. When <paramref name="softcap"/> &gt; 0, each logit is soft-capped via
    /// <c>softcap·tanh(score/softcap)</c> before the softmax (Gemma-2 attention-logit soft-cap); 0 disables it.
    /// When <paramref name="sink"/> is non-null (a <c>[Hq]</c> F32 tensor, GPT-OSS attention sinks), each head's
    /// sink logit joins the softmax denominator but contributes no value.
    /// When <paramref name="slidingWindow"/> &gt; 0, each query additionally attends only the most recent
    /// <c>slidingWindow</c> keys (Gemma-2/3, Cohere2, GPT-OSS local layers); 0 = full causal prefix.
    /// When <paramref name="alibiSlopes"/> is non-null (a <c>[Hq]</c> F32 tensor), each head adds the ALiBi linear
    /// distance penalty <c>slope_h·(k_pos − q_pos)</c> to its scores (MPT/BLOOM/Falcon-classic/Jais); these models
    /// use no RoPE.
    /// The CUDA backend overrides this with a kernel; the default is a correct CPU reference.</summary>
    unsafe void FlashAttention(Tensor output, Tensor query, Tensor key, Tensor value, int kvLen, int kvGroup, bool causal, int qOffset, float scale, float softcap = 0f, Tensor? sink = null, int slidingWindow = 0, Tensor? alibiSlopes = null)
        => AttentionReference.FlashAttention(output, query, key, value, kvLen, kvGroup, causal, qOffset, scale, softcap, sink, slidingWindow, alibiSlopes);

    /// <summary>Gathers rows: <c>output[m] = input[rowIndices[m]]</c>, where a "row" is the last-dim-sized
    /// vector. <paramref name="output"/> is <c>[M, K]</c>, <paramref name="input"/> <c>[N, K]</c>,
    /// <paramref name="rowIndices"/> length M. Used for MoE expert dispatch (collect the tokens routed to one
    /// expert into a contiguous group). The CUDA backend overrides this; the default is a host gather.</summary>
    unsafe void GatherRows(Tensor output, Tensor input, ReadOnlySpan<int> rowIndices)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32)
            throw new NotSupportedException("GatherRows default fallback only supports F32.");
        int k = (int)input.Shape[input.Shape.Rank - 1];
        int m = rowIndices.Length;
        float* pOut = (float*)output.DataPointer;
        float* pIn = (float*)input.DataPointer;
        for (int row = 0; row < m; row++)
        {
            long src = (long)rowIndices[row] * k;
            long dst = (long)row * k;
            Buffer.MemoryCopy(pIn + src, pOut + dst, (long)k * 4, (long)k * 4);
        }
    }

    /// <summary>Scatter-add with per-row scale: <c>output[rowIndices[m]] += scales[m] * input[m]</c>.
    /// <paramref name="input"/> is <c>[M, K]</c>, <paramref name="output"/> <c>[N, K]</c> (must be pre-zeroed /
    /// already accumulating), <paramref name="rowIndices"/> + <paramref name="scales"/> length M. Used to combine
    /// each MoE expert's weighted contribution back into the per-token output. Within one call the indices are
    /// distinct; calling once per expert into the same output accumulates the mixture. CUDA backend overrides.</summary>
    unsafe void ScatterAddWeightedRows(Tensor output, Tensor input, ReadOnlySpan<int> rowIndices, ReadOnlySpan<float> scales)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32)
            throw new NotSupportedException("ScatterAddWeightedRows default fallback only supports F32.");
        int k = (int)input.Shape[input.Shape.Rank - 1];
        int m = rowIndices.Length;
        float* pOut = (float*)output.DataPointer;
        float* pIn = (float*)input.DataPointer;
        for (int row = 0; row < m; row++)
        {
            float s = scales[row];
            long dst = (long)rowIndices[row] * k;
            long src = (long)row * k;
            for (int j = 0; j < k; j++) pOut[dst + j] += s * pIn[src + j];
        }
    }

    /// <summary>In-place KV-cache append: writes <paramref name="newKv"/> <c>[1, H, tNew, D]</c> into the
    /// fixed-capacity <paramref name="buffer"/> <c>[1, H, maxSeq, D]</c> starting at sequence position
    /// <paramref name="offset"/> (per head). Lets a KV cache grow without reallocating (O(tNew)/step vs the
    /// Concat-grown cache's O(n²)). The CUDA backend overrides this; the default is a host copy.</summary>
    unsafe void KvCacheAppend(Tensor buffer, Tensor newKv, int offset)
    {
        if (buffer.DType != DType.F32 || newKv.DType != DType.F32)
            throw new NotSupportedException("KvCacheAppend default fallback only supports F32.");
        int h = (int)buffer.Shape[1];
        int maxSeq = (int)buffer.Shape[2];
        int d = (int)buffer.Shape[3];
        int tNew = (int)newKv.Shape[2];
        float* bp = (float*)buffer.DataPointer;
        float* np = (float*)newKv.DataPointer;
        for (int hi = 0; hi < h; hi++)
        {
            long srcBase = (long)hi * tNew * d;
            long dstBase = ((long)hi * maxSeq + offset) * d;
            Buffer.MemoryCopy(np + srcBase, bp + dstBase, (long)tNew * d * 4, (long)tNew * d * 4);
        }
    }

    // ── Activations ─────────────────────────────────────────────────────

    /// <summary>GELU activation (exact).</summary>
    void Gelu(Tensor output, Tensor input);

    /// <summary>SiLU activation (x * sigmoid(x)).</summary>
    void Silu(Tensor output, Tensor input);

    /// <summary>Sigmoid activation (1 / (1 + exp(-x))). Used by LSTM gating.</summary>
    void Sigmoid(Tensor output, Tensor input);

    /// <summary>Hyperbolic tangent activation. Used by LSTM cell update / output and
    /// several vocoders (Mish, snake-bias).</summary>
    void Tanh(Tensor output, Tensor input);

    /// <summary>ELU activation: <c>x if x &gt;= 0 else alpha * (exp(x) - 1)</c>. Used by
    /// the SEANet residual blocks in EnCodec and DAC — alpha is typically 1.0.</summary>
    void Elu(Tensor output, Tensor input, float alpha);

    /// <summary>Leaky ReLU: <c>x if x &gt;= 0 else slope * x</c>. Kokoro / StyleTTS 2's
    /// text encoder + decoder use slope=0.2; HiFi-GAN MRF blocks use the same.</summary>
    void LeakyRelu(Tensor output, Tensor input, float slope);

    /// <summary>Snake activation: <c>x + (sin(alpha * x))^2 / divisor</c>, where
    /// <c>divisor = alpha</c> when <paramref name="beta"/> is null (vanilla snake from
    /// the Stable Audio Oobleck VAE), and <c>divisor = beta + 1e-8</c> when supplied
    /// (snake-beta variant from BigVGAN). <paramref name="alpha"/> and
    /// <paramref name="beta"/> are per-channel learnable params of shape <c>[1, C, 1]</c>
    /// (broadcast across batch and time for <c>[B, C, T]</c> input).</summary>
    void Snake(Tensor output, Tensor input, Tensor alpha, Tensor? beta);

    /// <summary>Parametric ReLU over <c>[B, C, T]</c>: <c>x if x&gt;=0 else alpha*x</c>, with
    /// <paramref name="alpha"/> per-channel (<c>ElementCount == C</c>) or a single shared value.
    /// Default host implementation; CUDA overrides with a kernel to keep the codec chain GPU-resident.</summary>
    unsafe void Prelu(Tensor output, Tensor input, Tensor alpha)
    {
        int batch = (int)input.Shape[0], channels = (int)input.Shape[1], timeDim = (int)input.Shape[2];
        float* op = (float*)output.DataPointer, ip = (float*)input.DataPointer, ap = (float*)alpha.DataPointer;
        bool perCh = alpha.ElementCount > 1;
        for (int bi = 0; bi < batch; bi++)
            for (int ci = 0; ci < channels; ci++)
            {
                float a = perCh ? ap[ci] : ap[0];
                long off = ((long)bi * channels + ci) * timeDim;
                for (int ti = 0; ti < timeDim; ti++) { float v = ip[off + ti]; op[off + ti] = v >= 0f ? v : a * v; }
            }
    }

    /// <summary>Frame-repeat over time on <c>[B, C, inT]</c>: each time-step is duplicated
    /// <paramref name="numSamples"/> times consecutively → <c>[B, C, inT*numSamples]</c> (nearest 1D upsample).
    /// Default host implementation; CUDA overrides with a kernel to keep the codec chain GPU-resident.</summary>
    unsafe void RepeatTime(Tensor output, Tensor input, int numSamples)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32)
            throw new NotSupportedException("RepeatTime default fallback only supports F32.");
        int batch = (int)input.Shape[0], channels = (int)input.Shape[1], inT = (int)input.Shape[2];
        int outT = inT * numSamples;
        float* ip = (float*)input.DataPointer, op = (float*)output.DataPointer;
        for (int bc = 0; bc < batch * channels; bc++)
        {
            long inBase = (long)bc * inT, outBase = (long)bc * outT;
            for (int ti = 0; ti < inT; ti++)
            {
                float v = ip[inBase + ti];
                for (int s = 0; s < numSamples; s++) op[outBase + (long)ti * numSamples + s] = v;
            }
        }
    }

    // ── Element-wise ────────────────────────────────────────────────────

    /// <summary>Element-wise addition: output = a + b</summary>
    void Add(Tensor output, Tensor a, Tensor b);

    /// <summary>Element-wise multiplication: output = a * b</summary>
    void Mul(Tensor output, Tensor a, Tensor b);

    /// <summary>Scalar multiplication: output = input * scalar</summary>
    void Scale(Tensor output, Tensor input, float scalar);

    /// <summary>Element-wise clamp: output = clamp(input, min, max)</summary>
    void Clamp(Tensor output, Tensor input, float min, float max);

    // ── Transpose / Permute ─────────────────────────────────────────────

    /// <summary>Batched 2D transpose: [B, D1, D2] → [B, D2, D1].</summary>
    void Transpose2D(Tensor output, Tensor input, int d1, int d2);

    /// <summary>4D permute swapping dims 1 and 2: [B, S, H, D] → [B, H, S, D].</summary>
    void Permute0213(Tensor output, Tensor input, int s, int h, int d);

    /// <summary>Grouped-query attention K/V head repeat. Expands <paramref name="input"/> <c>[B, Hkv, L, D]</c>
    /// to <paramref name="output"/> <c>[B, Hkv*groupSize, L, D]</c> using the block pattern (output head
    /// <c>h*groupSize + g</c> = input head <c>h</c>), matching HF <c>repeat_kv</c>. Keeps the tensor
    /// GPU-resident on backends that override this; the default fallback is a host copy.</summary>
    unsafe void RepeatKvHeads(Tensor output, Tensor input, int kvHeads, int groupSize)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32)
            throw new NotSupportedException("RepeatKvHeads default fallback only supports F32.");
        int batch = (int)input.Shape[0];
        int seqLen = (int)input.Shape[2];
        int headDim = (int)input.Shape[3];
        long perHead = (long)seqLen * headDim;
        float* ip = (float*)input.DataPointer;
        float* op = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < kvHeads; h++)
            {
                long srcOff = ((long)b * kvHeads + h) * perHead;
                for (int g = 0; g < groupSize; g++)
                {
                    int qHead = h * groupSize + g;
                    long dstOff = ((long)b * (kvHeads * groupSize) + qHead) * perHead;
                    Buffer.MemoryCopy(ip + srcOff, op + dstOff, perHead * 4, perHead * 4);
                }
            }
    }

    /// <summary>GEGLU activation: splits input in half along last dim, applies GELU gate. Output has half the elements of input.</summary>
    void GeGlu(Tensor output, Tensor input);

    /// <summary>Broadcast add: hidden [B, C, ...spatial] += bias [B, C] in-place.</summary>
    void BroadcastAdd(Tensor hidden, Tensor bias, int channels, int spatial);

    // ── Shape Operations ────────────────────────────────────────────────

    /// <summary>Concatenate tensors along the specified dimension.</summary>
    void Concat(Tensor output, ReadOnlySpan<Tensor> inputs, int dim);

    /// <summary>Split a tensor into chunks along the specified dimension.</summary>
    void Split(ReadOnlySpan<Tensor> outputs, Tensor input, int dim);

    // ── Convolution ─────────────────────────────────────────────────────

    /// <summary>1D convolution with explicit asymmetric padding. Inputs and outputs are
    /// channels-first: <paramref name="input"/> <c>[B, C_in, T_in]</c>,
    /// <paramref name="output"/> <c>[B, C_out, T_out]</c> (pre-allocated by the caller).
    /// <paramref name="weight"/> follows PyTorch convention <c>[C_out, C_in / groups, K]</c>.
    /// Pass <paramref name="padLeft"/>/<paramref name="padRight"/> separately so the same
    /// op covers both causal (left-only) and symmetric padding; pass
    /// <paramref name="groups"/> equal to channels for depthwise mode.</summary>
    void Conv1d(Tensor output, Tensor input, Tensor weight, Tensor? bias,
        int stride, int padLeft, int padRight, int dilation, int groups);

    /// <summary>1D transposed convolution. Input/output channels-first;
    /// <paramref name="weight"/> follows PyTorch convention <c>[C_in, C_out / groups, K]</c>.
    /// Output length is <c>(T_in - 1) * stride + dilation * (K - 1) + 1 - padLeft - padRight</c>.
    /// For VibeVoice / EnCodec causal decoders pass <c>padLeft = 0</c>,
    /// <c>padRight = K - stride</c> to remove all trailing pad (matches
    /// <c>trim_right_ratio = 1.0</c>). Pass <paramref name="groups"/> equal to channels for
    /// depthwise mode (e.g. BigVGAN anti-aliased upsampling with a shared lowpass filter).</summary>
    void ConvTranspose1d(Tensor output, Tensor input, Tensor weight, Tensor? bias,
        int stride, int padLeft, int padRight, int dilation, int groups);

    // ── Sampling ────────────────────────────────────────────────────────

    /// <summary>Nearest-neighbor 2D upsample by the given scale factor.</summary>
    void UpsampleNearest2D(Tensor output, Tensor input, int scaleH, int scaleW);

    /// <summary>Bilinear 2D upsample by the given scale factor.</summary>
    void UpsampleBilinear2D(Tensor output, Tensor input, int scaleH, int scaleW);

    /// <summary>2D transposed convolution. Used by YOLO seg's Proto module for upsampling mask
    /// prototypes (k=2, s=2, p=0 doubles spatial dims). Weight shape is <c>[C_in, C_out, kH, kW]</c>
    /// — PyTorch convention, note input channels come first (opposite of standard Conv2d).
    /// Default implementation is a CPU scatter-add loop over F32 NCHW tensors; backends should
    /// override for performance.</summary>
    unsafe void ConvTranspose2d(Tensor output, Tensor input, Tensor weight, Tensor? bias,
        int strideH, int strideW, int padH, int padW)
    {
        if (input.DType != DType.F32 || output.DType != DType.F32 || weight.DType != DType.F32)
            throw new NotSupportedException($"ConvTranspose2d default fallback only supports F32 — got input={input.DType}, output={output.DType}, weight={weight.DType}.");
        if (input.Shape.Rank != 4 || output.Shape.Rank != 4 || weight.Shape.Rank != 4)
            throw new ArgumentException($"ConvTranspose2d requires 4D tensors; got input {input.Shape}, output {output.Shape}, weight {weight.Shape}.");

        int n = (int)input.Shape[0];
        int cIn = (int)input.Shape[1];
        int iH = (int)input.Shape[2];
        int iW = (int)input.Shape[3];
        int cOut = (int)output.Shape[1];
        int oH = (int)output.Shape[2];
        int oW = (int)output.Shape[3];
        int kH = (int)weight.Shape[2];
        int kW = (int)weight.Shape[3];
        if (weight.Shape[0] != cIn || weight.Shape[1] != cOut)
            throw new ArgumentException($"ConvTranspose2d weight shape [{weight.Shape[0]}, {weight.Shape[1]}, ...] must equal [C_in={cIn}, C_out={cOut}, ...].");

        float* srcBase = (float*)input.DataPointer;
        float* dstBase = (float*)output.DataPointer;
        float* wBase = (float*)weight.DataPointer;
        float* bBase = bias is null ? null : (float*)bias.DataPointer;

        // Initialize output to bias (so the scatter-add can accumulate on top).
        for (int b = 0; b < n; b++)
        {
            for (int co = 0; co < cOut; co++)
            {
                float biasVal = bBase is null ? 0f : bBase[co];
                float* plane = dstBase + ((long)b * cOut + co) * oH * oW;
                for (long i = 0; i < (long)oH * oW; i++) plane[i] = biasVal;
            }
        }

        // Scatter-add: each input pixel (ci, yi, xi) contributes weight[ci, co, ky, kx] * value
        // to every output position (yi*sH+ky-pH, xi*sW+kx-pW) for each (co, ky, kx).
        for (int b = 0; b < n; b++)
        {
            for (int ci = 0; ci < cIn; ci++)
            {
                float* srcPlane = srcBase + ((long)b * cIn + ci) * iH * iW;
                for (int co = 0; co < cOut; co++)
                {
                    float* dstPlane = dstBase + ((long)b * cOut + co) * oH * oW;
                    long wOff = ((long)ci * cOut + co) * kH * kW;
                    for (int yi = 0; yi < iH; yi++)
                    {
                        for (int xi = 0; xi < iW; xi++)
                        {
                            float v = srcPlane[yi * iW + xi];
                            if (v == 0f) continue;
                            for (int ky = 0; ky < kH; ky++)
                            {
                                int yo = yi * strideH + ky - padH;
                                if (yo < 0 || yo >= oH) continue;
                                for (int kx = 0; kx < kW; kx++)
                                {
                                    int xo = xi * strideW + kx - padW;
                                    if (xo < 0 || xo >= oW) continue;
                                    dstPlane[yo * oW + xo] += wBase[wOff + ky * kW + kx] * v;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// <summary>Depthwise 2D convolution — each output channel sees exactly one input channel
    /// (groups = C). Used by YOLO11's class branch and the C2PSA positional encoding. Weight
    /// shape is <c>[C, 1, kH, kW]</c> and bias <c>[C]</c>. Default implementation is a CPU loop
    /// over F32 NCHW tensors; backends should override for performance once a depthwise
    /// kernel is worth shipping.</summary>
    unsafe void Conv2dDepthwise(Tensor output, Tensor input, Tensor weight, Tensor? bias,
        int strideH, int strideW, int padH, int padW)
    {
        if (input.DType != DType.F32 || output.DType != DType.F32 || weight.DType != DType.F32)
            throw new NotSupportedException($"Conv2dDepthwise default fallback only supports F32 — got input={input.DType}, output={output.DType}, weight={weight.DType}.");
        if (input.Shape.Rank != 4 || output.Shape.Rank != 4)
            throw new ArgumentException($"Conv2dDepthwise requires [N, C, H, W] tensors; got input {input.Shape} / output {output.Shape}.");
        if (weight.Shape.Rank != 4 || weight.Shape[1] != 1)
            throw new ArgumentException($"Conv2dDepthwise weight must be [C, 1, kH, kW]; got {weight.Shape}.");
        if (input.Shape[1] != weight.Shape[0] || output.Shape[1] != weight.Shape[0])
            throw new ArgumentException("Conv2dDepthwise requires input/output channel count to equal weight channel count.");

        int n = (int)input.Shape[0];
        int c = (int)input.Shape[1];
        int iH = (int)input.Shape[2];
        int iW = (int)input.Shape[3];
        int oH = (int)output.Shape[2];
        int oW = (int)output.Shape[3];
        int kH = (int)weight.Shape[2];
        int kW = (int)weight.Shape[3];

        float* srcBase = (float*)input.DataPointer;
        float* dstBase = (float*)output.DataPointer;
        float* wBase = (float*)weight.DataPointer;
        float* bBase = bias is null ? null : (float*)bias.DataPointer;

        for (int b = 0; b < n; b++)
        {
            for (int ch = 0; ch < c; ch++)
            {
                float* srcPlane = srcBase + ((long)b * c + ch) * iH * iW;
                float* dstPlane = dstBase + ((long)b * c + ch) * oH * oW;
                float* kernel = wBase + (long)ch * kH * kW;
                float biasVal = bBase is null ? 0f : bBase[ch];

                for (int oy = 0; oy < oH; oy++)
                {
                    int iy0 = oy * strideH - padH;
                    for (int ox = 0; ox < oW; ox++)
                    {
                        int ix0 = ox * strideW - padW;
                        float v = biasVal;
                        for (int ky = 0; ky < kH; ky++)
                        {
                            int iy = iy0 + ky;
                            if (iy < 0 || iy >= iH) continue;
                            for (int kx = 0; kx < kW; kx++)
                            {
                                int ix = ix0 + kx;
                                if (ix < 0 || ix >= iW) continue;
                                v += kernel[ky * kW + kx] * srcPlane[iy * iW + ix];
                            }
                        }
                        dstPlane[oy * oW + ox] = v;
                    }
                }
            }
        }
    }

    /// <summary>2D max-pooling with explicit kernel, stride, and zero-padding. Used by YOLO's
    /// SPPF block (k=5, s=1, p=2 — preserves spatial dims). Default implementation is a CPU loop
    /// over F32 NCHW tensors; backends should override for performance.</summary>
    unsafe void MaxPool2D(Tensor output, Tensor input, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW)
    {
        if (input.DType != DType.F32 || output.DType != DType.F32)
            throw new NotSupportedException($"MaxPool2D default fallback only supports F32 — got input={input.DType}, output={output.DType}. Override in the backend if you need other dtypes.");
        if (input.Shape.Rank != 4 || output.Shape.Rank != 4)
            throw new ArgumentException($"MaxPool2D requires [N, C, H, W] tensors; got input {input.Shape} / output {output.Shape}.");

        int n = (int)input.Shape[0];
        int c = (int)input.Shape[1];
        int iH = (int)input.Shape[2];
        int iW = (int)input.Shape[3];
        int oH = (int)output.Shape[2];
        int oW = (int)output.Shape[3];

        float* srcBase = (float*)input.DataPointer;
        float* dstBase = (float*)output.DataPointer;

        // NCHW: outer indices [n, c] address a contiguous H*W plane.
        for (int b = 0; b < n; b++)
        {
            for (int ch = 0; ch < c; ch++)
            {
                long planeOffset = ((long)b * c + ch) * iH * iW;
                long outPlaneOffset = ((long)b * c + ch) * oH * oW;
                float* srcPlane = srcBase + planeOffset;
                float* dstPlane = dstBase + outPlaneOffset;

                for (int oy = 0; oy < oH; oy++)
                {
                    int iy0 = oy * strideH - padH;
                    for (int ox = 0; ox < oW; ox++)
                    {
                        int ix0 = ox * strideW - padW;
                        float maxVal = float.NegativeInfinity;
                        for (int ky = 0; ky < kernelH; ky++)
                        {
                            int iy = iy0 + ky;
                            if (iy < 0 || iy >= iH) continue;
                            for (int kx = 0; kx < kernelW; kx++)
                            {
                                int ix = ix0 + kx;
                                if (ix < 0 || ix >= iW) continue;
                                float v = srcPlane[iy * iW + ix];
                                if (v > maxVal) maxVal = v;
                            }
                        }
                        // If the entire receptive field was out-of-bounds (impossible for k=5,s=1,p=2
                        // but defensible for arbitrary configs), fall back to 0 instead of -inf to
                        // avoid poisoning downstream layers. Won't trigger in any YOLO config.
                        dstPlane[oy * oW + ox] = float.IsNegativeInfinity(maxVal) ? 0f : maxVal;
                    }
                }
            }
        }
    }

    // ── Data Movement ───────────────────────────────────────────────────

    /// <summary>Copy tensor data to a different device.</summary>
    void CopyTo(Tensor destination, Tensor source);

    /// <summary>Fill a tensor with a constant value.</summary>
    void Fill(Tensor tensor, float value);

    // ── Audio (optional — backends may throw NotSupportedException) ──

    /// <summary>Radix-2 FFT for audio processing.</summary>
    void Fft(Tensor output, Tensor input);

    /// <summary>Short-time Fourier transform.</summary>
    void Stft(Tensor output, Tensor input, int fftSize, int hopLength, Tensor window);

    /// <summary>Apply mel filterbank to FFT magnitude spectrogram.</summary>
    void MelFilterbank(Tensor output, Tensor input, Tensor filters);

    // ── Fused Operations ────────────────────────────────────────────────

    /// <summary>Fused GroupNorm + SiLU: normalize, apply affine, then SiLU in one pass. Eliminates intermediate allocation. Default falls back to separate GroupNorm + Silu.</summary>
    void GroupNormSilu(Tensor output, Tensor input, Tensor weight, Tensor bias, int groups, float eps)
    {
        GroupNorm(output, input, weight, bias, groups, eps);
        Silu(output, output);
    }

    // ── Dtype Casting ────────────────────────────────────────────────────

    /// <summary>Cast tensor from FP32 to FP16. Default: CPU scalar loop.</summary>
    unsafe void CastToF16(Tensor output, Tensor input)
    {
        float* src = (float*)input.DataPointer;
        Half* dst = (Half*)output.DataPointer;
        int count = (int)input.Shape.ElementCount;
        for (int i = 0; i < count; i++)
            dst[i] = (Half)src[i];
    }

    /// <summary>Cast tensor from FP16 to FP32. Default: CPU scalar loop.</summary>
    unsafe void CastToF32(Tensor output, Tensor input)
    {
        Half* src = (Half*)input.DataPointer;
        float* dst = (float*)output.DataPointer;
        int count = (int)input.Shape.ElementCount;
        for (int i = 0; i < count; i++)
            dst[i] = (float)src[i];
    }

    /// <summary>Cast tensor from FP32 to BF16. Default: CPU fallback via Tensor.CastTo.</summary>
    void CastToBf16(Tensor output, Tensor input)
    {
        Tensor casted = input.CastTo(DType.BF16);
        try
        {
            unsafe
            {
                Buffer.MemoryCopy((void*)casted.DataPointer, (void*)output.DataPointer,
                    output.ElementCount * 2, casted.ElementCount * 2);
            }
        }
        finally { casted.Dispose(); }
    }

    /// <summary>Cast tensor from FP8 E4M3 to FP16. Default: CPU via F32 intermediate.</summary>
    void CastF8E4M3ToF16(Tensor output, Tensor input)
    {
        Tensor f32 = input.CastTo(DType.F32);
        CastToF16(output, f32);
        f32.Dispose();
    }

    /// <summary>Cast tensor from FP16 to FP8 E4M3. Default: CPU via Tensor.CastTo.</summary>
    void CastF16ToF8E4M3(Tensor output, Tensor input)
    {
        Tensor f8 = input.CastTo(DType.F8E4M3);
        unsafe
        {
            long byteCount = output.Shape.ElementCount; // 1 byte per F8 element
            Buffer.MemoryCopy(f8.DataPointer, output.DataPointer, byteCount, byteCount);
        }
        f8.Dispose();
    }

    // ── Synchronization ──────────────────────────────────────────────────

    /// <summary>Waits for all pending GPU work to complete. No-op on CPU backends. Call at pipeline phase boundaries to ensure deferred memory frees are processed before large allocations.</summary>
    void Sync() { }

    /// <summary>Frees specific weight tensors from accelerator memory. No-op on CPU backends. Call between pipeline phases to reclaim VRAM (e.g., free UNet weights before VAE decode).</summary>
    void FreeWeights(IEnumerable<Tensor> weights) { }

    /// <summary>Frees cached activation device buffers while keeping preloaded weights. Backends that keep
    /// activations GPU-resident (CUDA) override this so long multi-step pipelines can reclaim memory between steps;
    /// the default is a no-op.</summary>
    void FreeActivations() { }

    /// <summary>As <see cref="FreeActivations()"/>, but with control over the memory-pool trim. Pass
    /// <paramref name="trimPool"/>=false on hot per-step/per-tile cleanups: the freed blocks stay reserved in the
    /// stream-ordered pool for immediate reuse, avoiding a multi-GB driver release + re-map every iteration
    /// (persistent allocators reclaim the pool automatically via their OOM-retry if they ever need it). Use
    /// <paramref name="trimPool"/>=true (or <see cref="TrimMemoryPool"/>) at pipeline stage transitions.</summary>
    void FreeActivations(bool trimPool) => FreeActivations();

    /// <summary>Returns pool-reserved-but-free device memory to the driver WITHOUT clearing the activation cache, so
    /// it is safe to call mid-computation (e.g. between VAE decode tiles) to cap peak memory. No-op on backends
    /// without a stream-ordered mempool.</summary>
    void TrimMemoryPool() { }

    /// <summary>Frees EVERY cached device allocation this backend holds — preloaded weights, cached dtype casts,
    /// and activations — and returns pool reservations to the driver. For host-application model switching: cache
    /// entries hold weight Tensors that were preloaded via <see cref="PreloadWeights"/>, and disposing the model
    /// objects alone does NOT release those device copies (the weight cache keeps them, and their Tensors, alive).
    /// Call this after evicting ALL models to guarantee a clean-slate VRAM budget for the next load. Do NOT call
    /// while any loaded model may still be used — its preloaded weights would silently degrade to per-op re-upload.
    /// No-op on backends without device caches.</summary>
    void FreeAllDeviceMemory() { }

    /// <summary>Free device memory in bytes (0 if not a device backend). For diagnostics / adaptive tiling.</summary>
    long FreeMemoryBytes() => 0;

    /// <summary>Pre-uploads weights into the backend's weight cache so subsequent ops hit cached device memory instead of re-uploading per call. No-op on backends without a weight cache; pair with <see cref="FreeWeights"/> at pipeline phase boundaries.</summary>
    void PreloadWeights(IEnumerable<Tensor> weights) { }

    /// <summary>Streaming cache for backends that overlap weight uploads with compute on a side stream. <c>null</c> on backends without that capability — consumers should fall back to <see cref="PreloadWeights"/> + <see cref="FreeWeights"/>.</summary>
    IStreamingWeightCache? StreamingCache => null;
}
