using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

// Layout moves: the generic strided copy, accumulate and pad, patchify and pixel shuffles, row and sequence slices and scatters.
public partial interface IBackend
{
    // Default host impls = numeric reference; CudaBackend overrides with PTX kernels to keep
    // the attention island GPU-resident.

    /// <summary>Strided copy: <c>output[dst(i)] = input[src(i)]</c> for every multi-index <c>i</c> of the spec's extents, a
    /// pure move of 1, 2, 4 or 8-byte elements between tensors of one dtype. Elements of <paramref name="output"/> the
    /// destination view does not address keep their contents, so a scatter into a slot of a larger tensor is in place.
    /// Slices, head splits and merges, permutes, broadcasts and <c>repeat_interleave</c> are all one call: build the view
    /// with <see cref="StridedView"/> or a <see cref="StridedCopies"/> helper. Contract: <c>docs/BACKEND_OPS.md</c>;
    /// <see cref="StridedCopyReference.Copy"/> is its executable form.</summary>
    void CopyStrided(Tensor output, Tensor input, in StridedCopySpec spec) => StridedCopyReference.Copy(output, input, spec);

    /// <summary>Strided accumulate: <c>output[dst(i)] += input[src(i)]</c> over the spec's extents, in place, for F32, F16 or
    /// BF16 tensors of one dtype; the add is F32 and the result rounds to the storage dtype (bfloat16 ties to even). The
    /// destination addresses every element once, so no two adds race and the order never matters. Contract:
    /// <c>docs/BACKEND_OPS.md</c>; <see cref="StridedCopyReference.Accumulate"/> is its executable form.</summary>
    void AccumulateStrided(Tensor output, Tensor input, in StridedCopySpec spec) => StridedCopyReference.Accumulate(output, input, spec);

    /// <summary>Pads <paramref name="input"/> into <paramref name="output"/> (<c>F.pad</c>): every axis grows by the spec's
    /// before and after counts, and the new positions are zero bits, the nearest edge element or its reflection per
    /// <see cref="PadMode"/>. Any 1, 2, 4 or 8-byte dtype. Contract: <c>docs/BACKEND_OPS.md</c>;
    /// <see cref="StridedCopyReference.Pad"/> is its executable form.</summary>
    void Pad(Tensor output, Tensor input, in PadSpec spec) => StridedCopyReference.Pad(output, input, spec);

    /// <summary>Oasis per-frame unpatchify: <c>proj[t·sp, c·p²]</c> → <c>out[t,c,H,W]</c>, out-vector laid <c>[py,px,ci]</c>.</summary>
    unsafe void OasisUnpatchify(Tensor output, Tensor proj, int frames, int channels, int gh, int gw, int patch)
    {
        if (output.DType != DType.F32 || proj.DType != DType.F32) throw new NotSupportedException("OasisUnpatchify default fallback only supports F32.");
        int p = patch, h = gh * p, w = gw * p, outVec = channels * p * p;
        float* src = (float*)proj.DataPointer, dst = (float*)output.DataPointer;
        for (int f = 0; f < frames; f++)
            for (int ty = 0; ty < gh; ty++)
                for (int tx = 0; tx < gw; tx++)
                {
                    long token = ((long)f * gh + ty) * gw + tx;
                    for (int ci = 0; ci < channels; ci++)
                        for (int py = 0; py < p; py++)
                            for (int px = 0; px < p; px++)
                                dst[(((long)f * channels + ci) * h + ty * p + py) * w + tx * p + px] = src[token * outVec + (long)(py * p + px) * channels + ci];
                }
    }

    /// <summary>LTX-2.5 channels-last 3D pixel shuffle of one temporal chunk: each source token
    /// <c>[·, strideT·strideH·strideW·outChannels]</c> unpacks into a <c>strideT×strideH×strideW</c> sub-block of
    /// output tokens <c>[·, outChannels]</c>, source channel <c>((c·strideT+i1)·strideH+i2)·strideW+i3</c> going to
    /// destination channel <c>c</c> of sub-block position <c>(i1, i2, i3)</c>.</summary>
    /// <param name="output">The FULL <c>[outT·outH·outW, outChannels]</c> result; this call fills only the frames
    /// this chunk covers, so the destination persists across chunk calls.</param>
    /// <param name="start">First source frame of this chunk on the full temporal axis.</param>
    /// <param name="dropped">1 while the causal shuffle's duplicated leading frame is being dropped, which shifts
    /// every destination frame down by one and leaves this chunk's first sub-frame without a destination.</param>
    unsafe void Ltx25PixelShuffle(Tensor output, Tensor projected, int start, int h, int w,
        int strideT, int strideH, int strideW, int outChannels, int dropped, int outH, int outW)
        => Ltx25PixelShuffleReference(output, projected, start, h, w,
            strideT, strideH, strideW, outChannels, dropped, outH, outW);

    /// <summary>The managed <see cref="Ltx25PixelShuffle"/> body, callable directly so a device backend can fall back
    /// to it without going through interface dispatch — which would re-enter the backend's own override, not this
    /// default — and so a test can diff against it independently.</summary>
    static unsafe void Ltx25PixelShuffleReference(Tensor output, Tensor projected, int start, int h, int w,
        int strideT, int strideH, int strideW, int outChannels, int dropped, int outH, int outW)
    {
        if (output.DType != DType.F32 || projected.DType != DType.F32)
            throw new NotSupportedException($"Ltx25PixelShuffle default fallback only supports F32 — got output={output.DType}, projected={projected.DType}.");
        int projChannels = strideT * strideH * strideW * outChannels;
        long plane = (long)h * w;
        int count = (int)(projected.Shape[0] / plane);
        float* src = (float*)projected.DataPointer;
        float* dst = (float*)output.DataPointer;
        for (int ti = start; ti < start + count; ti++)
        for (int i1 = 0; i1 < strideT; i1++)
        {
            int frame = ti * strideT + i1 - dropped;
            if (frame < 0) continue;
            for (int hi = 0; hi < h; hi++)
            for (int i2 = 0; i2 < strideH; i2++)
            for (int wi = 0; wi < w; wi++)
            for (int i3 = 0; i3 < strideW; i3++)
            {
                long srcRow = ((long)(ti - start) * h + hi) * w + wi;
                long dstRow = (((long)frame * outH + hi * strideH + i2) * outW) + wi * strideW + i3;
                for (int c = 0; c < outChannels; c++)
                {
                    int packed = ((c * strideT + i1) * strideH + i2) * strideW + i3;
                    dst[dstRow * outChannels + c] = src[srcRow * projChannels + packed];
                }
            }
        }
    }

    /// <summary>SeedVR2 MAGViT channel→space shuffle: <c>[1,cIn,f,h,w] → [1,c,fFinal,h·sr,w·sr]</c> with
    /// <c>c = cIn/(sr²·tr)</c>, source channel <c>((xi·sr+yi)·tr+zi)·c+ci</c>, and (temporal only) OUTPUT
    /// FRAME INDEX 1 dropped (T→2T−1). Shapes carry every dim; F32/BF16. Default is the host reference.</summary>
    unsafe void SeedVr2PixelShuffle(Tensor output, Tensor input, int spatialRatio, int temporalRatio)
    {
        if (input.DType != DType.F32 || output.DType != DType.F32)
            throw new NotSupportedException("SeedVr2PixelShuffle default fallback only supports F32.");
        Sync();
        int cIn = (int)input.Shape[1], f = (int)input.Shape[2], h = (int)input.Shape[3], w = (int)input.Shape[4];
        int c = (int)output.Shape[1], fFinal = (int)output.Shape[2], hOut = (int)output.Shape[3], wOut = (int)output.Shape[4];
        bool dropDup = temporalRatio > 1;
        float* src = (float*)input.DataPointer;
        float* dst = (float*)output.DataPointer;
        long total = (long)c * fFinal * hOut * wOut;
        for (long idx = 0; idx < total; idx++)
        {
            int ox = (int)(idx % wOut);
            long r = idx / wOut;
            int oy = (int)(r % hOut); r /= hOut;
            int of = (int)(r % fFinal);
            int ci = (int)(r / fFinal);
            int outF = dropDup && of >= 1 ? of + 1 : of;
            int fi = outF / temporalRatio, zi = outF % temporalRatio;
            int y = oy / spatialRatio, xi = oy % spatialRatio;
            int x = ox / spatialRatio, yi = ox % spatialRatio;
            int srcC = ((xi * spatialRatio + yi) * temporalRatio + zi) * c + ci;
            dst[idx] = src[(((long)srcC * f + fi) * h + y) * w + x];
        }
    }

    /// <summary>Image-output conversion: CHW F32 in [-1,1] → interleaved HWC u8 in [0,255] (round-half-up, clamped).</summary>
    unsafe void ChwF32ToHwcU8(Tensor output, Tensor input) => ChwF32ToHwcU8Reference(output, input);

    /// <summary>The host implementation of <see cref="ChwF32ToHwcU8"/>, for a backend that overrides it and needs
    /// to bail on a dtype or rank its kernel does not cover.</summary>
    /// <remarks>Static because an override cannot reach its own interface default — the call binds back to the
    /// class and recurses until the stack ends.</remarks>
    static unsafe void ChwF32ToHwcU8Reference(Tensor output, Tensor input)
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

    /// <summary>DiT token-grid patchify: <c>[B,C,H,W]</c> → <c>[B,(H/p)·(W/p),p²·C]</c>. This byte-preserving shuffle supports F32, F16, and BF16.</summary>
    /// <param name="innerChannelFastest">Token order: true=(ph,pw,c) (Z-Image/Lumina2), false=channel-outer (Krea2/diffusers).</param>
    unsafe void PatchifyTokens(Tensor output, Tensor input, int patch, bool innerChannelFastest)
    {
        PatchTokenGeometry geometry = PatchTokenContract.ValidatePatchify(output, input, patch);
        PatchTokenHostShuffle.Patchify(output, input, geometry, patch, innerChannelFastest);
    }

    /// <summary>DiT token-grid unpatchify: <c>[B,hP·wP,C·p²]</c> → <c>[B,C,hP·p,wP·p]</c> (F32/F16/BF16).</summary>
    /// <param name="innerChannelFastest">Token order: true=(ph,pw,c) (Z-Image/Lumina2), false=channel-outer (Krea2/diffusers).</param>
    unsafe void UnpatchifyTokens(Tensor output, Tensor tokens, int channels, int hPacked, int wPacked, int patch,
        bool innerChannelFastest)
    {
        PatchTokenGeometry geometry = PatchTokenContract.ValidateUnpatchify(
            output, tokens, channels, hPacked, wPacked, patch);
        PatchTokenHostShuffle.Unpatchify(output, tokens, geometry, patch, innerChannelFastest);
    }

    /// <summary>Wan2.2 VAE unpatchify: <c>[b, c·p², t, h, w] → [b, c, t, h·p, w·p]</c>, unpack <c>oc = ci·p² + r·p + q</c>.
    /// Default = host loop, copying by element size so any dtype passes through unchanged.</summary>
    unsafe void UnpatchifyVae(Tensor output, Tensor input, int patchSize)
    {
        if (output.DType != input.DType)
            throw new ArgumentException($"UnpatchifyVae output {output.DType} must match input {input.DType}.", nameof(output));
        int b = (int)input.Shape[0], packedC = (int)input.Shape[1], t = (int)input.Shape[2], h = (int)input.Shape[3], w = (int)input.Shape[4];
        int p = patchSize, c = packedC / (p * p), outH = h * p, outW = w * p;
        int elem = (int)input.DType.ComputeByteCount(1);
        byte* src = (byte*)input.DataPointer, dst = (byte*)output.DataPointer;
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
                                    Buffer.MemoryCopy(src + srcOff * elem, dst + dstOff * elem, elem, elem);
                                }
    }

    /// <summary>Wan2.2 VAE duplicating up-sampler (<c>DupUp3D</c>): <c>[b, inC, t, h, w] → [b, outC, t·fT − dropT, h·fS, w·fS]</c>,
    /// output cell <c>(oc, t·fT+tt, h·fS+s1, w·fS+s2)</c> reading input channel <c>(((oc·fT + tt)·fS + s1)·fS + s2) / repeats</c>.
    /// Default = host loop.</summary>
    unsafe void DupUp3dVae(Tensor output, Tensor input, int factorT, int factorS, int dropT)
    {
        int b = (int)input.Shape[0], inC = (int)input.Shape[1], t = (int)input.Shape[2];
        int h = (int)input.Shape[3], w = (int)input.Shape[4], outC = (int)output.Shape[1];
        int keepT = t * factorT - dropT, outH = h * factorS, outW = w * factorS;
        int repeats = outC * factorT * factorS * factorS / inC;
        int elem = (int)input.DType.ComputeByteCount(1);
        byte* src = (byte*)input.DataPointer, dst = (byte*)output.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (int oc = 0; oc < outC; oc++)
                for (int tt = 0; tt < factorT; tt++)
                    for (int s1 = 0; s1 < factorS; s1++)
                        for (int s2 = 0; s2 < factorS; s2++)
                        {
                            int srcC = (((oc * factorT + tt) * factorS + s1) * factorS + s2) / repeats;
                            for (int ti = 0; ti < t; ti++)
                            {
                                int oTime = ti * factorT + tt - dropT;
                                if (oTime < 0) continue;
                                for (int hi = 0; hi < h; hi++)
                                    for (int wi = 0; wi < w; wi++)
                                    {
                                        long srcOff = ((((long)bi * inC + srcC) * t + ti) * h + hi) * w + wi;
                                        long dstOff = ((((long)bi * outC + oc) * keepT + oTime) * outH + (hi * factorS + s1))
                                            * outW + (wi * factorS + s2);
                                        Buffer.MemoryCopy(src + srcOff * elem, dst + dstOff * elem, elem, elem);
                                    }
                            }
                        }
    }

    /// <summary>Copies a contiguous last-dim slice: <c>out[row,d] = in[row,offset+d]</c>; splits a fused tensor (e.g. QKV) into a chunk.</summary>
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

    /// <summary>Per-row scalar multiply: <c>out[row, c] = in[row, c] * rowMask[row]</c> — token masking for DiT embeddings.</summary>
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

    /// <summary>Writes a <c>[1, heads, c, hd]</c> chunk into sequence rows <c>[seqOffset, seqOffset + c)</c> of a
    /// <c>[1, heads, seq, hd]</c> head-major tensor, in place and accumulating across calls. Same bytes to the same
    /// offsets as concatenating every chunk along dim 2, but without holding the whole chunk list alive alongside the
    /// result — which is what makes long-sequence chunked attention fit (see <c>MiniMaxH3Transformer</c>).</summary>
    void ScatterSeqHeadMajor(Tensor output, Tensor input, int seqOffset)
        => ScatterSeqHeadMajor(output, input, seqOffset, input.Shape.Rank > 2 ? (int)input.Shape[2] : 0);

    /// <summary>As <see cref="ScatterSeqHeadMajor(Tensor, Tensor, int)"/> for only the first <paramref name="rows"/>
    /// sequence rows of <paramref name="input"/>, so the two sequence extents are independent: this is how a KV
    /// cache's committed prefix moves into a larger or smaller buffer. A byte copy, for any non-quantized dtype that
    /// matches on both sides.</summary>
    unsafe void ScatterSeqHeadMajor(Tensor output, Tensor input, int seqOffset, int rows)
        => ScatterSeqHeadMajorReference(output, input, seqOffset, rows);

    /// <summary>The managed <see cref="ScatterSeqHeadMajor(Tensor, Tensor, int, int)"/> body, callable directly. A
    /// backend override must call THIS to fall back — <c>((IBackend)this).ScatterSeqHeadMajor(...)</c> re-enters the
    /// override through interface dispatch and recurses until the stack overflows.</summary>
    static unsafe void ScatterSeqHeadMajorReference(Tensor output, Tensor input, int seqOffset, int rows)
    {
        SeqHeadMajorScatter geometry = SeqHeadMajorScatterContract.Validate(output, input, seqOffset, rows);
        if (geometry.IsEmpty) return;
        byte* pOut = (byte*)output.DataPointer;
        byte* pIn = (byte*)input.DataPointer;
        for (int h = 0; h < geometry.Heads; h++)
        {
            Buffer.MemoryCopy(pIn + geometry.SourceOffset(h), pOut + geometry.DestinationOffset(h),
                geometry.SliceBytes, geometry.SliceBytes);
        }
    }

    /// <summary>Contiguous row-block slice: copies <c>output.ElementCount</c> elements starting at row rowOffset of input into output.</summary>
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

    /// <summary>Contiguous row-block slice, dtype-agnostic (a pure byte-range copy — used where <see cref="SliceRows"/>'s
    /// F32/F16/BF16 guard is too narrow, e.g. slicing an fp8 activation for chunked attention/MLP). Copies
    /// <see cref="Tensor.Fp8ScaleFactor"/> onto <paramref name="output"/> so a sliced fp8 chunk still carries its
    /// parent's per-tensor dequant scale into a downstream fp8 GEMM.</summary>
    unsafe void SliceRowsGeneric(Tensor output, Tensor input, int rowOffset)
    {
        if (output.DType != input.DType)
            throw new ArgumentException($"SliceRowsGeneric requires matching dtypes, got output {output.DType} vs input {input.DType}.");
        if (output.DType.IsQuantized)
            throw new NotSupportedException("SliceRowsGeneric does not support block-quantized dtypes.");
        int dim = (int)output.Shape[output.Shape.Rank - 1];
        long rowBytes = output.DType.ComputeByteCount(dim);
        long byteOffset = (long)rowOffset * rowBytes;
        long totalBytes = output.DType.ComputeByteCount(output.ElementCount);
        Buffer.MemoryCopy((byte*)input.DataPointer + byteOffset, output.DataPointer, totalBytes, totalBytes);
        output.Fp8ScaleFactor = input.Fp8ScaleFactor;
    }

    /// <summary>Writes <paramref name="input"/>'s rows into <paramref name="output"/> starting at row
    /// <paramref name="rowOffset"/>, in place and accumulating across calls — the byte-offset inverse of
    /// <see cref="SliceRowsGeneric"/>. Same bytes to the same offsets as concatenating every chunk along dim 0, but
    /// without holding the whole chunk list alive alongside the result, which is what makes long-sequence chunked
    /// attention/MLP fit (see <c>MiniMaxH3Transformer</c>; the head-major sibling is
    /// <see cref="ScatterSeqHeadMajor(Tensor, Tensor, int)"/>).</summary>
    unsafe void ScatterRowsGeneric(Tensor output, Tensor input, int rowOffset)
    {
        if (output.DType != input.DType)
            throw new ArgumentException($"ScatterRowsGeneric requires matching dtypes, got output {output.DType} vs input {input.DType}.");
        if (output.DType.IsQuantized)
            throw new NotSupportedException("ScatterRowsGeneric does not support block-quantized dtypes.");
        int dim = (int)input.Shape[input.Shape.Rank - 1];
        long byteOffset = (long)rowOffset * input.DType.ComputeByteCount(dim);
        long totalBytes = input.DType.ComputeByteCount(input.ElementCount);
        Buffer.MemoryCopy(input.DataPointer, (byte*)output.DataPointer + byteOffset, totalBytes, totalBytes);
    }

    /// <summary>Frame-repeat over time on <c>[B,C,inT]</c>: each time-step duplicated numSamples times (nearest 1D upsample).</summary>
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

    /// <summary>3D pixel-shuffle (TRELLIS SS-VAE upsample): <c>[N,C·r³,D,H,W] → [N,C,rD,rH,rW]</c>, matching reshape+permute.</summary>
    unsafe void PixelShuffle3d(Tensor output, Tensor input, int ratio)
    {
        if (input.DType != DType.F32 || output.DType != DType.F32)
            throw new NotSupportedException("PixelShuffle3d default fallback only supports F32.");
        int n = (int)input.Shape[0], cIn = (int)input.Shape[1];
        int iD = (int)input.Shape[2], iH = (int)input.Shape[3], iW = (int)input.Shape[4];
        int r = ratio, r3 = r * r * r, cOut = cIn / r3;
        int oD = iD * r, oH = iH * r, oW = iW * r;
        float* src = (float*)input.DataPointer, dst = (float*)output.DataPointer;
        for (int b = 0; b < n; b++)
            for (int c = 0; c < cOut; c++)
                for (int d = 0; d < iD; d++)
                    for (int h = 0; h < iH; h++)
                        for (int w = 0; w < iW; w++)
                            for (int a = 0; a < r; a++)
                                for (int bb = 0; bb < r; bb++)
                                    for (int e = 0; e < r; e++)
                                    {
                                        long ci = (long)c * r3 + a * r * r + bb * r + e;
                                        long si = (((((long)b * cIn + ci) * iD + d) * iH + h) * iW + w);
                                        long di = (((((long)b * cOut + c) * oD + (d * r + a)) * oH + (h * r + bb)) * oW + (w * r + e));
                                        dst[di] = src[si];
                                    }
    }

    /// <summary>Spatial-only pixel-shuffle over a 5-D volume (torch <c>nn.PixelShuffle</c> applied per depth slice):
    /// <c>[N,C·r²,D,H,W] → [N,C,D,rH,rW]</c> with source channel <c>c·r² + p1·r + p2</c>. Channel-OUTER nesting —
    /// <see cref="SeedVr2PixelShuffle"/> nests the other way and is not interchangeable. Default is the host reference.</summary>
    unsafe void PixelShuffle2d(Tensor output, Tensor input, int ratio)
        => PixelShuffle2dReference(output, input, ratio);

    /// <summary>The managed <see cref="PixelShuffle2d"/> body, callable directly. A backend override must
    /// call THIS to fall back — <c>((IBackend)this).PixelShuffle2d(...)</c> re-enters the override and
    /// recurses until the stack overflows.</summary>
    static unsafe void PixelShuffle2dReference(Tensor output, Tensor input, int ratio)
    {
        if (input.DType != DType.F32 || output.DType != DType.F32)
            throw new NotSupportedException("PixelShuffle2d default fallback only supports F32.");
        // The instance default used to drain the whole device here. Reading DataPointer below already forces
        // this tensor's own lazy sync, which is the dependency that actually matters; a full drain was wider
        // than the op needs and is not available from a static.
        int n = (int)input.Shape[0], cIn = (int)input.Shape[1];
        int d = (int)input.Shape[2], iH = (int)input.Shape[3], iW = (int)input.Shape[4];
        int r = ratio, r2 = r * r, cOut = cIn / r2;
        int oH = iH * r, oW = iW * r;
        if ((long)cOut * r2 != cIn || output.Shape[1] != cOut || output.Shape[3] != oH || output.Shape[4] != oW)
            throw new ArgumentException($"PixelShuffle2d ratio {r}: input {input.Shape} does not shuffle into {output.Shape}.");
        float* src = (float*)input.DataPointer, dst = (float*)output.DataPointer;
        for (int b = 0; b < n; b++)
            for (int c = 0; c < cOut; c++)
                for (int z = 0; z < d; z++)
                    for (int h = 0; h < iH; h++)
                        for (int w = 0; w < iW; w++)
                            for (int p1 = 0; p1 < r; p1++)
                                for (int p2 = 0; p2 < r; p2++)
                                {
                                    long ci = (long)c * r2 + p1 * r + p2;
                                    long si = ((((long)b * cIn + ci) * d + z) * iH + h) * iW + w;
                                    long di = ((((long)b * cOut + c) * d + z) * oH + (h * r + p1)) * oW + (w * r + p2);
                                    dst[di] = src[si];
                                }
    }
}
