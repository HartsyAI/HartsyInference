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

}
