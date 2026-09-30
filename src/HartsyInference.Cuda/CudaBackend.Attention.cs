using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda.Profiling;
using HartsyInference.Gpu;

namespace HartsyInference.Cuda;

// Single-latent sparse attention, indexer, hyper-connection, latent-cache quantization, window and rope primitives.
public sealed partial class CudaBackend
{
    /// <inheritdoc />
    public void SparseLatentAttention(Tensor output, Tensor query, in LatentSource window, in LatentSource main,
        Tensor indices, int windowSlots, Tensor sink, float scale)
    {
        using NvtxRange _nvtx = NvtxRange.Push("SparseLatentAttention");
        SparseLatentAttentionReference.Validate(output, query, window, main, indices, windowSlots, sink);
        int k = (int)indices.Shape[1];
        if (k > CudaKernels.LatentAttentionMaxK)
            throw new NotSupportedException(
                $"CUDA SparseLatentAttention holds k <= {CudaKernels.LatentAttentionMaxK} probabilities in shared memory; got {k}.");
        using OpScope _op = EnterOp();
        EnsureKernels();
        RequireAttentionKernels();
        int tokens = (int)query.Shape[0], heads = (int)query.Shape[1], dim = (int)query.Shape[2];
        ulong pQ = 0, pIdx = 0, pSink = 0, pOut = 0, pWc = 0, pWs = 0, pMc = 0, pMs = 0;
        bool cached = false;
        try
        {
            pQ = GpuTransferHelper.CopyToDevice(query);
            pIdx = GpuTransferHelper.CopyToDevice(indices);
            pSink = GpuTransferHelper.CopyToDevice(sink);
            UploadSource(window, out pWc, out pWs);
            UploadSource(main, out pMc, out pMs);
            nuint bytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(bytes);
            _kernels!.LaunchSparseLatentAttention(pOut, pQ, pWc, pWs, pMc, pMs, pIdx, pSink, tokens, heads, dim, k,
                windowSlots, main.Rows, window.Encoding, main.Encoding, scale, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, bytes);
            cached = true;
        }
        finally
        {
            if (!cached) GpuTransferHelper.FreeDevice(pOut);
            FreeAll(pQ, pIdx, pSink, pWc, pWs, pMc, pMs);
        }
    }

    /// <inheritdoc />
    public void IndexerScores(Tensor scores, Tensor query, in LatentSource keys, Tensor headWeights, Tensor compressLens,
        Tensor? candidates, float scale)
    {
        using NvtxRange _nvtx = NvtxRange.Push("IndexerScores");
        IndexerScoresReference.Validate(scores, query, keys, headWeights, compressLens, candidates);
        int tokens = (int)query.Shape[0], heads = (int)query.Shape[1], dim = (int)query.Shape[2];
        if (keys.Rows > 0 && dim > CudaKernels.IndexerMaxDim)
            throw new NotSupportedException($"CUDA IndexerScores supports dim <= {CudaKernels.IndexerMaxDim}; got {dim}.");
        if (keys.Rows == 0) return;
        using OpScope _op = EnterOp();
        EnsureKernels();
        RequireAttentionKernels();
        ulong pQ = 0, pW = 0, pLen = 0, pCand = 0, pOut = 0, pKc = 0, pKs = 0;
        bool cached = false;
        try
        {
            pQ = GpuTransferHelper.CopyToDevice(query);
            pW = GpuTransferHelper.CopyToDevice(headWeights);
            pLen = GpuTransferHelper.CopyToDevice(compressLens);
            if (candidates is not null) pCand = GpuTransferHelper.CopyToDevice(candidates);
            UploadSource(keys, out pKc, out pKs);
            nuint bytes = GpuTransferHelper.ByteSize(scores);
            pOut = GpuTransferHelper.AllocateDevice(bytes);
            _kernels!.LaunchIndexerScores(pOut, pQ, pKc, pKs, pW, pLen, pCand, tokens, heads, dim, keys.Rows, keys.Encoding,
                scale, _stream.Handle);
            GpuTransferHelper.CacheActivation(scores, pOut, bytes);
            cached = true;
        }
        finally
        {
            if (!cached) GpuTransferHelper.FreeDevice(pOut);
            FreeAll(pQ, pW, pLen, pCand, pKc, pKs);
        }
    }

    /// <inheritdoc />
    public void HcSplitSinkhorn(Tensor pre, Tensor post, Tensor comb, Tensor mixes, Tensor scale, Tensor bias, int hc,
        int iters, float eps)
    {
        using NvtxRange _nvtx = NvtxRange.Push("HcSplitSinkhorn");
        HcReference.ValidateSplit(pre, post, comb, mixes, scale, bias, hc, iters);
        using OpScope _op = EnterOp();
        EnsureKernels();
        RequireAttentionKernels();
        int tokens = (int)(mixes.ElementCount / ((2 + hc) * hc));
        ulong pM = 0, pS = 0, pB = 0, pPre = 0, pPost = 0, pComb = 0;
        bool cPre = false, cPost = false, cComb = false;
        try
        {
            pM = GpuTransferHelper.CopyToDevice(mixes);
            pS = GpuTransferHelper.CopyToDevice(scale);
            pB = GpuTransferHelper.CopyToDevice(bias);
            nuint preBytes = GpuTransferHelper.ByteSize(pre), postBytes = GpuTransferHelper.ByteSize(post);
            nuint combBytes = GpuTransferHelper.ByteSize(comb);
            pPre = GpuTransferHelper.AllocateDevice(preBytes);
            pPost = GpuTransferHelper.AllocateDevice(postBytes);
            pComb = GpuTransferHelper.AllocateDevice(combBytes);
            _kernels!.LaunchHcSplitSinkhorn(pPre, pPost, pComb, pM, pS, pB, tokens, hc, iters, eps, _stream.Handle);
            GpuTransferHelper.CacheActivation(pre, pPre, preBytes); cPre = true;
            GpuTransferHelper.CacheActivation(post, pPost, postBytes); cPost = true;
            GpuTransferHelper.CacheActivation(comb, pComb, combBytes); cComb = true;
        }
        finally
        {
            if (!cPre) GpuTransferHelper.FreeDevice(pPre);
            if (!cPost) GpuTransferHelper.FreeDevice(pPost);
            if (!cComb) GpuTransferHelper.FreeDevice(pComb);
            FreeAll(pM, pS, pB);
        }
    }

    /// <inheritdoc />
    public void HcPreMix(Tensor output, Tensor x, Tensor pre)
    {
        using NvtxRange _nvtx = NvtxRange.Push("HcPreMix");
        HcReference.ValidatePreMix(output, x, pre);
        using OpScope _op = EnterOp();
        EnsureKernels();
        RequireAttentionKernels();
        long tokens = x.Shape[0];
        int hc = (int)x.Shape[1], dim = (int)x.Shape[2];
        ulong pX = 0, pPre = 0, pOut = 0;
        bool cached = false;
        try
        {
            pX = GpuTransferHelper.CopyToDevice(x);
            pPre = GpuTransferHelper.CopyToDevice(pre);
            nuint bytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(bytes);
            _kernels!.LaunchHcPreMix(pOut, pX, pPre, tokens, hc, dim, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, bytes);
            cached = true;
        }
        finally
        {
            if (!cached) GpuTransferHelper.FreeDevice(pOut);
            FreeAll(pX, pPre);
        }
    }

    /// <inheritdoc />
    public void HcPostMix(Tensor output, Tensor x, Tensor residual, Tensor post, Tensor comb)
    {
        using NvtxRange _nvtx = NvtxRange.Push("HcPostMix");
        HcReference.ValidatePostMix(output, x, residual, post, comb);
        using OpScope _op = EnterOp();
        EnsureKernels();
        RequireAttentionKernels();
        long tokens = residual.Shape[0];
        int hc = (int)residual.Shape[1], dim = (int)residual.Shape[2];
        ulong pX = 0, pRes = 0, pPost = 0, pComb = 0, pOut = 0;
        bool cached = false;
        try
        {
            pX = GpuTransferHelper.CopyToDevice(x);
            pRes = GpuTransferHelper.CopyToDevice(residual);
            pPost = GpuTransferHelper.CopyToDevice(post);
            pComb = GpuTransferHelper.CopyToDevice(comb);
            nuint bytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(bytes);
            _kernels!.LaunchHcPostMix(pOut, pX, pRes, pPost, pComb, tokens, hc, dim, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, bytes);
            cached = true;
        }
        finally
        {
            if (!cached) GpuTransferHelper.FreeDevice(pOut);
            FreeAll(pX, pRes, pPost, pComb);
        }
    }

    /// <inheritdoc />
    /// <remarks>A destination row past the last row is skipped on the device; the CPU reference throws.</remarks>
    public void QuantizeLatentRows(in LatentSource dest, Tensor rows, Tensor physicalRows)
    {
        using NvtxRange _nvtx = NvtxRange.Push("QuantizeLatentRows");
        LatentQuantReference.ValidateQuantize(dest, rows, physicalRows);
        using OpScope _op = EnterOp();
        EnsureKernels();
        RequireAttentionKernels();
        int count = (int)physicalRows.ElementCount;
        ulong pRows = 0, pPhys = 0, pWinner = 0, pCodes = 0, pScales = 0;
        try
        {
            pRows = GpuTransferHelper.CopyToDevice(rows);
            pPhys = GpuTransferHelper.CopyToDevice(physicalRows);
            pWinner = GpuTransferHelper.AllocateDevice((nuint)((long)dest.Rows * sizeof(int)));
            // The destination keeps its untouched rows, so its current contents go up and are updated in place.
            pCodes = GpuTransferHelper.CopyToDevice(dest.Codes!);
            if (dest.Scales is not null) pScales = GpuTransferHelper.CopyToDevice(dest.Scales);
            _kernels!.LaunchQuantizeLatentRows(pCodes, pScales, pRows, pPhys, pWinner, count, dest.Dim, dest.Rows,
                dest.Encoding, _stream.Handle);
            RecacheInPlace(dest.Codes!, pCodes);
            if (dest.Scales is not null) RecacheInPlace(dest.Scales, pScales);
        }
        finally
        {
            FreeAll(pRows, pPhys, pWinner);
        }
    }

    /// <inheritdoc />
    public void ActQuantDequantInPlace(Tensor x, LatentEncoding encoding)
    {
        using NvtxRange _nvtx = NvtxRange.Push("ActQuantDequantInPlace");
        LatentQuantReference.ValidateActQuant(x, encoding);
        int group = LatentEncodings.GroupSize(encoding);
        if (group == 0) return;
        using OpScope _op = EnterOp();
        EnsureKernels();
        RequireAttentionKernels();
        ulong pX = 0;
        try
        {
            pX = GpuTransferHelper.CopyToDevice(x);
            _kernels!.LaunchActQuantDequantInPlace(pX, x.ElementCount / group, encoding, _stream.Handle);
            RecacheInPlace(x, pX);
        }
        finally
        {
            // pX is the tensor's own cached buffer now; FreeDevice ignores cached pointers.
            GpuTransferHelper.FreeDevice(pX);
        }
    }

    /// <inheritdoc />
    public void BuildWindowIndices(Tensor indices, int windowSize, int seqLen, int startPos)
    {
        using NvtxRange _nvtx = NvtxRange.Push("BuildWindowIndices");
        WindowIndicesReference.Validate(indices, windowSize, seqLen, startPos);
        using OpScope _op = EnterOp();
        EnsureKernels();
        RequireAttentionKernels();
        (int rows, int cols) = WindowIndicesReference.Shape(windowSize, seqLen, startPos);
        ulong pOut = 0;
        bool cached = false;
        try
        {
            nuint bytes = GpuTransferHelper.ByteSize(indices);
            pOut = GpuTransferHelper.AllocateDevice(bytes);
            _kernels!.LaunchWindowIndices(pOut, windowSize, startPos, rows, cols, _stream.Handle);
            GpuTransferHelper.CacheActivation(indices, pOut, bytes);
            cached = true;
        }
        finally
        {
            if (!cached) GpuTransferHelper.FreeDevice(pOut);
        }
    }

    /// <inheritdoc />
    public void ApplyRopeInterleaved(Tensor x, Tensor cos, Tensor sin, int rotaryDim, int dimOffset)
    {
        using NvtxRange _nvtx = NvtxRange.Push("RopeInterleavedOffset");
        RopeInterleavedOffsetReference.Validate(x, cos, sin, rotaryDim, dimOffset);
        using OpScope _op = EnterOp();
        EnsureKernels();
        RequireAttentionKernels();
        int rank = x.Shape.Rank;
        int heads = rank == 4 ? (int)x.Shape[2] : 1;
        int dim = (int)x.Shape[rank - 1];
        long positions = x.Shape[0] * x.Shape[1];
        ulong pX = 0, pCos = 0, pSin = 0;
        try
        {
            pX = GpuTransferHelper.CopyToDevice(x);
            pCos = GpuTransferHelper.CopyToDevice(cos);
            pSin = GpuTransferHelper.CopyToDevice(sin);
            _kernels!.LaunchRopeInterleavedOffset(pX, pCos, pSin, positions, heads, dim, rotaryDim / 2, dimOffset, _stream.Handle);
            RecacheInPlace(x, pX);
        }
        finally
        {
            FreeAll(pX, pCos, pSin);
        }
    }

    // Uploads a source's codes and scales; both stay 0 for an empty source or for F32 (no scales).
    private static void UploadSource(in LatentSource source, out ulong codes, out ulong scales)
    {
        codes = 0;
        scales = 0;
        if (source.Rows == 0) return;
        codes = GpuTransferHelper.CopyToDevice(source.Codes!);
        if (source.Scales is not null) scales = GpuTransferHelper.CopyToDevice(source.Scales);
    }

    // An in-place op rebinds the tensor to its updated device buffer; stale callbacks must go first (pitfall #17).
    private static void RecacheInPlace(Tensor tensor, ulong devicePtr)
    {
        tensor._gpuSyncCallback = null;
        tensor._gpuDisposeCallback = null;
        GpuTransferHelper.CacheActivation(tensor, devicePtr, GpuTransferHelper.ByteSize(tensor));
    }

    private static void FreeAll(params ulong[] pointers)
    {
        foreach (ulong p in pointers)
            if (p != 0) GpuTransferHelper.FreeDevice(p);
    }

    private void RequireAttentionKernels()
    {
        if (!_kernels!.HasAttentionKernels)
            throw new NotSupportedException("The latent attention PTX modules (Kernels/latent) are not available on this device.");
    }
}
