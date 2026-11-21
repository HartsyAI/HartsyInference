using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda.Profiling;

namespace HartsyInference.Cuda;

// Causal prefill attention through the tensor-core FlashAttention-2 kernel.
public sealed partial class CudaBackend
{
    /// <summary>Smallest query block the FlashAttention-2 kernel takes; a single decode row stays on the split-K path.</summary>
    internal const int Fa2MinQueryRows = 16;

    private void FlashAttentionFa2(Tensor output, Tensor query, Tensor key, Tensor value, int b, int hq, int hkv, int tq, int keyStride,
        int d, int kvLen, int kvGroup, int qOffset, float scale, int slidingWindow, bool f16Kv)
    {
        using NvtxRange _nvtx = NvtxRange.Push("FlashAttention.Fa2");
        using OpScope _op = EnterOp();
        EnsureKernels();
        ulong pQ = 0, pK = 0, pV = 0, pOut = 0, kTmp = 0, vTmp = 0;
        bool cachedOutput = false;
        try
        {
            pQ = GpuTransferHelper.CopyToDevice(query);
            pK = GpuTransferHelper.CopyToDevice(key);
            pV = GpuTransferHelper.CopyToDevice(value);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            ulong kUse = pK, vUse = pV;
            int stride = keyStride;
            if (!f16Kv)
            {
                // The F32 cache becomes a tight F16 copy of its valid positions; one pass over kvLen rows per KV head.
                nuint bytes = (nuint)((long)b * hkv * kvLen * d * sizeof(ushort));
                kTmp = GpuTransferHelper.AllocateDevice(bytes);
                vTmp = GpuTransferHelper.AllocateDevice(bytes);
                AttentionEvidence.Observed(AttentionEvidenceKind.Workspace, "cuda.fa2.kv.f16", 2 * (long)bytes, count: 2);
                AttentionEvidence.Observed(AttentionEvidenceKind.Cast, "cuda.fa2.kv.f16", 2 * (long)bytes, count: 2);
                _kernels!.LaunchKvToF16(kTmp, vTmp, pK, pV, b * hkv, kvLen, keyStride, d, _stream.Handle);
                kUse = kTmp;
                vUse = vTmp;
                stride = kvLen;
            }
            _kernels!.LaunchFa2Causal(pOut, pQ, kUse, vUse, b, hq, tq, d, hkv, stride, kvLen, kvGroup, qOffset, scale, slidingWindow, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pQ);
            GpuTransferHelper.FreeDevice(pK);
            GpuTransferHelper.FreeDevice(pV);
            if (kTmp != 0) GpuTransferHelper.FreeDevice(kTmp);
            if (vTmp != 0) GpuTransferHelper.FreeDevice(vTmp);
        }
    }

    /// <summary>True when the grouped-query flash-decoding kernel can serve this single-row decode call.</summary>
    private bool DecodeGqaEligible(int b, int tq, bool causal, int hq, int hkv, int kvGroup, int d) =>
        tq == 1 && causal && hkv > 0 && hq == hkv * kvGroup && _kernels is { HasDecodeGqa: true } && CudaKernels.DecodeGqaSupports(d, kvGroup)
        && CudaKernels.GridYzFits(hkv, b) && EngineKnobs.FlashDecodeGqa.Value;

    /// <summary>Grouped-query flash-decoding into <paramref name="pOut"/>: partial states per (split, KV head), then the shared combine.
    /// With <paramref name="devicePos"/> the split count follows the cache capacity <paramref name="keySpan"/> so a captured graph keeps its grid.
    /// With <paramref name="pageTable"/> the keys and values are rows of F32 page buffers of <paramref name="keyStride"/> rows a head, found
    /// through that table; the split geometry is the contiguous call's, so the same rows give the same partials.</summary>
    private void RunFlashDecodeGqa(ulong pOut, ulong pQ, ulong pK, ulong pV, int b, int hq, int hkv, int keyStride, int keySpan, int d, int kvLen,
        int kvGroup, int qOffset, float scale, float softcap, int window, ulong devicePos, bool f16Kv, AttnSidecar sidecar = default,
        ulong pageTable = 0, int pageShift = 0)
    {
        // One wave: shared memory leaves room for four blocks per SM with an F16 cache and two with F32, so more blocks than that run a half-empty second wave.
        int target = (f16Kv ? 4 : 2) * _context.MultiprocessorCount;
        int splits = Math.Clamp((target + b * hkv - 1) / (b * hkv), 1, Math.Min(64, Math.Max(1, keySpan / 64)));
        int chunk = (keySpan + splits - 1) / splits;
        splits = (keySpan + chunk - 1) / chunk;
        long n = (long)b * hq;
        DecodePartials partials = default;
        try
        {
            partials = RentDecodePartials(n, splits, d, "cuda.flash.decode.partials");
            ulong pM = partials.M, pL = partials.L, pAcc = partials.Acc;
            if (pageTable != 0)
                _kernels!.LaunchFlashDecodeGqaPaged(pM, pL, pAcc, pQ, pK, pV, hq, d, hkv, keyStride, kvLen, kvGroup, qOffset, scale, softcap, window,
                    splits, chunk, pageTable, pageShift, _stream.Handle);
            else
                _kernels!.LaunchFlashDecodeGqa(pM, pL, pAcc, pQ, pK, pV, b, hq, d, hkv, keyStride, kvLen, kvGroup, qOffset, scale, softcap, window,
                    splits, chunk, devicePos, f16Kv, _stream.Handle);
            if (sidecar.Xq != 0)
                _kernels.LaunchFlashAttentionCombineQ8(pOut, sidecar.Xq, sidecar.Xd, sidecar.Xs, pM, pL, pAcc, b, hq, 1, d, splits, _stream.Handle);
            else
                _kernels.LaunchFlashAttentionCombine(pOut, pM, pL, pAcc, b, hq, 1, d, splits, _stream.Handle);
        }
        finally
        {
            ReturnDecodePartials(in partials);
        }
    }

    /// <summary>The paged decode row's admission: the single-row causal call the contiguous grouped decode (F.4) takes, over F32
    /// pages of a power-of-two size, with the paged kernel loaded. Anything else over paged keys and values is gathered by the caller.</summary>
    private bool PagedDecodeAdmits(in AttentionCall c)
    {
        if (c.KvPages is not { } pages) return false;
        int size = pages.PageSize, group = c.KvGroup <= 0 ? 1 : c.KvGroup;
        return size > 0 && (size & (size - 1)) == 0 && c.Batch == 1 && c.Logits.Sinks is null && c.Logits.AlibiSlopes is null
            && c.Mask.Bias is null && c.Mask.WindowRight is null && c.Mask.BlockSize == 0 && c.Positions.KvLength >= 64
            && c.Query.DType == DType.F32 && c.Output.DType == DType.F32 && c.Key.DType == DType.F32 && c.Value.DType == DType.F32
            && _kernels is { HasDecodeGqaPaged: true }
            && DecodeGqaEligible(1, c.QueryTokens, c.Mask.Causal, c.Heads, c.KvHeads, group, c.QkHeadDim);
    }

    /// <summary>One decode row over paged keys and values (F.8): the grouped decode kernel reading each row through the page table.</summary>
    private unsafe void FlashAttentionDecodePaged(in AttentionCall c)
    {
        AttentionKvPages pages = c.KvPages!.Value;
        Tensor output = c.Output.Storage, query = c.Query.Storage, keyPages = c.Key.Storage, valuePages = c.Value.Storage, table = pages.Table;
        int hq = c.Heads, hkv = c.KvHeads, d = c.QkHeadDim, poolRows = c.KeyTokens, kvLen = c.Positions.KvLength;
        int group = c.KvGroup <= 0 ? 1 : c.KvGroup;
        using OpScope _op = EnterOp(nameof(FlashAttention));
        EnsureKernels();
        ulong pQ = 0, pK = 0, pV = 0, pT = 0, pOut = 0;
        bool cachedOutput = false;
        try
        {
            pQ = GpuTransferHelper.CopyToDevice(query);
            pK = GpuTransferHelper.CopyToDevice(keyPages);
            pV = GpuTransferHelper.CopyToDevice(valuePages);
            pT = GpuTransferHelper.CopyToDevice(table);
            // The owner replaces the table tensor when the table changes, so its device copy serves every call until then
            // instead of being uploaded for each layer of each step.
            if (!GpuTransferHelper.IsActivationCached(table) && !GpuTransferHelper.IsWeightCached(table))
                GpuTransferHelper.CacheActivation(table, pT, GpuTransferHelper.ByteSize(table));
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            t_attentionRoute = "F.8";
            RunFlashDecodeGqa(pOut, pQ, pK, pV, 1, hq, hkv, poolRows, kvLen, d, kvLen, group, c.Positions.QueryOffset, c.Scale, c.Logits.Softcap,
                c.Mask.LegacySlidingWindow, 0, f16Kv: false, pageTable: pT, pageShift: System.Numerics.BitOperations.Log2((uint)pages.PageSize));
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pQ);
            GpuTransferHelper.FreeDevice(pK);
            GpuTransferHelper.FreeDevice(pV);
            GpuTransferHelper.FreeDevice(pT);
        }
    }

    /// <summary>The most partial-state workspace a decode with sequence lengths may ask for. A step of many rows over many
    /// sequences multiplies the states; past this the row declines and the call is refused, not run in pieces.</summary>
    private const long DecodeLengthsWorkspaceLimit = 256L << 20;

    /// <summary>The split geometry of a decode with sequence lengths: the contiguous decode's rule over the longest sequence.</summary>
    private (int Splits, int Chunk) DecodeLengthsSplits(in AttentionCall c)
    {
        int span = 1;
        foreach (int length in c.SequenceLengths!) span = Math.Max(span, Math.Min(length, c.KeyTokens));
        long blocks = (long)c.Batch * c.QueryTokens * c.KvHeads;
        int target = 2 * _context.MultiprocessorCount;
        int splits = (int)Math.Clamp((target + blocks - 1) / blocks, 1, Math.Min(64, Math.Max(1, span / 64)));
        int chunk = (span + splits - 1) / splits;
        return ((span + chunk - 1) / chunk, chunk);
    }

    /// <summary>The lengths row's admission (F.9). It takes only a call with sequence lengths, so the single-length decode rows
    /// keep every call they had: F32 head-major Q, a head-major cache whose keys and values are both F32, both F16 or both
    /// BF16, the <c>[B, S, Hq × D]</c> output, grouped heads the decode kernel is built for, and a workspace inside
    /// <see cref="DecodeLengthsWorkspaceLimit"/>.</summary>
    private bool DecodeLengthsAdmits(in AttentionCall c)
    {
        if (c.SequenceLengths is null || _kernels is not { HasDecodeLengths: true }) return false;
        int group = c.KvGroup <= 0 ? 1 : c.KvGroup;
        long rows = (long)c.Batch * c.QueryTokens, outputRows = rows * c.Heads;
        if (c.Query.Layout != AttentionLayout.HeadMajor || c.Key.Layout != AttentionLayout.HeadMajor || c.Value.Layout != AttentionLayout.HeadMajor
            || c.Output.Layout != AttentionLayout.TokenMajor || c.ValueHeadDim != c.QkHeadDim || c.Heads != c.KvHeads * group
            || c.Key.DType != c.Value.DType || !_kernels.DecodeLengthsSupports(c.QkHeadDim, group, c.Key.DType)
            || !CudaKernels.GridYzFits(c.KvHeads, rows) || outputRows > int.MaxValue)
            return false;
        return outputRows * DecodeLengthsSplits(in c).Splits * (c.QkHeadDim + 2) * sizeof(float) <= DecodeLengthsWorkspaceLimit;
    }

    /// <summary>Test hook: whether this device and PTX file serve a decode with sequence lengths at this head dim, group
    /// and cache type. Head dim 256 is left out where a block cannot have the shared memory it needs.</summary>
    internal bool DecodeLengthsServedForTest(int headDim, int group, DType cache) =>
        _kernels is { HasDecodeLengths: true } kernels && kernels.DecodeLengthsSupports(headDim, group, cache);

    /// <summary>A decode whose sequences have their own key lengths, with the LSE when the call names one (F.9).</summary>
    private unsafe void FlashAttentionDecodeLengths(in AttentionCall c)
    {
        Tensor output = c.Output.Storage, query = c.Query.Storage, key = c.Key.Storage, value = c.Value.Storage;
        Tensor? lse = c.Lse;
        int batch = c.Batch, rows = c.QueryTokens, hq = c.Heads, hkv = c.KvHeads, d = c.QkHeadDim, capacity = c.KeyTokens;
        int group = c.KvGroup <= 0 ? 1 : c.KvGroup;
        (int splits, int chunk) = DecodeLengthsSplits(in c);
        using OpScope _op = EnterOp(nameof(FlashAttention));
        EnsureKernels();
        using Tensor lengths = new(new TensorShape(batch), DType.I32);
        c.SequenceLengths.CopyTo(lengths.AsSpan<int>());
        ulong pQ = 0, pK = 0, pV = 0, pLengths = 0, pOut = 0, pLse = 0;
        bool cachedOutput = false, cachedLse = false;
        DecodePartials partials = default;
        try
        {
            pQ = GpuTransferHelper.CopyToDevice(query);
            pK = GpuTransferHelper.CopyToDevice(key);
            pV = GpuTransferHelper.CopyToDevice(value);
            // The lengths are the host's and change every step: they are uploaded for the call. A caller that issues this
            // for every layer of a step uploads the same array each time; before one does, the lengths should arrive as a
            // tensor its owner replaces, as the page table does.
            pLengths = GpuTransferHelper.CopyToDevice(lengths);
            AttentionEvidence.Observed(AttentionEvidenceKind.StagingCopy, "cuda.flash.decode.lengths", batch * sizeof(int));
            nuint outBytes = GpuTransferHelper.ByteSize(output), lseBytes = lse is null ? 0 : GpuTransferHelper.ByteSize(lse);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            if (lse is not null) pLse = GpuTransferHelper.AllocateDevice(lseBytes);
            partials = RentDecodePartials((long)batch * rows * hq, splits, d, "cuda.flash.decode.lengths.partials");
            _kernels!.LaunchFlashDecodeLengths(partials.M, partials.L, partials.Acc, pQ, pK, pV, pLengths, batch, rows, hq, d, hkv, capacity, group,
                c.Mask.Causal, c.Scale, splits, chunk, key.DType, _stream.Handle);
            _kernels.LaunchFlashDecodeLengthsCombine(pOut, pLse, partials.M, partials.L, partials.Acc, batch, rows, hq, d, splits, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
            if (lse is not null)
            {
                GpuTransferHelper.CacheActivation(lse, pLse, lseBytes);
                cachedLse = true;
            }
        }
        finally
        {
            ReturnDecodePartials(in partials);
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            if (!cachedLse && pLse != 0) GpuTransferHelper.FreeDevice(pLse);
            GpuTransferHelper.FreeDevice(pQ);
            GpuTransferHelper.FreeDevice(pK);
            GpuTransferHelper.FreeDevice(pV);
            GpuTransferHelper.FreeDevice(pLengths);
        }
    }

    /// <summary>True for a key/value cache pair the fused graph-decode scatter kernels can write: both F32, or both F16 with the F16 twins loaded.</summary>
    private bool KvScatterCacheOk(Tensor kCache, Tensor vCache) =>
        kCache.DType == vCache.DType && (kCache.DType == DType.F32 || (kCache.DType == DType.F16 && _kernels is { HasKvScatterF16: true }));
}
