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
    private bool DecodeGqaEligible(int tq, bool causal, int hq, int hkv, int kvGroup, int d) =>
        tq == 1 && causal && hkv > 0 && hq == hkv * kvGroup && _kernels is { HasDecodeGqa: true } && CudaKernels.DecodeGqaSupports(d, kvGroup)
        && EngineKnobs.FlashDecodeGqa.Value;

    /// <summary>Grouped-query flash-decoding into <paramref name="pOut"/>: partial states per (split, KV head), then the shared combine.
    /// With <paramref name="devicePos"/> the split count follows the cache capacity <paramref name="keySpan"/> so a captured graph keeps its grid.</summary>
    private void RunFlashDecodeGqa(ulong pOut, ulong pQ, ulong pK, ulong pV, int b, int hq, int hkv, int keyStride, int keySpan, int d, int kvLen,
        int kvGroup, int qOffset, float scale, float softcap, int window, ulong devicePos, bool f16Kv)
    {
        // One wave: the kernel's ~42 KB of shared memory leaves room for two blocks per SM, so more blocks than that run a half-empty second wave.
        int target = 2 * _context.MultiprocessorCount;
        int splits = Math.Clamp((target + b * hkv - 1) / (b * hkv), 1, Math.Min(64, Math.Max(1, keySpan / 64)));
        int chunk = (keySpan + splits - 1) / splits;
        splits = (keySpan + chunk - 1) / chunk;
        long n = (long)b * hq;
        ulong pM = 0, pL = 0, pAcc = 0;
        try
        {
            pM = GpuTransferHelper.AllocateDevice((nuint)(n * splits * sizeof(float)));
            pL = GpuTransferHelper.AllocateDevice((nuint)(n * splits * sizeof(float)));
            pAcc = GpuTransferHelper.AllocateDevice((nuint)(n * splits * d * sizeof(float)));
            _kernels!.LaunchFlashDecodeGqa(pM, pL, pAcc, pQ, pK, pV, b, hq, d, hkv, keyStride, kvLen, kvGroup, qOffset, scale, softcap, window,
                splits, chunk, devicePos, f16Kv, _stream.Handle);
            _kernels.LaunchFlashAttentionCombine(pOut, pM, pL, pAcc, b, hq, 1, d, splits, _stream.Handle);
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pM);
            GpuTransferHelper.FreeDevice(pL);
            GpuTransferHelper.FreeDevice(pAcc);
        }
    }

    /// <summary>True for a key/value cache pair the fused graph-decode scatter kernels can write: both F32, or both F16 with the F16 twins loaded.</summary>
    private bool KvScatterCacheOk(Tensor kCache, Tensor vCache) =>
        kCache.DType == vCache.DType && (kCache.DType == DType.F32 || (kCache.DType == DType.F16 && _kernels is { HasKvScatterF16: true }));

    private void LaunchQkvRopeScatterAny(bool f16Kv, ulong qOut, ulong kCache, ulong vCache, ulong qIn, ulong kIn, ulong vIn, ulong cos, ulong sin,
        int nq, int nkv, int headDim, int rotaryDim, bool interleaved, int maxSeq, ulong devicePos, nint stream)
    {
        if (f16Kv) _kernels!.LaunchQkvRopeScatterF16Kv(qOut, kCache, vCache, qIn, kIn, vIn, cos, sin, nq, nkv, headDim, rotaryDim, interleaved, maxSeq, devicePos, stream);
        else _kernels!.LaunchQkvRopeScatter(qOut, kCache, vCache, qIn, kIn, vIn, cos, sin, nq, nkv, headDim, rotaryDim, interleaved, maxSeq, devicePos, stream);
    }

    private void LaunchQkNormRopeScatterAny(bool f16Kv, ulong qOut, ulong kCache, ulong vCache, ulong qIn, ulong kIn, ulong vIn, ulong qNormW, ulong kNormW,
        ulong cos, ulong sin, int nq, int nkv, int headDim, int rotaryDim, bool interleaved, float eps, int maxSeq, ulong devicePos, nint stream)
    {
        if (f16Kv) _kernels!.LaunchQkNormRopeScatterF16Kv(qOut, kCache, vCache, qIn, kIn, vIn, qNormW, kNormW, cos, sin, nq, nkv, headDim, rotaryDim, interleaved, eps, maxSeq, devicePos, stream);
        else _kernels!.LaunchQkNormRopeScatter(qOut, kCache, vCache, qIn, kIn, vIn, qNormW, kNormW, cos, sin, nq, nkv, headDim, rotaryDim, interleaved, eps, maxSeq, devicePos, stream);
    }
}
