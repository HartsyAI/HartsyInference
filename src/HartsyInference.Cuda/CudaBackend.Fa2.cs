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
}
