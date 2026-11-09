using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Backends.Decode;
using HartsyInference.Core.Backends.Rotary;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda.Profiling;

namespace HartsyInference.Cuda;

// The graph-decode QKV epilogue: one fused kernel per norm kind (lm_f32.cu and kv_scatter_f16.cu for none and per-head,
// lm_decode_fused.cu for full width), each bit-identical to the composed ops it replaces. numerics.qkvDecodeFused=0, a
// cache dtype the kernels do not serve or a missing full-width kernel composes instead.
public sealed partial class CudaBackend
{
    /// <inheritdoc />
    public void QkvDecodeEpilogue(in QkvDecodeCall call)
    {
        QkvDecodeContract.Validate(call);
        EnsureKernels();
        bool f16Kv = call.KCache.DType == DType.F16;
        bool fused = EngineKnobs.QkvDecodeFused.Value && KvScatterCacheOk(call.KCache, call.VCache)
            && (call.Norm != QkvNormKind.FullWidth || _kernels!.HasQkNormFullRopeScatter(f16Kv));
        if (!fused)
        {
            IBackend.ComposeQkvDecodeEpilogue(this, call);
            return;
        }
        using NvtxRange _nvtx = NvtxRange.Push("QkvDecodeEpilogue");
        using OpScope _op = EnterOp();
        int maxSeq = (int)call.CacheLength;
        bool interleaved = call.Pairing == RotaryPairing.Interleaved;
        // The sections may share one projection tensor; each distinct tensor is staged once and released once.
        ulong pQT = 0, pKT = 0, pVT = 0, pQ = 0;
        bool cachedOutput = false;
        try
        {
            pQT = GpuTransferHelper.CopyToDevice(call.Q.Tensor);
            pKT = ReferenceEquals(call.K.Tensor, call.Q.Tensor) ? pQT : GpuTransferHelper.CopyToDevice(call.K.Tensor);
            pVT = ReferenceEquals(call.V.Tensor, call.Q.Tensor) ? pQT : ReferenceEquals(call.V.Tensor, call.K.Tensor) ? pKT : GpuTransferHelper.CopyToDevice(call.V.Tensor);
            ulong qIn = pQT + (ulong)(call.Q.Offset * sizeof(float));
            ulong kIn = pKT + (ulong)(call.K.Offset * sizeof(float));
            ulong vIn = pVT + (ulong)(call.V.Offset * sizeof(float));
            // Tables, norm weights and caches are resident for the whole decode, so their device copies stay cached.
            ulong pCos = GpuTransferHelper.CopyToDevice(call.Cos);
            ulong pSin = GpuTransferHelper.CopyToDevice(call.Sin);
            ulong pK = GpuTransferHelper.CopyToDevice(call.KCache);
            ulong pV = GpuTransferHelper.CopyToDevice(call.VCache);
            nuint qBytes = GpuTransferHelper.ByteSize(call.QOut);
            pQ = GpuTransferHelper.AllocateDevice(qBytes);
            if (call.Norm == QkvNormKind.None)
            {
                if (f16Kv) _kernels!.LaunchQkvRopeScatterF16Kv(pQ, pK, pV, qIn, kIn, vIn, pCos, pSin, call.Heads, call.KvHeads, call.HeadDim, call.RotaryDim, interleaved, maxSeq, call.DevicePosition, _stream.Handle);
                else _kernels!.LaunchQkvRopeScatter(pQ, pK, pV, qIn, kIn, vIn, pCos, pSin, call.Heads, call.KvHeads, call.HeadDim, call.RotaryDim, interleaved, maxSeq, call.DevicePosition, _stream.Handle);
            }
            else
            {
                ulong pQw = GpuTransferHelper.CopyToDevice(call.QNormWeight!);
                ulong pKw = GpuTransferHelper.CopyToDevice(call.KNormWeight!);
                if (call.Norm == QkvNormKind.FullWidth)
                    _kernels!.LaunchQkNormFullRopeScatter(f16Kv, pQ, pK, pV, qIn, kIn, vIn, pQw, pKw, pCos, pSin, call.Heads, call.KvHeads, call.HeadDim, call.RotaryDim, interleaved, call.Eps, maxSeq, call.DevicePosition, _stream.Handle);
                else if (f16Kv)
                    _kernels!.LaunchQkNormRopeScatterF16Kv(pQ, pK, pV, qIn, kIn, vIn, pQw, pKw, pCos, pSin, call.Heads, call.KvHeads, call.HeadDim, call.RotaryDim, interleaved, call.Eps, maxSeq, call.DevicePosition, _stream.Handle);
                else
                    _kernels!.LaunchQkNormRopeScatter(pQ, pK, pV, qIn, kIn, vIn, pQw, pKw, pCos, pSin, call.Heads, call.KvHeads, call.HeadDim, call.RotaryDim, interleaved, call.Eps, maxSeq, call.DevicePosition, _stream.Handle);
            }
            // The caches were written in place: keep them resident without a host sync, as KvCacheAppendDev does.
            call.KCache._gpuSyncCallback = null;
            call.KCache._gpuDisposeCallback = null;
            GpuTransferHelper.CacheActivation(call.KCache, pK, GpuTransferHelper.ByteSize(call.KCache));
            call.VCache._gpuSyncCallback = null;
            call.VCache._gpuDisposeCallback = null;
            GpuTransferHelper.CacheActivation(call.VCache, pV, GpuTransferHelper.ByteSize(call.VCache));
            GpuTransferHelper.CacheActivation(call.QOut, pQ, qBytes);
            cachedOutput = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pQT);
            if (pKT != pQT) GpuTransferHelper.FreeDevice(pKT);
            if (pVT != pQT && pVT != pKT) GpuTransferHelper.FreeDevice(pVT);
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pQ);
        }
    }
}
