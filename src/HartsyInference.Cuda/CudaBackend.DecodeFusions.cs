using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda.Profiling;

namespace HartsyInference.Cuda;

// Graph-decode fusions (Kernels/lm/lm_decode_fused.cu, launched through CudaKernels.DecodeFusions.cs). Every fused op here is
// bit-identical to the composition it replaces; each has a numerics.* kill-switch in EngineKnobs.Numerics.
public sealed partial class CudaBackend
{
    /// <summary>The Q8_1 sidecar buffers an attention combine writes for the o-projection; all zero when it writes none.</summary>
    private readonly record struct AttnSidecar(ulong Xq, ulong Xd, ulong Xs);

    /// <summary>The fused add-RMSNorm kernels are bit-identical to Add followed by the norm, so the graph decode step may fold the
    /// residual add of layer i into layer i+1's input norm.</summary>
    public bool FoldsDecodeResidual => EngineKnobs.DecodeResidualFold.Value;

    /// <summary>Allocates the o-projection's Q8_1 sidecar for a single-row decode attention output when the combine kernel can emit it
    /// and the dp4a GEMV would consume it; otherwise returns the empty sidecar and the plain combine runs.</summary>
    private AttnSidecar TryAllocAttnCombineSidecar(int b, int tq, int hq, int d)
    {
        if (!EngineKnobs.AttnCombineQ8.Value || !_quantAtProducer || !EnableDp4aGemv || b != 1 || tq != 1
            || _kernels is null || !_kernels.HasFlashAttentionCombineQ8(d))
        {
            return default;
        }
        (ulong xq, ulong xd, ulong xs) = AllocSidecar(hq * d);
        return new AttnSidecar(xq, xd, xs);
    }

    private static void FreeAttnSidecar(AttnSidecar s)
    {
        GpuTransferHelper.FreeDevice(s.Xq);
        GpuTransferHelper.FreeDevice(s.Xd);
        GpuTransferHelper.FreeDevice(s.Xs);
    }

    public void QkNormFullRopeScatterDecodeStep(Tensor qOut, Tensor kCache, Tensor vCache, Tensor qk, Tensor? v,
        Tensor qNorm, Tensor kNorm, float eps,
        Tensor cosTable, Tensor sinTable, int hq, int hkv, int headDim, int rotaryDim, bool interleaved, ulong devicePos)
    {
        bool f16Kv = kCache.DType == DType.F16;
        if (!EngineKnobs.QknormFullScatter.Value || devicePos == 0 || qk.DType != DType.F32 || (v is not null && v.DType != DType.F32)
            || qNorm.DType != DType.F32 || kNorm.DType != DType.F32 || !KvScatterCacheOk(kCache, vCache))
        {
            HartsyInference.Core.Backends.IBackend.ComposeQkNormFullRopeScatter(this, qOut, kCache, vCache, qk, v, qNorm, kNorm, eps, cosTable, sinTable,
                hq, hkv, headDim, rotaryDim, interleaved, devicePos);
            return;
        }
        using NvtxRange _nvtx = NvtxRange.Push("QkNormFullRopeScatter");
        using OpScope _op = EnterOp();
        EnsureKernels();
        if (!_kernels!.HasQkNormFullRopeScatter(f16Kv))
        {
            HartsyInference.Core.Backends.IBackend.ComposeQkNormFullRopeScatter(this, qOut, kCache, vCache, qk, v, qNorm, kNorm, eps, cosTable, sinTable,
                hq, hkv, headDim, rotaryDim, interleaved, devicePos);
            return;
        }
        int maxSeq = (int)kCache.Shape[2];
        ulong pQk = 0, pVi = 0, pQ = 0;
        bool cachedOutput = false;
        try
        {
            pQk = GpuTransferHelper.CopyToDevice(qk);
            if (v is not null) pVi = GpuTransferHelper.CopyToDevice(v);
            ulong pQw = GpuTransferHelper.CopyToDevice(qNorm);
            ulong pKw = GpuTransferHelper.CopyToDevice(kNorm);
            ulong pCos = GpuTransferHelper.CopyToDevice(cosTable);
            ulong pSin = GpuTransferHelper.CopyToDevice(sinTable);
            ulong pK = GpuTransferHelper.CopyToDevice(kCache);
            ulong pV = GpuTransferHelper.CopyToDevice(vCache);
            nuint qBytes = GpuTransferHelper.ByteSize(qOut);
            pQ = GpuTransferHelper.AllocateDevice(qBytes);
            ulong kOff = pQk + (ulong)((long)hq * headDim * sizeof(float));
            ulong vOff = v is null ? kOff + (ulong)((long)hkv * headDim * sizeof(float)) : pVi;
            _kernels.LaunchQkNormFullRopeScatter(f16Kv, pQ, pK, pV, pQk, kOff, vOff, pQw, pKw, pCos, pSin,
                hq, hkv, headDim, rotaryDim, interleaved, eps, maxSeq, devicePos, _stream.Handle);
            kCache._gpuSyncCallback = null;
            kCache._gpuDisposeCallback = null;
            GpuTransferHelper.CacheActivation(kCache, pK, GpuTransferHelper.ByteSize(kCache));
            vCache._gpuSyncCallback = null;
            vCache._gpuDisposeCallback = null;
            GpuTransferHelper.CacheActivation(vCache, pV, GpuTransferHelper.ByteSize(vCache));
            GpuTransferHelper.CacheActivation(qOut, pQ, qBytes);
            cachedOutput = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pQk);
            if (pVi != 0) GpuTransferHelper.FreeDevice(pVi);
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pQ);
        }
    }

    public void MoeCombineAddRmsNormEmitQ8(Tensor residOut, Tensor normOut, Tensor a, Tensor slotOut, Tensor topkWeight, Tensor? shared,
        Tensor? sharedGateLogit, int topk, Tensor weight, float eps)
    {
        int normDim = (int)weight.ElementCount;
        EnsureKernels();
        bool fused = EngineKnobs.DecodeResidualFold.Value && _quantAtProducer && EnableDp4aGemv
            && a.ElementCount == normDim && residOut.ElementCount == normDim && normOut.ElementCount == normDim
            && slotOut.ElementCount == (long)topk * normDim && topkWeight.ElementCount == topk
            && (shared is null || shared.ElementCount == normDim) && (sharedGateLogit is null || sharedGateLogit.ElementCount == 1)
            && a.DType == DType.F32 && slotOut.DType == DType.F32 && topkWeight.DType == DType.F32 && weight.DType == DType.F32
            && residOut.DType == DType.F32 && normOut.DType == DType.F32
            && (shared is null || shared.DType == DType.F32) && (sharedGateLogit is null || sharedGateLogit.DType == DType.F32)
            && _kernels!.HasMoeCombineAddRmsNormQ8(normDim);
        if (!fused)
        {
            using Tensor combined = new(a.Shape, DType.F32);
            MoeCombineSlots(combined, slotOut, topkWeight, shared, sharedGateLogit, topk);
            AddRmsNormEmitQ8(residOut, normOut, a, combined, weight, eps);
            return;
        }
        using NvtxRange _nvtx = NvtxRange.Push("MoeCombineAddRmsNormQ8");
        using OpScope _op = EnterOp();
        ulong pA = 0, pSlots = 0, pW = 0, pShared = 0, pGate = 0, pNw = 0, pResid = 0, pNorm = 0, xq = 0, xd = 0, xs = 0;
        bool cachedOutput = false;
        try
        {
            pA = GpuTransferHelper.CopyToDevice(a);
            pSlots = GpuTransferHelper.CopyToDevice(slotOut);
            pW = GpuTransferHelper.CopyToDevice(topkWeight);
            if (shared is not null) pShared = GpuTransferHelper.CopyToDevice(shared);
            if (sharedGateLogit is not null) pGate = GpuTransferHelper.CopyToDevice(sharedGateLogit);
            pNw = GpuTransferHelper.CopyToDevice(weight);
            nuint outBytes = GpuTransferHelper.ByteSize(residOut);
            pResid = GpuTransferHelper.AllocateDevice(outBytes);
            pNorm = GpuTransferHelper.AllocateDevice(outBytes);
            (xq, xd, xs) = AllocSidecar(normDim);
            _kernels!.LaunchMoeCombineAddRmsNormQ8(pResid, pNorm, xq, xd, xs, pA, pSlots, pW, pShared, pGate, topk, pNw, normDim, eps,
                _stream.Handle);
            GpuTransferHelper.CacheActivation(residOut, pResid, outBytes);
            GpuTransferHelper.CacheActivation(normOut, pNorm, outBytes);
            GpuTransferHelper.RegisterSidecar(normOut, xq, xd, xs, normDim);
            cachedOutput = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pA);
            GpuTransferHelper.FreeDevice(pSlots);
            GpuTransferHelper.FreeDevice(pW);
            if (pShared != 0) GpuTransferHelper.FreeDevice(pShared);
            if (pGate != 0) GpuTransferHelper.FreeDevice(pGate);
            if (!cachedOutput)
            {
                GpuTransferHelper.FreeDevice(pResid);
                GpuTransferHelper.FreeDevice(pNorm);
                GpuTransferHelper.FreeDevice(xq);
                GpuTransferHelper.FreeDevice(xd);
                GpuTransferHelper.FreeDevice(xs);
            }
        }
    }
}
