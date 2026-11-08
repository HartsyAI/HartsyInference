using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda.Profiling;
using HartsyInference.Core.Backends.Gpu;
using HartsyInference.Core.Backends.Routing;
using HartsyInference.Core.Backends.Host;

namespace HartsyInference.Cuda;

// Mixture-of-experts routing, dispatch, combine, top-k and softplus on the device.
public sealed partial class CudaBackend
{
    /// <inheritdoc />
    public void Softplus(Tensor output, Tensor input)
    {
        using NvtxRange _nvtx = NvtxRange.Push("Softplus");
        if (input.DType != DType.F32 || output.DType != DType.F32)
            throw new NotSupportedException("CUDA Softplus supports F32 only.");
        if (input.ElementCount != output.ElementCount)
            throw new ArgumentException("Softplus input and output must have the same element count.");
        using OpScope _op = EnterOp();
        EnsureKernels();
        RequireMoeKernels();
        bool inPlace = ReferenceEquals(input, output);
        ulong pIn = 0, pOut = 0;
        bool cached = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint bytes = GpuTransferHelper.ByteSize(output);
            pOut = inPlace ? pIn : GpuTransferHelper.AllocateDevice(bytes);
            _kernels!.LaunchSoftplus(pOut, pIn, input.ElementCount, _stream.Handle);
            if (inPlace) { output._gpuSyncCallback = null; output._gpuDisposeCallback = null; }
            GpuTransferHelper.CacheActivation(output, pOut, bytes);
            cached = true;
        }
        finally
        {
            if (!cached && !inPlace) GpuTransferHelper.FreeDevice(pOut);
            if (!inPlace) GpuTransferHelper.FreeDevice(pIn);
        }
    }

    /// <inheritdoc />
    public void MoeRoute(Tensor topkIdx, Tensor topkWeight, Tensor logits, in MoeRouteArgs args,
        Tensor? bias = null, Tensor? altBias = null, Tensor? tokenKinds = null)
    {
        using NvtxRange _nvtx = NvtxRange.Push("MoeRoute");
        MoeReference.ValidateRoute(topkIdx, topkWeight, logits, args, bias, altBias, tokenKinds);
        using OpScope _op = EnterOp();
        EnsureKernels();
        RequireMoeKernels();
        int tokens = (int)(logits.ElementCount / args.NumExperts);
        ulong pL = 0, pB = 0, pA = 0, pK = 0, pIdx = 0, pW = 0;
        bool cachedIdx = false, cachedW = false;
        try
        {
            pL = GpuTransferHelper.CopyToDevice(logits);
            if (bias is not null) pB = GpuTransferHelper.CopyToDevice(bias);
            if (altBias is not null) pA = GpuTransferHelper.CopyToDevice(altBias);
            if (tokenKinds is not null) pK = GpuTransferHelper.CopyToDevice(tokenKinds);
            nuint idxBytes = GpuTransferHelper.ByteSize(topkIdx), wBytes = GpuTransferHelper.ByteSize(topkWeight);
            pIdx = GpuTransferHelper.AllocateDevice(idxBytes);
            pW = GpuTransferHelper.AllocateDevice(wBytes);
            _kernels!.LaunchMoeRoute(pIdx, pW, pL, pB, pA, pK, tokens, args, _stream.Handle);
            GpuTransferHelper.CacheActivation(topkIdx, pIdx, idxBytes);
            cachedIdx = true;
            GpuTransferHelper.CacheActivation(topkWeight, pW, wBytes);
            cachedW = true;
        }
        finally
        {
            if (!cachedIdx) GpuTransferHelper.FreeDevice(pIdx);
            if (!cachedW) GpuTransferHelper.FreeDevice(pW);
            GpuTransferHelper.FreeDevice(pL);
            if (pB != 0) GpuTransferHelper.FreeDevice(pB);
            if (pA != 0) GpuTransferHelper.FreeDevice(pA);
            if (pK != 0) GpuTransferHelper.FreeDevice(pK);
        }
    }

    /// <inheritdoc />
    public void MoeBuildDispatch(Tensor counts, Tensor offsets, Tensor permutedToken, Tensor pairSlot, Tensor topkIdx,
        int numExperts)
    {
        using NvtxRange _nvtx = NvtxRange.Push("MoeBuildDispatch");
        MoeReference.ValidateDispatch(counts, offsets, permutedToken, pairSlot, topkIdx, numExperts);
        using OpScope _op = EnterOp();
        EnsureKernels();
        RequireMoeKernels();
        int k = (int)topkIdx.Shape[topkIdx.Shape.Rank - 1];
        int pairs = (int)topkIdx.ElementCount;
        ulong pIdx = 0, pCounts = 0, pOffsets = 0, pPerm = 0, pSlot = 0;
        bool cCounts = false, cOffsets = false, cPerm = false, cSlot = false;
        try
        {
            pIdx = GpuTransferHelper.CopyToDevice(topkIdx);
            nuint countBytes = GpuTransferHelper.ByteSize(counts), offBytes = GpuTransferHelper.ByteSize(offsets);
            nuint permBytes = GpuTransferHelper.ByteSize(permutedToken), slotBytes = GpuTransferHelper.ByteSize(pairSlot);
            pCounts = GpuTransferHelper.AllocateDevice(countBytes);
            pOffsets = GpuTransferHelper.AllocateDevice(offBytes);
            pPerm = GpuTransferHelper.AllocateDevice(permBytes);
            pSlot = GpuTransferHelper.AllocateDevice(slotBytes);
            _kernels!.LaunchMoeBuildDispatch(pCounts, pOffsets, pPerm, pSlot, pIdx, pairs, k, numExperts, _stream.Handle);
            GpuTransferHelper.CacheActivation(counts, pCounts, countBytes); cCounts = true;
            GpuTransferHelper.CacheActivation(offsets, pOffsets, offBytes); cOffsets = true;
            GpuTransferHelper.CacheActivation(permutedToken, pPerm, permBytes); cPerm = true;
            GpuTransferHelper.CacheActivation(pairSlot, pSlot, slotBytes); cSlot = true;
        }
        finally
        {
            if (!cCounts) GpuTransferHelper.FreeDevice(pCounts);
            if (!cOffsets) GpuTransferHelper.FreeDevice(pOffsets);
            if (!cPerm) GpuTransferHelper.FreeDevice(pPerm);
            if (!cSlot) GpuTransferHelper.FreeDevice(pSlot);
            GpuTransferHelper.FreeDevice(pIdx);
        }
    }

    /// <inheritdoc />
    public void MoeCombine(Tensor output, Tensor expertOut, Tensor pairSlot, Tensor topkWeight, int k, bool accumulate)
    {
        using NvtxRange _nvtx = NvtxRange.Push("MoeCombine");
        MoeReference.ValidateCombine(output, expertOut, pairSlot, topkWeight, k);
        using OpScope _op = EnterOp();
        EnsureKernels();
        RequireMoeKernels();
        int hidden = (int)output.Shape[output.Shape.Rank - 1];
        int tokens = (int)(output.ElementCount / hidden);
        ulong pOut = 0, pX = 0, pSlot = 0, pW = 0;
        bool cached = false;
        try
        {
            nuint bytes = GpuTransferHelper.ByteSize(output);
            if (accumulate) pOut = GpuTransferHelper.CopyToDevice(output);
            else pOut = GpuTransferHelper.AllocateDevice(bytes);
            pX = GpuTransferHelper.CopyToDevice(expertOut);
            pSlot = GpuTransferHelper.CopyToDevice(pairSlot);
            pW = GpuTransferHelper.CopyToDevice(topkWeight);
            _kernels!.LaunchMoeCombine(pOut, pX, pSlot, pW, tokens, hidden, k, accumulate,
                (int)(expertOut.ElementCount / hidden), _stream.Handle);
            if (accumulate) { output._gpuSyncCallback = null; output._gpuDisposeCallback = null; }
            GpuTransferHelper.CacheActivation(output, pOut, bytes);
            cached = true;
        }
        finally
        {
            if (!cached) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pX);
            GpuTransferHelper.FreeDevice(pSlot);
            GpuTransferHelper.FreeDevice(pW);
        }
    }

    private void RequireMoeKernels()
    {
        if (!_kernels!.HasMoeKernels)
            throw new NotSupportedException("moe_route.ptx / moe_dispatch.ptx are not available on this device.");
    }
}
