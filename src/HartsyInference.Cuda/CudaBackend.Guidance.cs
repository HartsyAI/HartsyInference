using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Backends.Guidance;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda.Profiling;

namespace HartsyInference.Cuda;

// The guided Euler step and the guided top-K logits on the device.
public sealed partial class CudaBackend
{
    /// <inheritdoc />
    /// <remarks>One launch without a norm or with the per-row norm; the global norm reduces in three (partial sums, the ratio, the
    /// scaled update) through a small scratch buffer. The latent stays resident in place.</remarks>
    public void GuidedEulerStep(Tensor z, Tensor cond, Tensor uncond, in GuidedEulerSpec spec)
    {
        using NvtxRange _nvtx = NvtxRange.Push("GuidedEulerStep");
        GuidedEulerGeometry g = GuidedEulerContract.Validate(z, cond, uncond, spec);
        if (spec.Delta == 0f) return;
        using OpScope _op = EnterOp();
        EnsureKernels();
        ulong pZ = 0, pCond = 0, pUncond = 0, pScratch = 0;
        bool cachedZ = false;
        try
        {
            pZ = GpuTransferHelper.CopyToDevice(z);
            pCond = GpuTransferHelper.CopyToDevice(cond);
            pUncond = ReferenceEquals(cond, uncond) ? pCond : GpuTransferHelper.CopyToDevice(uncond);
            switch (spec.Norm.Kind)
            {
                case GuidanceNormKind.None:
                    _kernels!.LaunchGuidedEuler(pZ, pCond, pUncond, spec, g.Count, _stream.Handle);
                    break;
                case GuidanceNormKind.Global:
                    int partials = CudaKernels.GuidedEulerPartials(g.Count);
                    pScratch = GpuTransferHelper.AllocateDevice(CudaKernels.GuidedEulerScratchBytes(partials));
                    _kernels!.LaunchGuidedEulerGlobal(pZ, pCond, pUncond, pScratch, partials, spec, g.Count, _stream.Handle);
                    break;
                default:
                    _kernels!.LaunchGuidedEulerRows(pZ, pCond, pUncond, spec, g.Rows, g.Width, _stream.Handle);
                    break;
            }
            // In place on z: clear the stale callbacks before re-caching so the old sync callback cannot free the buffer kept resident.
            z._gpuSyncCallback = null;
            z._gpuDisposeCallback = null;
            GpuTransferHelper.CacheActivation(z, pZ, GpuTransferHelper.ByteSize(z));
            cachedZ = true;
        }
        finally
        {
            if (!cachedZ) GpuTransferHelper.FreeDevice(pZ);
            if (pUncond != pCond) GpuTransferHelper.FreeDevice(pUncond);
            GpuTransferHelper.FreeDevice(pCond);
            if (pScratch != 0) GpuTransferHelper.FreeDevice(pScratch);
        }
    }

    /// <inheritdoc />
    /// <remarks>One block per row; a row wider than <see cref="CudaKernels.TopKGuidedMaxWidth"/> is refused rather than run on the host.</remarks>
    public void TopKGuidedLogits(Tensor output, Tensor cond, Tensor uncond, int topK, float guidance)
    {
        using NvtxRange _nvtx = NvtxRange.Push("TopKGuidedLogits");
        (long rows, int width) = TopKGuidedContract.Validate(output, cond, uncond, topK, guidance);
        using OpScope _op = EnterOp();
        EnsureKernels();
        ulong pCond = 0, pUncond = 0, pOut = 0;
        bool cached = false;
        try
        {
            pCond = GpuTransferHelper.CopyToDevice(cond);
            pUncond = ReferenceEquals(cond, uncond) ? pCond : GpuTransferHelper.CopyToDevice(uncond);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchTopKGuidedLogits(pOut, pCond, pUncond, rows, width, topK, guidance, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cached = true;
        }
        finally
        {
            if (!cached) GpuTransferHelper.FreeDevice(pOut);
            if (pUncond != pCond) GpuTransferHelper.FreeDevice(pUncond);
            GpuTransferHelper.FreeDevice(pCond);
        }
    }
}
