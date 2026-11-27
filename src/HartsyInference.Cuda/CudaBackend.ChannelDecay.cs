using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda.Profiling;
using HartsyInference.Core.Backends.Gpu;
using HartsyInference.Core.Backends.Host;

namespace HartsyInference.Cuda;

// The per-key-channel delta-rule prefill (IBackend.SsmChannelDecayPrefill) on the device. Validates through its Core
// reference so the errors match the host default, and falls back to it when the module is absent or the key head is too wide.
public sealed partial class CudaBackend
{
    /// <inheritdoc />
    public void SsmChannelDecayPrefill(Tensor output, Tensor state, Tensor q, Tensor k, Tensor v, Tensor logDecay, Tensor betaRaw,
        int hv, int sv, int sk, int repeat)
    {
        using NvtxRange _nvtx = NvtxRange.Push("SsmChannelDecayPrefill");
        int seq = GatedDeltaNetReference.ValidateChannelDecayScan(output, state, q, k, v, logDecay, betaRaw, hv, sv, sk, repeat);
        using OpScope _op = EnterOp();
        EnsureKernels();
        if (!_kernels!.HasChannelDecayPrefillKernel || sk > CudaKernels.SsmPrefillMaxKeyDim)
        {
            GatedDeltaNetReference.ChannelDecayScan(output, state, q, k, v, logDecay, betaRaw, hv, sv, sk, repeat);
            return;
        }
        ulong pO = 0, pSt = 0, pQ = 0, pK = 0, pV = 0, pG = 0, pBe = 0;
        bool cached = false;
        try
        {
            pSt = GpuTransferHelper.CopyToDevice(state);
            pQ = GpuTransferHelper.CopyToDevice(q);
            pK = GpuTransferHelper.CopyToDevice(k);
            pV = GpuTransferHelper.CopyToDevice(v);
            pG = GpuTransferHelper.CopyToDevice(logDecay);
            pBe = GpuTransferHelper.CopyToDevice(betaRaw);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pO = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels.LaunchSsmChannelDecayPrefill(pO, pSt, pQ, pK, pV, pG, pBe, seq, hv, sv, sk, repeat, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pO, outBytes);
            // The recurrent state was advanced in place: keep it resident.
            GpuTransferHelper.CacheActivation(state, pSt, GpuTransferHelper.ByteSize(state));
            cached = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pQ); GpuTransferHelper.FreeDevice(pK); GpuTransferHelper.FreeDevice(pV);
            GpuTransferHelper.FreeDevice(pG); GpuTransferHelper.FreeDevice(pBe);
            if (!cached) { GpuTransferHelper.FreeDevice(pO); GpuTransferHelper.FreeDevice(pSt); }
        }
    }
}
