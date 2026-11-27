using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda.Profiling;
using HartsyInference.Core.Backends.Gpu;

namespace HartsyInference.Cuda;

// The Mamba-1 selective scan (IBackend.SelectiveScan) on the device. Validates through SelectiveScanReference so the errors
// match the host default, and falls back to it when the kernel module is absent or the state count is above the kernel's.
public sealed partial class CudaBackend
{
    /// <inheritdoc />
    public void SelectiveScan(Tensor output, Tensor u, Tensor delta, Tensor a, Tensor b, Tensor c, Tensor d, Tensor? z, Tensor? state,
        int rows, int frames, int channels, int states)
    {
        using NvtxRange _nvtx = NvtxRange.Push("SelectiveScan");
        SelectiveScanReference.Validate(output, u, delta, a, b, c, d, z, state, rows, frames, channels, states);
        using OpScope _op = EnterOp();
        EnsureKernels();
        if (!_kernels!.HasSelectiveScanKernel || states > CudaKernels.SelectiveScanMaxStates)
        {
            SelectiveScanReference.Apply(output, u, delta, a, b, c, d, z, state, rows, frames, channels, states);
            return;
        }
        ulong pO = 0, pU = 0, pDelta = 0, pA = 0, pB = 0, pC = 0, pD = 0, pZ = 0, pSt = 0;
        bool cached = false;
        try
        {
            pU = GpuTransferHelper.CopyToDevice(u);
            pDelta = GpuTransferHelper.CopyToDevice(delta);
            pA = GpuTransferHelper.CopyToDevice(a);
            pB = GpuTransferHelper.CopyToDevice(b);
            pC = GpuTransferHelper.CopyToDevice(c);
            pD = GpuTransferHelper.CopyToDevice(d);
            if (z is not null) pZ = GpuTransferHelper.CopyToDevice(z);
            if (state is not null) pSt = GpuTransferHelper.CopyToDevice(state);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            // A fresh output buffer: the kernel reads u and writes the output in different threads, so they must not alias.
            pO = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels.LaunchSsmSelectiveScan(pO, pU, pDelta, pA, pB, pC, pD, pZ, pSt, z is not null, state is not null,
                rows, frames, channels, states, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pO, outBytes);
            if (state is not null)
            {
                // The carried state was advanced in place: keep it resident.
                GpuTransferHelper.CacheActivation(state, pSt, GpuTransferHelper.ByteSize(state));
            }
            cached = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pU); GpuTransferHelper.FreeDevice(pDelta); GpuTransferHelper.FreeDevice(pA);
            GpuTransferHelper.FreeDevice(pB); GpuTransferHelper.FreeDevice(pC); GpuTransferHelper.FreeDevice(pD);
            GpuTransferHelper.FreeDevice(pZ);
            if (!cached) { GpuTransferHelper.FreeDevice(pO); GpuTransferHelper.FreeDevice(pSt); }
        }
    }
}
