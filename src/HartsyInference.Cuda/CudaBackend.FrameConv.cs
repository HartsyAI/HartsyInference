using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda.Profiling;

namespace HartsyInference.Cuda;

// IBackend.DepthwiseConvFrames on the device: one kernel over the token-major data, no transposes.
public sealed partial class CudaBackend
{
    /// <inheritdoc cref="IBackend.DepthwiseConvFrames"/>
    public void DepthwiseConvFrames(Tensor output, Tensor input, Tensor weight, int frames, int spatial, int channels, int taps, int pad)
    {
        DepthwiseConvFramesReference.Validate(output, input, weight, frames, spatial, channels, taps, pad);
        using NvtxRange _nvtx = NvtxRange.Push("DepthwiseConvFrames");
        using OpScope _op = EnterOp();
        EnsureKernels();
        if (!_kernels!.HasFrameConvKernels)
            throw new NotSupportedException("DepthwiseConvFrames needs frame_conv.ptx in the Ptx folder.");

        ulong pIn = 0, pW = 0, pOut = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            pW = GpuTransferHelper.CopyToDevice(weight);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels.LaunchDepthwiseConvFrames(pOut, pIn, pW, frames, spatial, channels, taps, pad, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pIn);
            GpuTransferHelper.FreeDevice(pW);
            if (!cachedOutput && pOut != 0) GpuTransferHelper.FreeDevice(pOut);
        }
    }
}
