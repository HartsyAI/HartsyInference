using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda.Profiling;

namespace HartsyInference.Cuda;

// IBackend.DepthwiseConvGlu on the device: one fused kernel over the channels-last activations.
public sealed partial class CudaBackend
{
    /// <inheritdoc cref="IBackend.DepthwiseConvGlu"/>
    public void DepthwiseConvGlu(Tensor output, Tensor input, Tensor taps, Tensor? bias, int frames, int height, int width, int kernel,
        bool inputSilu)
    {
        int hidden = DepthwiseConvGluReference.Validate(output, input, taps, bias, frames, height, width, kernel);
        using NvtxRange _nvtx = NvtxRange.Push("DepthwiseConvGlu");
        using OpScope _op = EnterOp();
        EnsureKernels();
        if (!_kernels!.HasDepthwiseGluKernels)
            throw new NotSupportedException("DepthwiseConvGlu needs depthwise_glu.ptx in the Ptx folder.");

        ulong pIn = 0, pW = 0, pB = 0, pOut = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            pW = GpuTransferHelper.CopyToDevice(taps);
            pB = bias is null ? 0 : GpuTransferHelper.CopyToDevice(bias);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels.LaunchDepthwiseGlu(pOut, pIn, pW, pB, input.DType == DType.BF16, frames, height, width, hidden, kernel, inputSilu,
                _stream.Handle);

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pIn);
            GpuTransferHelper.FreeDevice(pW);
            if (pB != 0) GpuTransferHelper.FreeDevice(pB);
            if (!cachedOutput && pOut != 0) GpuTransferHelper.FreeDevice(pOut);
        }
    }
}
