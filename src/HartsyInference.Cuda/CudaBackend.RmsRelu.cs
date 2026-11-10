using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda.Profiling;

namespace HartsyInference.Cuda;

// IBackend.RmsNormRelu on the device: one fused kernel for F32 rows; anything else composes the default ops.
public sealed partial class CudaBackend
{
    /// <inheritdoc cref="IBackend.RmsNormRelu"/>
    public void RmsNormRelu(Tensor output, Tensor input, Tensor weight, float eps, float scale)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(weight);
        EnsureKernels();
        long width = weight.ElementCount;
        bool fused = _kernels!.HasRmsReluKernels && input.DType == DType.F32 && output.DType == DType.F32 && weight.DType == DType.F32
            && width > 0 && input.ElementCount % width == 0 && output.ElementCount == input.ElementCount;
        if (!fused)
        {
            IBackend self = this;
            self.RmsNorm(output, input, weight, eps);
            self.LeakyRelu(output, output, 0f);
            if (scale != 1f)
                self.Scale(output, output, scale);
            return;
        }

        using NvtxRange _nvtx = NvtxRange.Push("RmsNormRelu");
        using OpScope _op = EnterOp();
        ulong pIn = 0, pW = 0, pOut = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            pW = GpuTransferHelper.CopyToDevice(weight);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels.LaunchRmsNormRelu(pOut, pIn, pW, checked((int)(input.ElementCount / width)), (int)width, eps, scale, _stream.Handle);
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
