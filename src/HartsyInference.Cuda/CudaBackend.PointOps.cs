using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda.Profiling;

namespace HartsyInference.Cuda;

// The per-element and per-point ops on the device: uniform fake quantization, the Fourier feature embedding, exp and reciprocal.
public sealed partial class CudaBackend
{
    /// <inheritdoc />
    public void FakeQuantizeUniform(Tensor output, Tensor input, float low, float high, int levels)
    {
        using NvtxRange _nvtx = NvtxRange.Push("FakeQuantizeUniform");
        long count = FakeQuantizeContract.Validate(output, input, low, high, levels);
        if (count == 0) return;
        using OpScope _op = EnterOp();
        EnsureKernels();
        ulong pIn = 0, pOut = 0;
        bool cached = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchFakeQuantizeUniform(pOut, pIn, low, high, levels, count, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cached = true;
        }
        finally
        {
            if (!cached) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    /// <inheritdoc />
    /// <remarks>The embedding stays resident for the projection that follows; the coordinates upload once per call.</remarks>
    public void FourierEmbed(Tensor output, Tensor coords, in FourierEmbedSpec spec)
    {
        using NvtxRange _nvtx = NvtxRange.Push("FourierEmbed");
        long count = FourierEmbedContract.Validate(output, coords, spec);
        if (count == 0) return;
        using OpScope _op = EnterOp();
        EnsureKernels();
        ulong pCoords = 0, pOut = 0;
        bool cached = false;
        try
        {
            pCoords = GpuTransferHelper.CopyToDevice(coords);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchFourierEmbed(pOut, pCoords, spec, count, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cached = true;
        }
        finally
        {
            if (!cached) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pCoords);
        }
    }

    /// <inheritdoc />
    public void Exp(Tensor output, Tensor input) => ElementwiseMath(output, input, ElementwiseMathFunction.Exp);

    /// <inheritdoc />
    public void Reciprocal(Tensor output, Tensor input) => ElementwiseMath(output, input, ElementwiseMathFunction.Reciprocal);

    private void ElementwiseMath(Tensor output, Tensor input, ElementwiseMathFunction function)
    {
        using NvtxRange _nvtx = NvtxRange.Push(function == ElementwiseMathFunction.Exp ? "Exp" : "Reciprocal");
        long count = ElementwiseMathContract.Validate(output, input, function);
        if (count == 0) return;
        using OpScope _op = EnterOp();
        EnsureKernels();
        ulong pIn = 0, pOut = 0;
        bool cached = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchElementwiseMath(pOut, pIn, input.DType, output.DType, function, count, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cached = true;
        }
        finally
        {
            if (!cached) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }
}
