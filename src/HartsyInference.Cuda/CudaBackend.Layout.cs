using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda.Profiling;

namespace HartsyInference.Cuda;

// Generic layout moves: strided copy, strided accumulate and pad, validated through the Core contracts so the
// kernels and the host reference refuse the same calls.
public sealed partial class CudaBackend
{
    /// <inheritdoc />
    public void CopyStrided(Tensor output, Tensor input, in StridedCopySpec spec)
    {
        using NvtxRange _nvtx = NvtxRange.Push("CopyStrided");
        StridedCopyGeometry g = StridedCopyContract.ValidateCopy(output, input, spec);
        using OpScope _op = EnterOp();
        EnsureKernels();
        // A destination covering the whole output gets a fresh buffer. A scatter into part of it keeps the rest, so the
        // output's current contents come along: its cached device buffer when it has one, else an upload.
        RunLayout(output, input, fresh: g.DestinationCoversOutput, LayoutOp.Copy, g, default);
    }

    /// <inheritdoc />
    public void AccumulateStrided(Tensor output, Tensor input, in StridedCopySpec spec)
    {
        using NvtxRange _nvtx = NvtxRange.Push("AccumulateStrided");
        StridedCopyGeometry g = StridedCopyContract.ValidateAccumulate(output, input, spec);
        using OpScope _op = EnterOp();
        EnsureKernels();
        RunLayout(output, input, fresh: false, LayoutOp.Accumulate, g, default);
    }

    /// <inheritdoc />
    public void Pad(Tensor output, Tensor input, in PadSpec spec)
    {
        using NvtxRange _nvtx = NvtxRange.Push("Pad");
        PadGeometry g = PadContract.Validate(output, input, spec);
        using OpScope _op = EnterOp();
        EnsureKernels();
        RunLayout(output, input, fresh: true, LayoutOp.Pad, default, g);
    }

    private enum LayoutOp
    {
        Copy,
        Accumulate,
        Pad,
    }

    /// <summary>Stages the operands, launches, and binds the output buffer; <paramref name="fresh"/> allocates the output
    /// instead of carrying its current contents to the device. No closure: the decode path repeats K/V heads every token.</summary>
    private void RunLayout(Tensor output, Tensor input, bool fresh, LayoutOp op, in StridedCopyGeometry strided, in PadGeometry pad)
    {
        ulong pIn = 0, pOut = 0;
        bool bound = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = fresh ? GpuTransferHelper.AllocateDevice(outBytes) : GpuTransferHelper.CopyToDevice(output);
            switch (op)
            {
                case LayoutOp.Copy: _kernels!.LaunchCopyStrided(pOut, pIn, strided, _stream.Handle); break;
                case LayoutOp.Accumulate: _kernels!.LaunchAccumulateStrided(pOut, pIn, strided, output.DType, input.DType, _stream.Handle); break;
                default: _kernels!.LaunchPad(pOut, pIn, pad, _stream.Handle); break;
            }
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            bound = true;
        }
        finally
        {
            // A cached pointer is left alone by FreeDevice; a fresh or transiently uploaded one is released on failure.
            if (!bound) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }
}
