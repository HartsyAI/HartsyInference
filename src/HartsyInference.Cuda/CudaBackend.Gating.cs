using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Backends.Gating;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda.Profiling;

namespace HartsyInference.Cuda;

// The gated linear unit, the group gate and the fused linear prologue and epilogue on the device.
public sealed partial class CudaBackend
{
    /// <inheritdoc />
    public void Glu(Tensor output, Tensor input, in GluSpec spec)
    {
        using NvtxRange _nvtx = NvtxRange.Push("Glu");
        (int inner, long rows) = GluContract.Validate(output, input, spec);
        GluContract.RequireBakedSitu(spec, "CUDA");
        if (rows == 0) return;
        using OpScope _op = EnterOp();
        EnsureKernels();
        ulong pIn = 0, pOut = 0;
        bool cached = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchGlu(pOut, pIn, input.DType, spec, rows, inner, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cached = true;
        }
        finally
        {
            if (!cached) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    /// <summary>The decode-step GLU with the Q8_1 sidecar of its output: one F32 row of whole 32-blocks through SiLU or the tanh GELU over
    /// <c>[gate | up]</c>, which is the shape <c>lm_glu_act_q8_1_f32</c> quantizes in its own registers; every other call is a plain <see cref="Glu"/>.</summary>
    public void GluEmitQ8(Tensor output, Tensor input, in GluSpec spec)
    {
        (int inner, long rows) = GluContract.Validate(output, input, spec);
        bool sidecar = _quantAtProducer && EnableDp4aGemv && rows == 1 && inner % 32 == 0 && input.DType == DType.F32
            && spec.Gate == GluGateHalf.First && spec.Activation is GluActivation.Silu or GluActivation.GeluTanh;
        if (!sidecar)
        {
            Glu(output, input, spec);
            return;
        }
        using NvtxRange _nvtx = NvtxRange.Push("GluActivateQ8");
        using OpScope _op = EnterOp();
        EnsureKernels();
        ulong pOut = 0, pIn = 0, xq = 0, xd = 0, xs = 0;
        bool cached = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            (xq, xd, xs) = AllocSidecar(inner);
            _kernels!.LaunchGluActQ8(pOut, xq, xd, xs, pIn, 1, inner, spec.Activation == GluActivation.GeluTanh, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            GpuTransferHelper.RegisterSidecar(output, xq, xd, xs, inner);
            cached = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pIn);
            if (!cached)
            {
                GpuTransferHelper.FreeDevice(pOut);
                GpuTransferHelper.FreeDevice(xq);
                GpuTransferHelper.FreeDevice(xd);
                GpuTransferHelper.FreeDevice(xs);
            }
        }
    }

    /// <inheritdoc />
    public void GroupScale(Tensor output, Tensor input, Tensor gates, in GroupScaleSpec spec)
    {
        using NvtxRange _nvtx = NvtxRange.Push("GroupScale");
        GroupScaleGeometry g = GroupScaleContract.Validate(output, input, gates, spec);
        if (g.Rows == 0) return;
        using OpScope _op = EnterOp();
        EnsureKernels();
        bool inPlace = ReferenceEquals(output, input);
        ulong pIn = 0, pGates = 0, pOut = 0;
        bool cached = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            pGates = GpuTransferHelper.CopyToDevice(gates);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            // An accumulate reads the output's current contents; in place the input's buffer is the output's.
            pOut = inPlace ? pIn : spec.Accumulate ? GpuTransferHelper.CopyToDevice(output) : GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchGroupScale(pOut, pIn, pGates, input.DType, gates.DType, spec, g, _stream.Handle);
            if (inPlace || spec.Accumulate)
            {
                // Written in place: keep the tensor resident on the same buffer without a host sync.
                output._gpuSyncCallback = null;
                output._gpuDisposeCallback = null;
            }
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cached = true;
        }
        finally
        {
            if (!cached && !inPlace) GpuTransferHelper.FreeDevice(pOut);
            if (!inPlace) GpuTransferHelper.FreeDevice(pIn);
            GpuTransferHelper.FreeDevice(pGates);
        }
    }

    /// <summary>A projection with its fusion folded into the resident int8 GEMM where that path serves it: the tanh GELU in the dequant
    /// epilogue, and the doubled-sigmoid head gate of an F16 input in the activation quantize that precedes the GEMM. Anything else
    /// runs the composition.</summary>
    public unsafe void LinearFused(Tensor output, Tensor input, Tensor weight, Tensor? bias, in LinearFusion fusion)
    {
        bool int8 = weight.LowRankAdjunct is not { IsActive: true } && weight.DType == DType.I8 && weight.QuantInfo is QuantWeightInfo qi
            && CanRunResidentInt8(output, input, weight, qi, 0, -1);
        if (fusion.PrologueGates is null)
        {
            if (fusion.Epilogue != LinearActivation.GeluTanh || !int8)
            {
                IBackend.ComposeLinearFused(this, output, input, weight, bias, fusion);
                return;
            }
            LinearImpl(output, input, weight, bias, cacheWeightCast: true, fuseGelu: true);
            return;
        }
        GroupScaleSpec gate = fusion.Prologue;
        int headDim = gate.GroupWidth, heads = (int)gate.GateStride;
        bool fused = int8 && fusion.Epilogue == LinearActivation.None && FuseHeadGateIntoQuant
            && gate.Function == GateFunction.TwoSigmoid && !gate.Accumulate && gate.GateOffset == 0 && headDim > 0 && (long)heads * headDim == input.Shape[input.Shape.Rank - 1]
            && input.DType == DType.F16 && fusion.PrologueGates.DType == DType.F16
            && weight.QuantInfo is QuantWeightInfo { ConvRotGroupSize: > 0 } gq && _kernels is not null
            && _kernels.HasFusedGatedConvRotQuant((int)weight.Shape[1], gq.ConvRotGroupSize, srcF16: true, heads, headDim);
        if (!fused)
        {
            IBackend.ComposeLinearFused(this, output, input, weight, bias, fusion);
            return;
        }
        LinearImpl(output, input, weight, bias, cacheWeightCast: true, preGate: fusion.PrologueGates, preGateHeads: heads, preGateHeadDim: headDim);
    }
}
