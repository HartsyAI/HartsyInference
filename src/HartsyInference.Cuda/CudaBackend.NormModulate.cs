using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Backends.Norm;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda.Profiling;

namespace HartsyInference.Cuda;

// The generic norm-and-modulate and chunk-activate ops on the device (Kernels/norm/norm_modulate.cu).
public sealed partial class CudaBackend
{
    /// <inheritdoc />
    public void NormModulate(in NormModulateCall call)
    {
        using NvtxRange _nvtx = NvtxRange.Push("NormModulate");
        NormModulateGeometry g = NormModulateContract.Validate(call);
        if (g.Rows == 0) return;
        using OpScope _op = EnterOp();
        EnsureKernels();
        if (!_kernels!.HasNormModulate(call.Input.DType, call.Output.DType))
            throw new NotSupportedException($"CUDA NormModulate serves F32, F16 and BF16 activations, F32 and F16 mixed; got input {call.Input.DType}, output {call.Output.DType}.");
        ModulationSpec mod = call.Modulation;
        ulong pIn = 0, pW = 0, pB = 0, pScale = 0, pShift = 0, pIndex = 0, pOut = 0;
        bool cached = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(call.Input);
            if (call.Weight is not null) pW = GpuTransferHelper.CopyToDevice(call.Weight);
            if (call.Bias is not null) pB = GpuTransferHelper.CopyToDevice(call.Bias);
            if (mod.Scale is not null) pScale = GpuTransferHelper.CopyToDevice(mod.Scale);
            if (mod.Shift is not null) pShift = ReferenceEquals(mod.Shift, mod.Scale) ? pScale : GpuTransferHelper.CopyToDevice(mod.Shift);
            if (mod.RowIndex is not null) pIndex = GpuTransferHelper.CopyToDevice(mod.RowIndex);
            nuint outBytes = GpuTransferHelper.ByteSize(call.Output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels.LaunchNormModulate(pOut, pIn, pW, pB, pScale, pShift, pIndex, call.Kind, call.Axis, call.Input.DType, call.Output.DType, mod.Mode == ScaleMode.OnePlus, call.Eps, g, _stream.Handle);
            GpuTransferHelper.CacheActivation(call.Output, pOut, outBytes);
            cached = true;
        }
        finally
        {
            if (!cached) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
            GpuTransferHelper.FreeDevice(pW);
            GpuTransferHelper.FreeDevice(pB);
            GpuTransferHelper.FreeDevice(pScale);
            if (pShift != pScale) GpuTransferHelper.FreeDevice(pShift);
            GpuTransferHelper.FreeDevice(pIndex);
        }
    }

    /// <inheritdoc />
    public void ChunkActivate(ReadOnlySpan<Tensor> outputs, Tensor input, ReadOnlySpan<ChunkActivation> activations)
    {
        using NvtxRange _nvtx = NvtxRange.Push("ChunkActivate");
        (long rows, int width) = ChunkActivateReference.Validate(outputs, input, activations);
        using OpScope _op = EnterOp();
        EnsureKernels();
        Span<ulong> pOuts = stackalloc ulong[outputs.Length];
        ulong pIn = 0;
        int cached = 0;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint bytes = (nuint)(rows * width * sizeof(float));
            for (int c = 0; c < outputs.Length; c++) pOuts[c] = GpuTransferHelper.AllocateDevice(bytes);
            _kernels!.LaunchChunkActivate(pOuts, pIn, activations, rows, width, _stream.Handle);
            for (; cached < outputs.Length; cached++) GpuTransferHelper.CacheActivation(outputs[cached], pOuts[cached], bytes);
        }
        finally
        {
            for (int c = cached; c < outputs.Length; c++) GpuTransferHelper.FreeDevice(pOuts[c]);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }
}
