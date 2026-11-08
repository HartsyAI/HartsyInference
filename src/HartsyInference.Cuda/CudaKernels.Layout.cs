using System.Runtime.InteropServices;
using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Cuda;

// Generic strided copy, strided accumulate and pad (Kernels/layout/layout_copy.cu). The geometry is one by-value
// kernel parameter, so a launch captures into a graph without constant memory or a device-side spec buffer.
public sealed partial class CudaKernels
{
    /// <summary>Most blocks one layout launch spans; the kernels grid-stride past it.</summary>
    private const uint LayoutMaxBlocks = 1u << 20;

    private CudaModule? _layoutModule;
    private nint _copyStridedU8, _copyStridedU16, _copyStridedU32, _copyStridedU64;
    private nint _accumulateStridedF32, _accumulateStridedF16, _accumulateStridedBf16, _accumulateStridedBf16F32In;
    private nint _padU8, _padU16, _padU32, _padU64;

    /// <summary>Kernel parameter of the strided kernels; mirrors <c>StridedGeom</c> in <c>layout_copy.cu</c> field for field.</summary>
    [StructLayout(LayoutKind.Sequential)]
    private unsafe struct StridedGeomArgs
    {
        public ulong Total;
        public fixed long Dims[StridedView.MaxRank];
        public fixed long Src[StridedView.MaxRank];
        public fixed long Dst[StridedView.MaxRank];
        public long SrcOff;
        public long DstOff;
        public int Rank;
    }

    /// <summary>Kernel parameter of the pad kernels; mirrors <c>PadGeom</c> in <c>layout_copy.cu</c> field for field.</summary>
    [StructLayout(LayoutKind.Sequential)]
    private unsafe struct PadGeomArgs
    {
        public ulong Total;
        public fixed long OutDims[StridedView.MaxRank];
        public fixed long InDims[StridedView.MaxRank];
        public fixed long Before[StridedView.MaxRank];
        public int Rank;
        public int Mode;
    }

    private void LoadLayoutKernels()
    {
        _layoutModule = LoadOwnedModule(Ptx("layout_copy"));
        _copyStridedU8 = _layoutModule.GetFunction("copy_strided_u8");
        _copyStridedU16 = _layoutModule.GetFunction("copy_strided_u16");
        _copyStridedU32 = _layoutModule.GetFunction("copy_strided_u32");
        _copyStridedU64 = _layoutModule.GetFunction("copy_strided_u64");
        _accumulateStridedF32 = _layoutModule.GetFunction("accumulate_strided_f32");
        _accumulateStridedF16 = _layoutModule.GetFunction("accumulate_strided_f16");
        _accumulateStridedBf16 = _layoutModule.GetFunction("accumulate_strided_bf16");
        _accumulateStridedBf16F32In = _layoutModule.GetFunction("accumulate_strided_bf16_f32in");
        _padU8 = _layoutModule.GetFunction("pad_u8");
        _padU16 = _layoutModule.GetFunction("pad_u16");
        _padU32 = _layoutModule.GetFunction("pad_u32");
        _padU64 = _layoutModule.GetFunction("pad_u64");
    }

    /// <summary>Launches the strided copy of <paramref name="g"/>'s element width: <c>output[dst(i)] = input[src(i)]</c>.</summary>
    public unsafe void LaunchCopyStrided(ulong output, ulong input, in StridedCopyGeometry g, nint stream)
    {
        nint func = g.ElementSize switch
        {
            1 => _copyStridedU8,
            2 => _copyStridedU16,
            4 => _copyStridedU32,
            _ => _copyStridedU64,
        };
        StridedGeomArgs args = StridedArgs(g);
        LaunchLayout(func, output, input, &args, g.ElementCount, stream);
    }

    /// <summary>Launches the strided accumulate <c>output[dst(i)] += input[src(i)]</c> in F32; an F32 input may add into BF16 output.</summary>
    public unsafe void LaunchAccumulateStrided(ulong output, ulong input, in StridedCopyGeometry g, DType outputDtype, DType inputDtype, nint stream)
    {
        nint func;
        if (outputDtype == DType.F32) func = _accumulateStridedF32;
        else if (outputDtype == DType.F16) func = _accumulateStridedF16;
        else func = inputDtype == DType.F32 ? _accumulateStridedBf16F32In : _accumulateStridedBf16;
        StridedGeomArgs args = StridedArgs(g);
        LaunchLayout(func, output, input, &args, g.ElementCount, stream);
    }

    /// <summary>Launches the pad of <paramref name="g"/>'s element width over every output element.</summary>
    public unsafe void LaunchPad(ulong output, ulong input, in PadGeometry g, nint stream)
    {
        nint func = g.ElementSize switch
        {
            1 => _padU8,
            2 => _padU16,
            4 => _padU32,
            _ => _padU64,
        };
        PadGeomArgs args = new() { Total = (ulong)g.ElementCount, Rank = g.Rank, Mode = (int)g.Mode };
        for (int axis = 0; axis < g.Rank; axis++)
        {
            args.OutDims[axis] = g.OutputDim(axis);
            args.InDims[axis] = g.InputDim(axis);
            args.Before[axis] = g.Before(axis);
        }
        LaunchLayout(func, output, input, &args, g.ElementCount, stream);
    }

    private static unsafe StridedGeomArgs StridedArgs(in StridedCopyGeometry g)
    {
        StridedGeomArgs args = new() { Total = (ulong)g.ElementCount, SrcOff = g.SourceOffset, DstOff = g.DestinationOffset, Rank = g.Rank };
        for (int axis = 0; axis < g.Rank; axis++)
        {
            args.Dims[axis] = g.Dim(axis);
            args.Src[axis] = g.SourceStride(axis);
            args.Dst[axis] = g.DestinationStride(axis);
        }
        return args;
    }

    private static unsafe void LaunchLayout(nint func, ulong output, ulong input, void* geometry, long total, nint stream)
    {
        ulong outArg = output, inArg = input;
        void** args = stackalloc void*[3];
        args[0] = &outArg;
        args[1] = &inArg;
        args[2] = geometry;
        uint grid = (uint)Math.Min((total + BlockSize - 1) / BlockSize, LayoutMaxBlocks);
        CudaDriverApi.cuLaunchKernel(func, grid, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }
}
