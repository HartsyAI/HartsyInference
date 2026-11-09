using System.Runtime.InteropServices;
using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Backends.Gating;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Cuda;

// The generic gated linear unit and group gate (Kernels/gating/glu_gate.cu): one by-value geometry parameter each, so a
// launch captures into a graph.
public sealed partial class CudaKernels
{
    /// <summary>Most blocks one gating launch spans; the kernels grid-stride past it.</summary>
    private const uint GatingMaxBlocks = 1u << 20;

    private CudaModule? _gatingModule;
    private nint _gluF32, _gluF16, _gluBf16;
    private readonly Dictionary<(DType Activation, DType Gates), nint> _groupScale = new();

    /// <summary>Kernel parameter of the GLU kernels; mirrors <c>GluGeom</c> in <c>glu_gate.cu</c> field for field.</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct GluGeomArgs
    {
        public ulong Rows;
        public int Inner;
        public int Act;
        public int GateFirst;
    }

    /// <summary>Kernel parameter of the group-scale kernels; mirrors <c>GroupScaleGeom</c> field for field.</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct GroupScaleGeomArgs
    {
        public ulong Total;
        public long GateStride;
        public long GateOffset;
        public int Width;
        public int GroupWidth;
        public int Function;
        public int Accumulate;
    }

    private void LoadGatingKernels()
    {
        _gatingModule = LoadOwnedModule(Ptx("glu_gate"));
        _gluF32 = _gatingModule.GetFunction("glu_f32");
        _gluF16 = _gatingModule.GetFunction("glu_f16");
        _gluBf16 = _gatingModule.GetFunction("glu_bf16");
        _groupScale[(DType.F32, DType.F32)] = _gatingModule.GetFunction("group_scale_f32");
        _groupScale[(DType.F16, DType.F32)] = _gatingModule.GetFunction("group_scale_f16_f32");
        _groupScale[(DType.F16, DType.F16)] = _gatingModule.GetFunction("group_scale_f16_f16");
        _groupScale[(DType.BF16, DType.F32)] = _gatingModule.GetFunction("group_scale_bf16_f32");
        _groupScale[(DType.BF16, DType.BF16)] = _gatingModule.GetFunction("group_scale_bf16_bf16");
    }

    /// <summary>Launches the gated linear unit of <paramref name="spec"/> over <paramref name="rows"/> rows of <c>2 · inner</c> elements.</summary>
    public unsafe void LaunchGlu(ulong output, ulong input, DType dtype, in GluSpec spec, long rows, int inner, nint stream)
    {
        nint func = dtype == DType.F32 ? _gluF32 : dtype == DType.F16 ? _gluF16 : _gluBf16;
        GluGeomArgs args = new() { Rows = (ulong)rows, Inner = inner, Act = (int)spec.Activation, GateFirst = spec.Gate == GluGateHalf.First ? 1 : 0 };
        ulong outArg = output, inArg = input;
        void** a = stackalloc void*[3];
        a[0] = &outArg; a[1] = &inArg; a[2] = &args;
        long total = rows * inner;
        uint grid = (uint)Math.Min((total + BlockSize - 1) / BlockSize, GatingMaxBlocks);
        CudaDriverApi.cuLaunchKernel(func, grid, 1, 1, BlockSize, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }

    /// <summary>Launches the group scale of <paramref name="spec"/> with geometry <paramref name="g"/>; <paramref name="output"/> may be <paramref name="input"/>.</summary>
    public unsafe void LaunchGroupScale(ulong output, ulong input, ulong gates, DType dtype, DType gateDtype, in GroupScaleSpec spec, in GroupScaleGeometry g, nint stream)
    {
        nint func = _groupScale[(dtype, gateDtype)];
        GroupScaleGeomArgs args = new()
        {
            Total = (ulong)(g.Rows * g.Width), GateStride = spec.GateStride, GateOffset = spec.GateOffset, Width = g.Width, GroupWidth = g.GroupWidth,
            Function = (int)spec.Function, Accumulate = spec.Accumulate ? 1 : 0,
        };
        ulong outArg = output, inArg = input, gateArg = gates;
        void** a = stackalloc void*[4];
        a[0] = &outArg; a[1] = &inArg; a[2] = &gateArg; a[3] = &args;
        uint grid = (uint)Math.Min(((long)args.Total + BlockSize - 1) / BlockSize, GatingMaxBlocks);
        CudaDriverApi.cuLaunchKernel(func, grid, 1, 1, BlockSize, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }
}
