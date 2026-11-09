using System.Runtime.InteropServices;
using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Cuda;

// The generic rotary position embedding (Kernels/rotary/rotary.cu): one by-value geometry parameter, so a launch
// captures into a decode graph and reads its table row from the device position buffer.
public sealed partial class CudaKernels
{
    private CudaModule? _rotaryModule;
    private nint _rotaryF32, _rotaryF16, _rotaryBf16;

    /// <summary>Kernel parameter of the rotary kernel; mirrors <c>RotaryGeom</c> in <c>rotary.cu</c> field for field.</summary>
    [StructLayout(LayoutKind.Sequential)]
    private unsafe struct RotaryGeomArgs
    {
        public ulong Total;
        public fixed long Dims[RotaryGeometry.IndexAxes];
        public fixed long X[RotaryGeometry.IndexAxes];
        public fixed long Cos[RotaryGeometry.IndexAxes];
        public fixed long Sin[RotaryGeometry.IndexAxes];
        public long XSecond;
        public long CosSecond;
        public long SinSecond;
        public long XOff;
        public long CosOff;
        public long SinOff;
        public long PosRowStride;
        public int PosSlot;
    }

    private void LoadRotaryKernels()
    {
        _rotaryModule = LoadOwnedModule(Ptx("rotary"));
        _rotaryF32 = _rotaryModule.GetFunction("rotary_f32");
        _rotaryF16 = _rotaryModule.GetFunction("rotary_f16");
        _rotaryBf16 = _rotaryModule.GetFunction("rotary_bf16");
    }

    /// <summary>Launches the rotation over every pair of <paramref name="g"/>; <paramref name="positions"/> is the device position buffer, or any valid pointer when the geometry carries no row stride.</summary>
    public unsafe void LaunchRotary(ulong x, ulong cos, ulong sin, ulong positions, in RotaryGeometry g, DType dtype, nint stream)
    {
        nint func = dtype == DType.F32 ? _rotaryF32 : dtype == DType.F16 ? _rotaryF16 : _rotaryBf16;
        RotaryGeomArgs args = new()
        {
            Total = (ulong)g.PairCount,
            XSecond = g.XSecond,
            CosSecond = g.CosSecond,
            SinSecond = g.SinSecond,
            XOff = g.XOffset,
            CosOff = g.CosOffset,
            SinOff = g.SinOffset,
            PosRowStride = g.PositionRowStride,
            PosSlot = g.PositionSlot,
        };
        for (int axis = 0; axis < RotaryGeometry.IndexAxes; axis++)
        {
            args.Dims[axis] = g.Dim(axis);
            args.X[axis] = g.XStride(axis);
            args.Cos[axis] = g.CosStride(axis);
            args.Sin[axis] = g.SinStride(axis);
        }
        ulong xArg = x, cosArg = cos, sinArg = sin, posArg = positions;
        void** kernelArgs = stackalloc void*[5];
        kernelArgs[0] = &xArg;
        kernelArgs[1] = &cosArg;
        kernelArgs[2] = &sinArg;
        kernelArgs[3] = &posArg;
        kernelArgs[4] = &args;
        uint grid = (uint)Math.Min((g.PairCount + BlockSize - 1) / BlockSize, LayoutMaxBlocks);
        CudaDriverApi.cuLaunchKernel(func, grid, 1, 1, BlockSize, 1, 1, 0, stream, (nint)kernelArgs, 0).ThrowOnError();
    }
}
