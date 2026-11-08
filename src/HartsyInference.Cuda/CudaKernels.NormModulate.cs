using System.Runtime.InteropServices;
using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Backends.Norm;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Cuda;

// The generic norm-and-modulate and chunk-activate kernels (Kernels/norm/norm_modulate.cu): one by-value geometry
// parameter, so a launch captures into a graph without constant memory.
public sealed partial class CudaKernels
{
    /// <summary>Most blocks one norm launch spans; the kernels grid-stride past it.</summary>
    private const uint NormMaxBlocks = 1u << 20;

    private CudaModule? _normModule;
    private readonly Dictionary<(DType In, DType Out), (nint Rows, nint Flat, nint Channel)> _normModulate = new();
    private nint _chunkActivateF32;

    /// <summary>Kernel parameter of the norm kernels; mirrors <c>NormModulateGeom</c> in <c>norm_modulate.cu</c> field for field.</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct NormModulateGeomArgs
    {
        public ulong Rows;
        public long Spatial;
        public long RowsPerMod;
        public long RowStride;
        public long ScaleOffset;
        public long ShiftOffset;
        public float Eps;
        public int Width;
        public int Kind;
        public int OnePlus;
        public int Flags;
    }

    /// <summary>Kernel parameter of <c>chunk_activate_f32</c>; mirrors <c>ChunkActivateArgs</c> field for field.</summary>
    [StructLayout(LayoutKind.Sequential)]
    private unsafe struct ChunkActivateArgs
    {
        public fixed ulong Out[ChunkActivateReference.MaxChunks];
        public fixed int Act[ChunkActivateReference.MaxChunks];
        public ulong Rows;
        public int Width;
        public int Count;
    }

    private void LoadNormModulateKernels()
    {
        _normModule = LoadOwnedModule(Ptx("norm_modulate"));
        foreach ((DType inDtype, DType outDtype, string suffix) in new[]
        {
            (DType.F32, DType.F32, "f32_f32"), (DType.F32, DType.F16, "f32_f16"), (DType.F16, DType.F16, "f16_f16"),
            (DType.F16, DType.F32, "f16_f32"), (DType.BF16, DType.BF16, "bf16_bf16"),
        })
        {
            _normModulate[(inDtype, outDtype)] = (_normModule.GetFunction("norm_modulate_rows_" + suffix),
                _normModule.GetFunction("norm_modulate_flat_" + suffix), _normModule.GetFunction("norm_modulate_channel_" + suffix));
        }
        _chunkActivateF32 = _normModule.GetFunction("chunk_activate_f32");
    }

    /// <summary>True when the kernels serve this activation dtype pair.</summary>
    public bool HasNormModulate(DType inDtype, DType outDtype) => _normModulate.ContainsKey((inDtype, outDtype));

    /// <summary>Launches a norm-and-modulate with geometry <paramref name="g"/>: the row kernel for a norm along the last axis, the
    /// channel kernel along the channel axis, the flat kernel for no norm. Absent vectors are 0; <paramref name="onePlus"/> is the scale mode.</summary>
    public unsafe void LaunchNormModulate(ulong output, ulong input, ulong weight, ulong bias, ulong scale, ulong shift, ulong rowIndex,
        NormKind kind, NormAxis axis, DType inDtype, DType outDtype, bool onePlus, float eps, in NormModulateGeometry g, nint stream)
    {
        (nint rows, nint flat, nint channel) = _normModulate[(inDtype, outDtype)];
        NormModulateGeomArgs args = new()
        {
            Rows = (ulong)g.Rows, Spatial = g.Spatial, RowsPerMod = g.RowsPerMod, RowStride = g.RowStride,
            ScaleOffset = g.ScaleOffset, ShiftOffset = g.ShiftOffset, Eps = eps, Width = g.Width, Kind = (int)kind,
            OnePlus = onePlus ? 1 : 0,
            Flags = (weight != 0 ? 1 : 0) | (bias != 0 ? 2 : 0) | (scale != 0 ? 4 : 0) | (shift != 0 ? 8 : 0) | (rowIndex != 0 ? 16 : 0),
        };
        ulong outArg = output, inArg = input, wArg = weight, bArg = bias, scArg = scale, shArg = shift, ixArg = rowIndex;
        if (kind == NormKind.None)
        {
            void** a = stackalloc void*[8];
            a[0] = &outArg; a[1] = &inArg; a[2] = &wArg; a[3] = &bArg; a[4] = &scArg; a[5] = &shArg; a[6] = &ixArg; a[7] = &args;
            long total = g.Rows * g.Width;
            uint grid = (uint)Math.Min((total + BlockSize - 1) / BlockSize, NormMaxBlocks);
            CudaDriverApi.cuLaunchKernel(flat, grid, 1, 1, BlockSize, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
        }
        else if (axis == NormAxis.LastDim)
        {
            void** a = stackalloc void*[8];
            a[0] = &outArg; a[1] = &inArg; a[2] = &wArg; a[3] = &bArg; a[4] = &scArg; a[5] = &shArg; a[6] = &ixArg; a[7] = &args;
            uint grid = (uint)Math.Min(g.Rows, NormMaxBlocks);
            CudaDriverApi.cuLaunchKernel(rows, grid, 1, 1, BlockSize, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
        }
        else
        {
            void** a = stackalloc void*[5];
            a[0] = &outArg; a[1] = &inArg; a[2] = &wArg; a[3] = &bArg; a[4] = &args;
            uint grid = (uint)Math.Min((g.Rows + BlockSize - 1) / BlockSize, NormMaxBlocks);
            CudaDriverApi.cuLaunchKernel(channel, grid, 1, 1, BlockSize, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
        }
    }

    /// <summary>Launches the chunk split of an F32 <c>[rows, count · width]</c> input into <paramref name="outputs"/> through <paramref name="activations"/>.</summary>
    public unsafe void LaunchChunkActivate(ReadOnlySpan<ulong> outputs, ulong input, ReadOnlySpan<ChunkActivation> activations, long rows, int width, nint stream)
    {
        ChunkActivateArgs args = new() { Rows = (ulong)rows, Width = width, Count = outputs.Length };
        for (int c = 0; c < outputs.Length; c++)
        {
            args.Out[c] = outputs[c];
            args.Act[c] = (int)activations[c];
        }
        ulong inArg = input;
        void** a = stackalloc void*[2];
        a[0] = &inArg; a[1] = &args;
        long total = rows * width * outputs.Length;
        uint grid = (uint)Math.Min((total + BlockSize - 1) / BlockSize, NormMaxBlocks);
        CudaDriverApi.cuLaunchKernel(_chunkActivateF32, grid, 1, 1, BlockSize, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }
}
