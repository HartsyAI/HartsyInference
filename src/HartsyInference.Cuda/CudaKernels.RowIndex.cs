using System.Runtime.InteropServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Backends.Layout;

namespace HartsyInference.Cuda;

// The row-index and dense-grid ops (Kernels/rows/row_index.cu) and the per-point embeddings (Kernels/point/point_ops.cu): one
// by-value geometry parameter each, so a launch captures into a graph.
public sealed partial class CudaKernels
{
    private CudaModule? _rowIndexModule;
    private CudaModule? _pointOpsModule;
    private nint _gatherRowsF32, _scatterAddRowsF32, _gatherAddRowsF32, _gridScatterF32, _gridGatherF32;
    private nint _fakeQuantizeUniformF32, _fourierEmbedF32;

    /// <summary>Kernel parameter of the row-index kernels; mirrors <c>RowIndexGeom</c> in <c>row_index.cu</c> field for field.</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct RowIndexGeomArgs
    {
        public ulong Total;
        public uint Width;
        public uint Reserved;
    }

    /// <summary>Kernel parameter of the grid kernels; mirrors <c>GridGeom</c> field for field.</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct GridGeomArgs
    {
        public ulong Total;
        public ulong Cells;
        public int Channels;
        public int Height;
        public int Width;
        public int CoordStride;
        public int FirstCoordColumn;
        public int Reserved;
    }

    /// <summary>Kernel parameter of the fake quantize; mirrors <c>FakeQuantizeArgs</c> in <c>point_ops.cu</c> field for field.</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct FakeQuantizeArgs
    {
        public float Low;
        public float High;
        public float Range;
        public float Steps;
        public ulong Count;
    }

    /// <summary>Kernel parameter of the Fourier embedding; mirrors <c>FourierEmbedArgs</c> field for field.</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct FourierEmbedArgs
    {
        public ulong Count;
        public uint Dims;
        public uint Bands;
        public uint Width;
        public uint IncludeInput;
    }

    private void LoadRowIndexKernels()
    {
        _rowIndexModule = LoadOwnedModule(Ptx("row_index"));
        _gatherRowsF32 = _rowIndexModule.GetFunction("gather_rows_f32");
        _scatterAddRowsF32 = _rowIndexModule.GetFunction("scatter_add_rows_f32");
        _gatherAddRowsF32 = _rowIndexModule.GetFunction("gather_add_rows_f32");
        _gridScatterF32 = _rowIndexModule.GetFunction("grid_scatter_f32");
        _gridGatherF32 = _rowIndexModule.GetFunction("grid_gather_f32");
        _pointOpsModule = LoadOwnedModule(Ptx("point_ops"));
        _fakeQuantizeUniformF32 = _pointOpsModule.GetFunction("fake_quantize_uniform_f32");
        _fourierEmbedF32 = _pointOpsModule.GetFunction("fourier_embed_f32");
    }

    /// <summary>Row gather: <c>output[m] = input[indices[m]]</c> over <paramref name="g"/>'s rows and width.</summary>
    public unsafe void LaunchGatherRows(ulong output, ulong input, ulong indices, in RowIndexGeometry g, nint stream)
    {
        RowIndexGeomArgs args = new() { Total = (ulong)(g.Rows * g.Width), Width = (uint)g.Width };
        ulong outArg = output, inArg = input, idxArg = indices;
        void** a = stackalloc void*[4];
        a[0] = &outArg; a[1] = &inArg; a[2] = &idxArg; a[3] = &args;
        CudaDriverApi.cuLaunchKernel(_gatherRowsF32, Blocks((long)args.Total), 1, 1, BlockSize, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }

    /// <summary>Row scatter-add in place: <c>output[indices[m]] += weights[m] · input[m]</c>; <paramref name="weights"/> 0 adds the rows unweighted.</summary>
    public unsafe void LaunchScatterAddRows(ulong output, ulong input, ulong indices, ulong weights, in RowIndexGeometry g, nint stream)
    {
        RowIndexGeomArgs args = new() { Total = (ulong)(g.Rows * g.Width), Width = (uint)g.Width };
        ulong outArg = output, inArg = input, idxArg = indices, wArg = weights;
        void** a = stackalloc void*[5];
        a[0] = &outArg; a[1] = &inArg; a[2] = &idxArg; a[3] = &wArg; a[4] = &args;
        CudaDriverApi.cuLaunchKernel(_scatterAddRowsF32, Blocks((long)args.Total), 1, 1, BlockSize, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }

    /// <summary>Gather-add in place: <c>h[r] += table[indices[r]]</c>.</summary>
    public unsafe void LaunchGatherAddRows(ulong h, ulong table, ulong indices, in RowIndexGeometry g, nint stream)
    {
        RowIndexGeomArgs args = new() { Total = (ulong)(g.Rows * g.Width), Width = (uint)g.Width };
        ulong hArg = h, tableArg = table, idxArg = indices;
        void** a = stackalloc void*[4];
        a[0] = &hArg; a[1] = &tableArg; a[2] = &idxArg; a[3] = &args;
        CudaDriverApi.cuLaunchKernel(_gatherAddRowsF32, Blocks((long)args.Total), 1, 1, BlockSize, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }

    /// <summary>Dense-grid scatter (<paramref name="scatter"/> true: <c>grid[c, cell(v)] = feats[v, c]</c>) or gather (the inverse), one thread per voxel · channel.</summary>
    public unsafe void LaunchGrid(bool scatter, ulong grid, ulong feats, ulong coords, in GridSpec spec, in GridGeometry g, nint stream)
    {
        GridGeomArgs args = new()
        {
            Total = (ulong)g.Voxels * (ulong)g.Channels, Cells = (ulong)g.Cells, Channels = g.Channels, Height = spec.Height, Width = spec.Width,
            CoordStride = spec.CoordStride, FirstCoordColumn = spec.FirstCoordColumn,
        };
        ulong gridArg = grid, featsArg = feats, coordsArg = coords;
        void** a = stackalloc void*[4];
        a[0] = scatter ? &gridArg : &featsArg; a[1] = scatter ? &featsArg : &gridArg; a[2] = &coordsArg; a[3] = &args;
        CudaDriverApi.cuLaunchKernel(scatter ? _gridScatterF32 : _gridGatherF32, Blocks((long)args.Total), 1, 1, BlockSize, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }

    /// <summary>Uniform fake quantization of <paramref name="count"/> elements into <paramref name="levels"/> steps over <c>[low, high]</c>.</summary>
    public unsafe void LaunchFakeQuantizeUniform(ulong output, ulong input, float low, float high, int levels, long count, nint stream)
    {
        FakeQuantizeArgs args = new() { Low = low, High = high, Range = high - low, Steps = levels - 1, Count = (ulong)count };
        ulong outArg = output, inArg = input;
        void** a = stackalloc void*[3];
        a[0] = &outArg; a[1] = &inArg; a[2] = &args;
        CudaDriverApi.cuLaunchKernel(_fakeQuantizeUniformF32, Blocks(count), 1, 1, BlockSize, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }

    /// <summary>Fourier embedding of <paramref name="count"/> points, one thread per point · coordinate.</summary>
    public unsafe void LaunchFourierEmbed(ulong output, ulong coords, in FourierEmbedSpec spec, long count, nint stream)
    {
        FourierEmbedArgs args = new() { Count = (ulong)count, Dims = (uint)spec.InputDims, Bands = (uint)spec.Bands, Width = (uint)spec.Width, IncludeInput = spec.IncludeInput ? 1u : 0u };
        ulong outArg = output, coordsArg = coords;
        void** a = stackalloc void*[3];
        a[0] = &outArg; a[1] = &coordsArg; a[2] = &args;
        CudaDriverApi.cuLaunchKernel(_fourierEmbedF32, Blocks(count * spec.InputDims), 1, 1, BlockSize, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }
}
