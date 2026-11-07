using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda.Profiling;

namespace HartsyInference.Cuda;

// The row-index ops and the dense-grid scatter and gather on the device.
public sealed partial class CudaBackend
{
    /// <inheritdoc />
    public void GatherRows(Tensor output, Tensor input, Tensor indices)
    {
        using NvtxRange _nvtx = NvtxRange.Push("GatherRows");
        RowIndexGeometry g = RowIndexContract.ValidateGather(output, input, indices);
        if (g.Rows == 0) return;
        using OpScope _op = EnterOp();
        EnsureKernels();
        ulong pIn = 0, pIdx = 0, pOut = 0;
        bool cached = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            pIdx = GpuTransferHelper.CopyToDevice(indices);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchGatherRows(pOut, pIn, pIdx, g, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cached = true;
        }
        finally
        {
            if (!cached) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIdx);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    /// <inheritdoc />
    public bool TryGatherRowsResident(Tensor output, Tensor input, ReadOnlySpan<int> rowIndices)
    {
        using OpScope _op = EnterOp();
        if (!GpuTransferHelper.IsWeightCached(input) && !GpuTransferHelper.IsActivationCached(input))
            return false;
        using Tensor indices = IndexTensor.Rows(rowIndices);
        GatherRows(output, input, indices);
        return true;
    }

    /// <inheritdoc />
    /// <remarks>The output is a resident accumulator: it stays cached after the add.</remarks>
    public void ScatterAddRows(Tensor output, Tensor input, Tensor indices, Tensor? weights = null)
    {
        using NvtxRange _nvtx = NvtxRange.Push("ScatterAddRows");
        RowIndexGeometry g = RowIndexContract.ValidateScatterAdd(output, input, indices, weights);
        if (g.Rows == 0) return;
        using OpScope _op = EnterOp();
        EnsureKernels();
        ulong pOut = 0, pIn = 0, pIdx = 0, pW = 0;
        try
        {
            pOut = GpuTransferHelper.CopyToDevice(output);
            pIn = GpuTransferHelper.CopyToDevice(input);
            pIdx = GpuTransferHelper.CopyToDevice(indices);
            pW = weights is null ? 0 : GpuTransferHelper.CopyToDevice(weights);
            _kernels!.LaunchScatterAddRows(pOut, pIn, pIdx, pW, g, _stream.Handle);
            output._gpuSyncCallback = null;
            output._gpuDisposeCallback = null;
            GpuTransferHelper.CacheActivation(output, pOut, GpuTransferHelper.ByteSize(output));
        }
        finally
        {
            if (pW != 0) GpuTransferHelper.FreeDevice(pW);
            GpuTransferHelper.FreeDevice(pIdx);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    /// <inheritdoc />
    public void GatherAddRows(Tensor h, Tensor table, Tensor indices)
    {
        using NvtxRange _nvtx = NvtxRange.Push("GatherAddRows");
        RowIndexGeometry g = RowIndexContract.ValidateGatherAdd(h, table, indices);
        if (g.Rows == 0) return;
        using OpScope _op = EnterOp();
        EnsureKernels();
        ulong pH = 0, pTable = 0, pIdx = 0;
        try
        {
            pH = GpuTransferHelper.CopyToDevice(h);
            pTable = GpuTransferHelper.CopyToDevice(table);
            pIdx = GpuTransferHelper.CopyToDevice(indices);
            _kernels!.LaunchGatherAddRows(pH, pTable, pIdx, g, _stream.Handle);
            h._gpuSyncCallback = null;
            h._gpuDisposeCallback = null;
            GpuTransferHelper.CacheActivation(h, pH, GpuTransferHelper.ByteSize(h));
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pIdx);
            GpuTransferHelper.FreeDevice(pTable);
        }
    }

    /// <inheritdoc />
    /// <remarks>The grid is a resident accumulator built and consumed on the device: it stays cached after the scatter.</remarks>
    public void GridScatter(Tensor grid, Tensor features, Tensor coords, in GridSpec spec)
    {
        using NvtxRange _nvtx = NvtxRange.Push("GridScatter");
        GridGeometry g = GridContract.Validate(grid, features, coords, spec, scatter: true);
        if (g.Voxels == 0) return;
        using OpScope _op = EnterOp();
        EnsureKernels();
        ulong pGrid = 0, pFeats = 0, pCoords = 0;
        try
        {
            pGrid = GpuTransferHelper.CopyToDevice(grid);
            pFeats = GpuTransferHelper.CopyToDevice(features);
            pCoords = GpuTransferHelper.CopyToDevice(coords);
            _kernels!.LaunchGrid(true, pGrid, pFeats, pCoords, spec, g, _stream.Handle);
            grid._gpuSyncCallback = null;
            grid._gpuDisposeCallback = null;
            GpuTransferHelper.CacheActivation(grid, pGrid, GpuTransferHelper.ByteSize(grid));
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pCoords);
            GpuTransferHelper.FreeDevice(pFeats);
        }
    }

    /// <inheritdoc />
    public void GridGather(Tensor features, Tensor grid, Tensor coords, in GridSpec spec)
    {
        using NvtxRange _nvtx = NvtxRange.Push("GridGather");
        GridGeometry g = GridContract.Validate(grid, features, coords, spec, scatter: false);
        if (g.Voxels == 0) return;
        using OpScope _op = EnterOp();
        EnsureKernels();
        ulong pGrid = 0, pCoords = 0, pFeats = 0;
        bool cached = false;
        try
        {
            pGrid = GpuTransferHelper.CopyToDevice(grid);
            pCoords = GpuTransferHelper.CopyToDevice(coords);
            nuint bytes = GpuTransferHelper.ByteSize(features);
            pFeats = GpuTransferHelper.AllocateDevice(bytes);
            _kernels!.LaunchGrid(false, pGrid, pFeats, pCoords, spec, g, _stream.Handle);
            GpuTransferHelper.CacheActivation(features, pFeats, bytes);
            cached = true;
        }
        finally
        {
            if (!cached) GpuTransferHelper.FreeDevice(pFeats);
            GpuTransferHelper.FreeDevice(pCoords);
            GpuTransferHelper.FreeDevice(pGrid);
        }
    }
}
