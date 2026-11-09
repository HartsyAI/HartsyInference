using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.SanaWm;

/// <summary>Camera-branch RoPE tables sliced from the main 3D Wan table, whose complex columns are <c>t | h | w</c>.
/// The camera slice keeps <c>t[0:camT] | h[originT:originT+camH] | w[originT+originH:…]</c>, the column rule of upstream
/// <c>sana_gdn_blocks_triton._forward_cam_branch</c>. At head dim 112 that is columns 0–9, 20–28 and 38–46.</summary>
public static class SanaWmCamRope
{
    /// <summary>Returns cosine and sine tables <c>[tokens, headDim/2]</c> with <c>c[2i] = c[2i+1]</c>, the layout
    /// <c>RopeTableLayout.InterleavedPair</c> reads.</summary>
    public static (Tensor Cos, Tensor Sin) BuildCameraTables(IBackend backend, Tensor mainCos, Tensor mainSin, int headDim)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(mainCos);
        ArgumentNullException.ThrowIfNull(mainSin);
        if (headDim < 24 || headDim % 8 != 0)
            throw new ArgumentOutOfRangeException(nameof(headDim), headDim, "Head dim must be a multiple of 8 and at least 24.");

        int mainColumns = headDim / 2;
        int originT = mainColumns - 2 * (headDim / 6);
        int originH = headDim / 6;
        int camWidth = mainColumns;
        int camHalf = camWidth / 2;
        int camT = camHalf - 2 * (camWidth / 6);
        int camH = camWidth / 6;
        if (originT + 2 * originH != mainColumns || camT <= 0 || camH <= 0 || camT + 2 * camH != camHalf)
            throw new ArgumentException($"Head dim {headDim} does not split into camera t|h|w RoPE axes.", nameof(headDim));
        ValidateTable(mainCos, mainColumns, nameof(mainCos));
        ValidateTable(mainSin, mainColumns, nameof(mainSin));

        long rows = mainCos.Shape[0];
        Tensor cos = new(new TensorShape(rows, camWidth), DType.F32);
        Tensor sin = new(new TensorShape(rows, camWidth), DType.F32);
        bool built = false;
        try
        {
            CopyAxes(backend, cos, mainCos, rows, originT, originH, camT, camH, mainColumns);
            CopyAxes(backend, sin, mainSin, rows, originT, originH, camT, camH, mainColumns);
            built = true;
            return (cos, sin);
        }
        finally
        {
            if (!built)
            {
                cos.Dispose();
                sin.Dispose();
            }
        }
    }

    private static void ValidateTable(Tensor table, int columns, string name)
    {
        if (table.DType != DType.F32 || table.Shape.Rank != 2 || table.Shape[1] != columns)
            throw new ArgumentException($"{name} must be F32 [tokens, {columns}]; got {table.DType} {table.Shape}.", name);
    }

    private static void CopyAxes(IBackend backend, Tensor destination, Tensor source, long rows, int originT, int originH,
        int camT, int camH, int mainColumns)
    {
        CopyPairs(backend, destination, source, rows, 0, 0, camT, mainColumns);
        CopyPairs(backend, destination, source, rows, originT, 2 * camT, camH, mainColumns);
        CopyPairs(backend, destination, source, rows, originT + originH, 2 * (camT + camH), camH, mainColumns);
    }

    /// <summary>Copies <paramref name="pairs"/> complex columns from <paramref name="sourceColumn"/> into the real columns
    /// from <paramref name="destinationColumn"/>, writing each complex value to both elements of its pair.</summary>
    private static void CopyPairs(IBackend backend, Tensor destination, Tensor source, long rows, int sourceColumn,
        int destinationColumn, int pairs, int sourceWidth)
    {
        long destinationWidth = destination.Shape[1];
        ReadOnlySpan<long> dims = [rows, pairs, 2];
        ReadOnlySpan<long> sourceStrides = [sourceWidth, 1, 0];
        ReadOnlySpan<long> destinationStrides = [destinationWidth, 2, 1];
        StridedView from = new(dims, sourceStrides, sourceColumn);
        StridedView to = new(dims, destinationStrides, destinationColumn);
        backend.CopyStrided(destination, source, new StridedCopySpec(to, from));
    }
}
