namespace HartsyInference.Core.Tensors.Quant;

/// <summary>How one quantized weight is encoded and where its scales live, so a codec or backend can read it without the checkpoint's file conventions.</summary>
/// <remarks><para>The scale is a multiplier: <c>w = q * 2^(e - 127)</c> for E8M0 bytes. All tensors are <b>borrowed</b> from
/// the checkpoint mapping, as in <see cref="QuantWeightInfo"/>. Slices narrow the recipe to a window of the logical
/// <c>[LogicalRows, LogicalCols]</c> matrix without copying, and refuse a window that would pair elements with another
/// block's scale rather than decode plausible garbage.</para></remarks>
public sealed record QuantRecipe
{
    /// <summary>The numerical encoding.</summary>
    public required QuantEncoding Encoding { get; init; }

    /// <summary>The weight region one scale covers.</summary>
    public required BlockGeometry Geometry { get; init; }

    /// <summary>How <see cref="Scale"/> is arranged.</summary>
    public ScaleLayout ScaleLayout { get; init; } = ScaleLayout.RowMajorBlocks;

    /// <summary>The scale element type.</summary>
    public required DType ScaleDType { get; init; }

    /// <summary>Weight rows (output features) after dequantization.</summary>
    public required long LogicalRows { get; init; }

    /// <summary>Weight columns (input features) after dequantization; twice the packed byte width for 4-bit encodings.</summary>
    public required long LogicalCols { get; init; }

    /// <summary>Block scales, <c>[ceil(rows/blockRows), &gt;= ceil(cols/blockCols)]</c> in <see cref="Quant.ScaleLayout.RowMajorBlocks"/>.</summary>
    public Tensor? Scale { get; init; }

    /// <summary>MLX affine per-group bias, same shape as <see cref="Scale"/>.</summary>
    public Tensor? Bias { get; init; }

    /// <summary>NVFP4 per-tensor F32 scalar.</summary>
    public Tensor? GlobalScale { get; init; }

    /// <summary>NVFP4 activation scale.</summary>
    public Tensor? InputScale { get; init; }

    /// <summary>EXL3 decode companions.</summary>
    public Exl3Companions? Exl3 { get; init; }

    /// <summary>Block columns to skip in <see cref="Scale"/> and <see cref="Bias"/>, set by <see cref="SliceCols"/>.</summary>
    public long ScaleColOffset { get; init; }

    /// <summary>The <see cref="QuantWeightInfo.Format"/> string for this encoding, always distinct from the Comfy names (<c>nvfp4</c>, <c>mxfp4</c>, <c>mxfp8</c>) that select the Blackwell cuBLASLt path.</summary>
    public string FormatName => Encoding switch
    {
        QuantEncoding.Fp8E4M3BlockE8M0 => "recipe-fp8-block-e8m0",
        QuantEncoding.Fp8E4M3RowE8M0 => "recipe-fp8-row-e8m0",
        QuantEncoding.Mxfp4E8M0 => "recipe-mxfp4-e8m0",
        QuantEncoding.Nvfp4 => "recipe-nvfp4",
        QuantEncoding.AffineInt4 => "recipe-affine-int4",
        QuantEncoding.Exl3Trellis => "recipe-exl3",
        QuantEncoding.Gguf => "recipe-gguf",
        _ => throw new NotSupportedException($"No format name for {Encoding}."),
    };

    /// <summary>Elements per packed byte: 2 for the 4-bit encodings, 1 otherwise.</summary>
    public int ElementsPerByte => Encoding is QuantEncoding.Mxfp4E8M0 or QuantEncoding.Nvfp4 ? 2 : 1;

    /// <summary>Narrows the recipe to <paramref name="rowCount"/> rows from <paramref name="rowOffset"/>, viewing the matching scale rows.</summary>
    public QuantRecipe SliceRows(long rowOffset, long rowCount, string weightKey)
    {
        if (rowOffset < 0 || rowCount < 0 || rowOffset + rowCount > LogicalRows)
            throw new NotSupportedException($"'{weightKey}': rows [{rowOffset}..{rowOffset + rowCount}) are outside [0..{LogicalRows}).");
        if (Encoding == QuantEncoding.Exl3Trellis)
            throw new NotSupportedException($"'{weightKey}' is EXL3, whose Hadamard rotation spans every row; rows cannot be windowed.");

        int blockRows = ScaleLayout == ScaleLayout.Swizzled128 ? QuantWeightInfo.BlockScaleTileRows : Geometry.BlockRows;
        bool endsAtEdge = rowOffset + rowCount == LogicalRows;
        if (rowOffset % blockRows != 0 || (rowCount % blockRows != 0 && !endsAtEdge))
            throw new NotSupportedException(
                $"'{weightKey}' ({Encoding}, {Geometry} blocks, {ScaleLayout}) can only be sliced on {blockRows}-row boundaries; "
                + $"rows [{rowOffset}..{rowOffset + rowCount}) split a scale block.");

        // Swizzled scales are indexed by weight row (one block row per row), row-major ones by block row.
        long scaleOffset = ScaleLayout == ScaleLayout.Swizzled128 ? rowOffset : rowOffset / Geometry.BlockRows;
        long scaleCount = ScaleLayout == ScaleLayout.Swizzled128 ? rowCount : (rowCount + Geometry.BlockRows - 1) / Geometry.BlockRows;
        return this with
        {
            LogicalRows = rowCount,
            Scale = Scale?.SliceRows(scaleOffset, scaleCount),
            Bias = Bias?.SliceRows(scaleOffset, scaleCount),
        };
    }

    /// <summary>Narrows the recipe to <paramref name="colCount"/> columns from <paramref name="colOffset"/>; the caller supplies a matching column-contiguous packed weight.</summary>
    public QuantRecipe SliceCols(long colOffset, long colCount, string weightKey)
    {
        if (colOffset < 0 || colCount < 0 || colOffset + colCount > LogicalCols)
            throw new NotSupportedException($"'{weightKey}': columns [{colOffset}..{colOffset + colCount}) are outside [0..{LogicalCols}).");
        if (Encoding == QuantEncoding.Exl3Trellis || ScaleLayout == ScaleLayout.Swizzled128)
            throw new NotSupportedException($"'{weightKey}' ({Encoding}, {ScaleLayout}) cannot be sliced by columns.");

        bool endsAtEdge = colOffset + colCount == LogicalCols;
        if (colOffset % Geometry.BlockCols != 0 || (colCount % Geometry.BlockCols != 0 && !endsAtEdge))
            throw new NotSupportedException(
                $"'{weightKey}' ({Encoding}, {Geometry} blocks) can only be sliced on {Geometry.BlockCols}-column boundaries; "
                + $"columns [{colOffset}..{colOffset + colCount}) split a scale block.");
        if (ElementsPerByte == 2 && (colOffset % 2 != 0 || colCount % 2 != 0))
            throw new NotSupportedException(
                $"'{weightKey}' packs two 4-bit values per byte, so columns [{colOffset}..{colOffset + colCount}) are not byte aligned.");

        return this with { LogicalCols = colCount, ScaleColOffset = ScaleColOffset + colOffset / Geometry.BlockCols };
    }
}
