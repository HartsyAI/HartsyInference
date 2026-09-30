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

    /// <summary>Weight columns (input features) after dequantization; twice the packed byte width for 4-bit encodings, and eight per U32 word for MLX.</summary>
    public required long LogicalCols { get; init; }

    /// <summary>Block scales, <c>[ceil(rows/blockRows), &gt;= ceil(cols/blockCols)]</c> in <see cref="Quant.ScaleLayout.RowMajorBlocks"/>.</summary>
    public Tensor? Scale { get; init; }

    /// <summary>MLX affine per-group bias, same shape and dtype as <see cref="Scale"/>.</summary>
    public Tensor? Bias { get; init; }

    /// <summary>NVFP4 per-tensor F32 scalar.</summary>
    public Tensor? GlobalScale { get; init; }

    /// <summary>NVFP4 activation scale; the dequant path never reads it (see <see cref="QuantExecutionPolicy"/>).</summary>
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
        QuantEncoding.AffineInt8 => "recipe-affine-int8",
        QuantEncoding.Exl3Trellis => "recipe-exl3",
        QuantEncoding.Gguf => "recipe-gguf",
        _ => throw new NotSupportedException($"No format name for {Encoding}."),
    };

    /// <summary>Elements per packed byte: 2 for the 4-bit encodings, 1 otherwise.</summary>
    public int ElementsPerByte => Encoding is QuantEncoding.Mxfp4E8M0 or QuantEncoding.Nvfp4 or QuantEncoding.AffineInt4 ? 2 : 1;

    /// <summary>The tensors a BF16 dequant reads besides the packed weight: scale, global scale and bias, whichever the encoding has, and for EXL3 the <c>suh</c>/<c>svh</c> sign-scale vectors. <see cref="InputScale"/> is not one of them, nor is EXL3's <c>mcg</c> (a constant the host validates, never read by a kernel).</summary>
    public IEnumerable<Tensor> DequantCompanions()
    {
        if (Scale is not null) yield return Scale;
        if (GlobalScale is not null) yield return GlobalScale;
        if (Bias is not null) yield return Bias;
        if (Exl3 is not null)
        {
            yield return Exl3.Suh;
            yield return Exl3.Svh;
        }
    }

    /// <summary>Narrows the recipe to <paramref name="rowCount"/> rows from <paramref name="rowOffset"/>, viewing the matching scale rows.</summary>
    public QuantRecipe SliceRows(long rowOffset, long rowCount, string weightKey)
    {
        if (rowOffset < 0 || rowCount < 0 || rowOffset + rowCount > LogicalRows)
            throw new NotSupportedException($"'{weightKey}': rows [{rowOffset}..{rowOffset + rowCount}) are outside [0..{LogicalRows}).");
        if (Encoding == QuantEncoding.Exl3Trellis)
        {
            // Rows are the output axis. The trellis is [in/16, out/16, 16*bits], so an output window is a strided set of byte runs,
            // not a contiguous slice the caller could hand over; the whole matrix is the only window a view can describe.
            if (rowOffset == 0 && rowCount == LogicalRows) return this;
            throw new NotSupportedException(
                $"'{weightKey}' is EXL3: its trellis is [in/16, out/16, 16*bits], so output rows [{rowOffset}..{rowOffset + rowCount}) of {LogicalRows} "
                + "are not a contiguous byte range. Decode the whole matrix, or window it by input columns in multiples of 128.");
        }

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
        if (Encoding == QuantEncoding.Exl3Trellis)
            return SliceExl3Cols(colOffset, colCount, weightKey);
        if (ScaleLayout == ScaleLayout.Swizzled128)
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

    // Columns are the input axis, the leading trellis dimension, so a window is a contiguous run of [in/16] rows the caller slices with
    // Tensor.SliceRows(colOffset / 16, colCount / 16). The Hadamard is block-diagonal over 128 inputs, so a window on a 128 boundary
    // decodes exactly as those columns of the whole matrix; anything else would mix inputs from outside the window.
    private QuantRecipe SliceExl3Cols(long colOffset, long colCount, string weightKey)
    {
        if (colOffset % Exl3Format.HadamardBlock != 0 || colCount % Exl3Format.HadamardBlock != 0 || colCount == 0)
            throw new NotSupportedException(
                $"'{weightKey}' is EXL3: input columns [{colOffset}..{colOffset + colCount}) split a {Exl3Format.HadamardBlock}-wide Hadamard block; "
                + $"windows must start and end on multiples of {Exl3Format.HadamardBlock}.");
        Exl3Companions exl3 = Exl3 ?? throw new NotSupportedException($"'{weightKey}' is EXL3 but the recipe carries no suh/svh/mcg companions.");
        return this with { LogicalCols = colCount, Exl3 = exl3 with { Suh = exl3.Suh.SliceRows(colOffset, colCount) } };
    }
}
