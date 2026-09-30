namespace HartsyInference.Core.Tensors.Quant;

/// <summary>The weight region one scale covers: <paramref name="BlockRows"/> output rows by <paramref name="BlockCols"/> input columns.</summary>
public readonly record struct BlockGeometry(int BlockRows, int BlockCols)
{
    /// <summary>The scale-tensor shape this geometry implies for a <paramref name="rows"/> x <paramref name="cols"/> weight.</summary>
    public (long Rows, long Cols) ScaleShape(long rows, long cols) =>
        ((rows + BlockRows - 1) / BlockRows, (cols + BlockCols - 1) / BlockCols);

    /// <inheritdoc />
    public override string ToString() => $"{BlockRows}x{BlockCols}";
}
