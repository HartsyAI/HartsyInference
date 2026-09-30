namespace HartsyInference.Core.Engram;

/// <summary>The DwarfStar GGUF: an I8 tensor of ne = [264, rows], plain row-major, each row 256 fp8 e4m3 codes followed by 8 e8m0 scales.</summary>
public sealed class GgufRow264Layout : IEngramRowLayout
{
    /// <summary>Bytes of one row.</summary>
    public const int RowBytes = Fp8E8M0RowDecoder.Columns + Fp8E8M0RowDecoder.Blocks;

    private readonly EngramRowSlice _row;

    /// <param name="rows">Table rows.</param>
    /// <param name="tensorFileOffset">Absolute offset of the tensor data in the GGUF file (aligned data start plus the tensor's offset).</param>
    public GgufRow264Layout(long rows, long tensorFileOffset)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegative(tensorFileOffset);
        Rows = rows;
        _row = new EngramRowSlice(tensorFileOffset, RowBytes);
    }

    /// <inheritdoc/>
    public string Name => "gguf-row264";

    /// <inheritdoc/>
    public long Rows { get; }

    /// <inheritdoc/>
    public int Dim => Fp8E8M0RowDecoder.Columns;

    /// <inheritdoc/>
    public int SliceCount => 1;

    /// <inheritdoc/>
    public int PackedRowBytes => RowBytes;

    /// <inheritdoc/>
    public EngramRowSlice GetSlice(int index) => index == 0 ? _row : throw new ArgumentOutOfRangeException(nameof(index));

    /// <inheritdoc/>
    public void DecodeRow(ReadOnlySpan<byte> packedRow, Span<ushort> destBf16)
    {
        if (packedRow.Length != RowBytes)
            throw new ArgumentException($"A packed row is {RowBytes} bytes, got {packedRow.Length}.", nameof(packedRow));
        Fp8E8M0RowDecoder.Decode(packedRow[..Fp8E8M0RowDecoder.Columns], packedRow[Fp8E8M0RowDecoder.Columns..], destBf16);
    }
}
