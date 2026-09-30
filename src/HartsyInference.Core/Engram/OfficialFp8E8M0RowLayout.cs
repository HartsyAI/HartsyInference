namespace HartsyInference.Core.Engram;

/// <summary>The official checkpoint: an F8_E4M3 <c>[rows, 256]</c> tensor and an F8_E8M0 <c>[rows, 8]</c> scale tensor, both row-major in one safetensors shard.</summary>
public sealed class OfficialFp8E8M0RowLayout : IEngramRowLayout
{
    private readonly EngramRowSlice _weight;
    private readonly EngramRowSlice _scale;

    /// <param name="rows">Table rows.</param>
    /// <param name="weightFileOffset">Absolute offset of the fp8 tensor's data in the shard file.</param>
    /// <param name="scaleFileOffset">Absolute offset of the e8m0 tensor's data in the shard file.</param>
    public OfficialFp8E8M0RowLayout(long rows, long weightFileOffset, long scaleFileOffset)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegative(weightFileOffset);
        ArgumentOutOfRangeException.ThrowIfNegative(scaleFileOffset);
        Rows = rows;
        _weight = new EngramRowSlice(weightFileOffset, Fp8E8M0RowDecoder.Columns);
        _scale = new EngramRowSlice(scaleFileOffset, Fp8E8M0RowDecoder.Blocks);
    }

    /// <inheritdoc/>
    public string Name => "official-fp8-e8m0";

    /// <inheritdoc/>
    public long Rows { get; }

    /// <inheritdoc/>
    public int Dim => Fp8E8M0RowDecoder.Columns;

    /// <inheritdoc/>
    public int SliceCount => 2;

    /// <inheritdoc/>
    public int PackedRowBytes => Fp8E8M0RowDecoder.Columns + Fp8E8M0RowDecoder.Blocks;

    /// <inheritdoc/>
    public EngramRowSlice GetSlice(int index) => index switch
    {
        0 => _weight,
        1 => _scale,
        _ => throw new ArgumentOutOfRangeException(nameof(index)),
    };

    /// <inheritdoc/>
    public void DecodeRow(ReadOnlySpan<byte> packedRow, Span<ushort> destBf16)
    {
        if (packedRow.Length != PackedRowBytes)
            throw new ArgumentException($"A packed row is {PackedRowBytes} bytes, got {packedRow.Length}.", nameof(packedRow));
        Fp8E8M0RowDecoder.Decode(packedRow[..Fp8E8M0RowDecoder.Columns], packedRow[Fp8E8M0RowDecoder.Columns..], destBf16);
    }
}
