namespace HartsyInference.Core.Engram;

/// <summary>How one checkpoint format stores an Engram table row and how to turn it into bf16.</summary>
/// <remarks>A row is read as the concatenation of its planes ("packed row"), in <see cref="GetSlice"/> order, and decoded from that.</remarks>
public interface IEngramRowLayout
{
    /// <summary>Short format name, for messages.</summary>
    string Name { get; }

    /// <summary>Rows in the table.</summary>
    long Rows { get; }

    /// <summary>Elements per decoded row.</summary>
    int Dim { get; }

    /// <summary>Number of byte planes a row is read from.</summary>
    int SliceCount { get; }

    /// <summary>Bytes of a packed row (all planes).</summary>
    int PackedRowBytes { get; }

    /// <summary>The <paramref name="index"/>-th plane.</summary>
    EngramRowSlice GetSlice(int index);

    /// <summary>Decodes one packed row into <see cref="Dim"/> bf16 values (raw bits), rounded to nearest even.</summary>
    void DecodeRow(ReadOnlySpan<byte> packedRow, Span<ushort> destBf16);
}
