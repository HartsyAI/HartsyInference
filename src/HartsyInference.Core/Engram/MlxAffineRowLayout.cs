using System.Buffers.Binary;

namespace HartsyInference.Core.Engram;

/// <summary>The MLX 4-bit conversion: U32 <c>[rows, 32]</c> weights (eight nibbles per word, lowest first), F32 <c>[rows, 4]</c> scales and F32 <c>[rows, 4]</c> biases (group size 64), <c>w = q * scale + bias</c>.</summary>
/// <remarks>The three tensors live in three shard files, so this layout reads from three byte sources: 0 weights, 1 scales, 2 biases.</remarks>
public sealed class MlxAffineRowLayout : IEngramRowLayout
{
    private const int Columns = 256;
    private const int Words = 32;
    private const int Groups = 4;
    private const int GroupSize = 64;
    private const int WeightBytes = Words * sizeof(uint);
    private const int GroupBytes = Groups * sizeof(float);

    private readonly EngramRowSlice _weight;
    private readonly EngramRowSlice _scales;
    private readonly EngramRowSlice _biases;

    /// <param name="rows">Table rows.</param>
    /// <param name="weightFileOffset">Absolute offset of the weight tensor's data in its shard.</param>
    /// <param name="scalesFileOffset">Absolute offset of the scales tensor's data in its shard.</param>
    /// <param name="biasesFileOffset">Absolute offset of the biases tensor's data in its shard.</param>
    public MlxAffineRowLayout(long rows, long weightFileOffset, long scalesFileOffset, long biasesFileOffset)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegative(weightFileOffset);
        ArgumentOutOfRangeException.ThrowIfNegative(scalesFileOffset);
        ArgumentOutOfRangeException.ThrowIfNegative(biasesFileOffset);
        Rows = rows;
        _weight = new EngramRowSlice(weightFileOffset, WeightBytes, 0);
        _scales = new EngramRowSlice(scalesFileOffset, GroupBytes, 1);
        _biases = new EngramRowSlice(biasesFileOffset, GroupBytes, 2);
    }

    /// <inheritdoc/>
    public string Name => "mlx-affine-4bit";

    /// <inheritdoc/>
    public long Rows { get; }

    /// <inheritdoc/>
    public int Dim => Columns;

    /// <inheritdoc/>
    public int SliceCount => 3;

    /// <inheritdoc/>
    public int PackedRowBytes => WeightBytes + 2 * GroupBytes;

    /// <inheritdoc/>
    public EngramRowSlice GetSlice(int index) => index switch
    {
        0 => _weight,
        1 => _scales,
        2 => _biases,
        _ => throw new ArgumentOutOfRangeException(nameof(index)),
    };

    /// <inheritdoc/>
    public void DecodeRow(ReadOnlySpan<byte> packedRow, Span<ushort> destBf16)
    {
        if (packedRow.Length != PackedRowBytes)
            throw new ArgumentException($"A packed row is {PackedRowBytes} bytes, got {packedRow.Length}.", nameof(packedRow));
        if (destBf16.Length < Columns)
            throw new ArgumentException("The destination holds fewer than 256 values.", nameof(destBf16));
        ReadOnlySpan<byte> scales = packedRow.Slice(WeightBytes, GroupBytes);
        ReadOnlySpan<byte> biases = packedRow.Slice(WeightBytes + GroupBytes, GroupBytes);
        for (int word = 0; word < Words; word++)
        {
            int group = word * 8 / GroupSize;
            float scale = BinaryPrimitives.ReadSingleLittleEndian(scales[(group * sizeof(float))..]);
            float bias = BinaryPrimitives.ReadSingleLittleEndian(biases[(group * sizeof(float))..]);
            uint bits = BinaryPrimitives.ReadUInt32LittleEndian(packedRow[(word * sizeof(uint))..]);
            int column = word * 8;
            for (int nibble = 0; nibble < 8; nibble++)
            {
                float scaled = (float)((bits >> (4 * nibble)) & 0xFu) * scale;
                destBf16[column + nibble] = Bf16Rounding.FromSingle(scaled + bias);
            }
        }
    }
}
