using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Hy4Flash;

/// <summary>Row access to one large embedding table (an Engram layer table or the token embedding), <c>[rows, dim]</c> in F32 or BF16, read in place by 64-bit offset so a memory-mapped 31.7 GB table
/// is never copied or spanned whole.</summary>
public sealed unsafe class Hy4FlashRowTable
{
    private readonly Tensor _table;

    /// <param name="table">The flat per-layer table; borrowed, the caller keeps it alive.</param>
    public Hy4FlashRowTable(Tensor table)
    {
        ArgumentNullException.ThrowIfNull(table);
        if (table.Shape.Rank != 2 || (table.DType != DType.F32 && table.DType != DType.BF16))
            throw new ArgumentException("An Engram table is a rank-2 F32 or BF16 tensor.", nameof(table));
        _table = table;
    }

    /// <summary>Rows in the table.</summary>
    public long Rows => _table.Shape[0];

    /// <summary>Elements per row.</summary>
    public int Dim => checked((int)_table.Shape[1]);

    /// <summary>Copies row <paramref name="row"/> into <paramref name="destination"/> as F32.</summary>
    public void Read(long row, Span<float> destination)
    {
        if ((ulong)row >= (ulong)Rows) throw new ArgumentOutOfRangeException(nameof(row), $"Row {row} is outside the {Rows}-row table.");
        if (destination.Length != Dim) throw new ArgumentException($"The destination must hold {Dim} values.", nameof(destination));
        long start = row * Dim;
        if (_table.DType == DType.F32)
        {
            new ReadOnlySpan<float>((float*)_table.DataPointer + start, Dim).CopyTo(destination);
            return;
        }
        ReadOnlySpan<ushort> bits = new((ushort*)_table.DataPointer + start, Dim);
        for (int i = 0; i < Dim; i++) destination[i] = BFloat16Bits.ToFloat(bits[i]);
    }
}
