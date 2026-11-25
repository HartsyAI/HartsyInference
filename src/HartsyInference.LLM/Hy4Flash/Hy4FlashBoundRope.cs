using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Hy4Flash;

/// <summary>The rotary rows the layers read, for whichever sequence state the model is running. Each state owns its rows (<see cref="Hy4FlashSequenceState"/>); <see cref="Bind"/> points this source at the calling state's rows before that state's layers run, so the layers never read another state's positions.</summary>
internal sealed class Hy4FlashBoundRope : IRopeRowSource
{
    private Hy4FlashMropeRows? _rows;

    /// <param name="halfDim">Pairs per row, half the rotated width.</param>
    /// <param name="length">Positions covered.</param>
    public Hy4FlashBoundRope(int halfDim, int length)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(halfDim, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(length, 1);
        HalfDim = halfDim;
        Length = length;
    }

    /// <inheritdoc />
    public int HalfDim { get; }

    /// <inheritdoc />
    public int Length { get; }

    /// <summary>Points the source at <paramref name="rows"/> until the next bind.</summary>
    public void Bind(Hy4FlashMropeRows rows) => _rows = rows ?? throw new ArgumentNullException(nameof(rows));

    /// <inheritdoc />
    public ReadOnlySpan<float> CosRow(int pos) => Bound.CosRow(pos);

    /// <inheritdoc />
    public ReadOnlySpan<float> SinRow(int pos) => Bound.SinRow(pos);

    private Hy4FlashMropeRows Bound => _rows ?? throw new InvalidOperationException("No sequence state is bound to the rotary rows.");
}
