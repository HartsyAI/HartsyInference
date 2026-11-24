using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Hy4Flash;

/// <summary>Interleaved 3D M-RoPE rows indexed by cache slot: each slot carries a (time, height, width) position triple and its row takes every pair frequency from one of the three planes.
/// Frequency <c>i</c> reads the height plane when <c>i % 3 == 1</c> and <c>i &lt; 3 * section[1]</c>, the width plane when <c>i % 3 == 2</c> and <c>i &lt; 3 * section[2]</c>, the time plane otherwise
/// (<c>recomposition_frequencies</c> of the reference). With three equal planes the row equals plain RoPE.</summary>
public sealed class Hy4FlashMropeRows : IRopeRowSource
{
    private readonly float[] _inverse;
    private readonly int[] _plane;
    private readonly float[] _cos;
    private readonly float[] _sin;

    /// <param name="halfDim">Pairs per row, half the rotated width.</param>
    /// <param name="capacity">Cache slots.</param>
    /// <param name="theta">RoPE base.</param>
    /// <param name="section">Time, height and width pair counts, summing to <paramref name="halfDim"/>.</param>
    public Hy4FlashMropeRows(int halfDim, int capacity, double theta, IReadOnlyList<int> section)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(halfDim, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(capacity, 1);
        if (section.Count != 3 || section.Sum() != halfDim) throw new ArgumentException("The M-RoPE section must hold three counts summing to halfDim.", nameof(section));
        HalfDim = halfDim;
        Length = capacity;
        _inverse = new float[halfDim];
        _plane = new int[halfDim];
        for (int i = 0; i < halfDim; i++)
        {
            _inverse[i] = (float)Math.Pow(theta, -2.0 * i / (2.0 * halfDim));
            _plane[i] = i % 3 == 1 && i < 3 * section[1] ? 1 : i % 3 == 2 && i < 3 * section[2] ? 2 : 0;
        }
        _cos = new float[capacity * halfDim];
        _sin = new float[capacity * halfDim];
    }

    /// <inheritdoc />
    public int HalfDim { get; }

    /// <inheritdoc />
    public int Length { get; }

    /// <summary>Sets the position triple of cache slot <paramref name="slot"/> and rebuilds its row.</summary>
    public void SetPosition(int slot, int time, int height, int width)
    {
        Span<int> planes = [time, height, width];
        for (int i = 0; i < HalfDim; i++)
        {
            double angle = (double)planes[_plane[i]] * _inverse[i];
            _cos[slot * HalfDim + i] = (float)Math.Cos(angle);
            _sin[slot * HalfDim + i] = (float)Math.Sin(angle);
        }
    }

    /// <inheritdoc />
    public ReadOnlySpan<float> CosRow(int pos) => _cos.AsSpan(pos * HalfDim, HalfDim);

    /// <inheritdoc />
    public ReadOnlySpan<float> SinRow(int pos) => _sin.AsSpan(pos * HalfDim, HalfDim);
}
