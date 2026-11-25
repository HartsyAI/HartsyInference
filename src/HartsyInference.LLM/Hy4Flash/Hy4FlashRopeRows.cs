using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Hy4Flash;

/// <summary>Plain interleaved-pair rotary rows of one base over every position: angle <c>pos * theta^(-2i/width)</c> for pair <c>i</c>. A text-only
/// Hy4 Flash sequence feeds all three M-RoPE planes the same positions, which reduces its interleaved M-RoPE to exactly this table.</summary>
public sealed class Hy4FlashRopeRows : IRopeRowSource
{
    private readonly float[] _cos;
    private readonly float[] _sin;

    /// <param name="halfDim">Pairs per row, half the rotated width.</param>
    /// <param name="length">Positions covered.</param>
    /// <param name="theta">RoPE base.</param>
    public Hy4FlashRopeRows(int halfDim, int length, double theta)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(halfDim, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(length, 1);
        HalfDim = halfDim;
        Length = length;
        _cos = new float[length * halfDim];
        _sin = new float[length * halfDim];
        for (int i = 0; i < halfDim; i++)
        {
            double inverse = Math.Pow(theta, -2.0 * i / (2.0 * halfDim));
            for (int pos = 0; pos < length; pos++)
            {
                double angle = pos * inverse;
                _cos[pos * halfDim + i] = (float)Math.Cos(angle);
                _sin[pos * halfDim + i] = (float)Math.Sin(angle);
            }
        }
    }

    /// <inheritdoc />
    public int HalfDim { get; }

    /// <inheritdoc />
    public int Length { get; }

    /// <inheritdoc />
    public ReadOnlySpan<float> CosRow(int pos) => _cos.AsSpan(pos * HalfDim, HalfDim);

    /// <inheritdoc />
    public ReadOnlySpan<float> SinRow(int pos) => _sin.AsSpan(pos * HalfDim, HalfDim);
}
