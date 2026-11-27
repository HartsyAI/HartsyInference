namespace HartsyInference.LLM.Transformer;

/// <summary>A rotary table that rotates nothing: every position's cosines are one and its sines are zero. Models without position
/// encoding on a slice (Kimi's NoPE multi-head latent attention) pass it where a rope table is required, so the shared attention
/// code runs unchanged and the slice comes out exactly as it went in.</summary>
public sealed class IdentityRopeRows : IRopeRowSource
{
    private readonly float[] _cos;
    private readonly float[] _sin;

    /// <summary>Builds the table for <paramref name="length"/> positions over <paramref name="halfDim"/> element pairs.</summary>
    public IdentityRopeRows(int halfDim, int length)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(halfDim, 1);
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        HalfDim = halfDim;
        Length = length;
        _cos = new float[halfDim];
        _sin = new float[halfDim];
        Array.Fill(_cos, 1f);
    }

    /// <inheritdoc/>
    public int HalfDim { get; }

    /// <inheritdoc/>
    public int Length { get; }

    /// <inheritdoc/>
    public ReadOnlySpan<float> CosRow(int pos) => _cos;

    /// <inheritdoc/>
    public ReadOnlySpan<float> SinRow(int pos) => _sin;
}
