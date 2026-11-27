namespace HartsyInference.Core.Backends;

/// <summary>The Fourier feature embedding of <see cref="IBackend.FourierEmbed"/>: each input coordinate <c>x</c> contributes
/// <c>sin(x · 2^b)</c> and <c>cos(x · 2^b)</c> for every band <c>b</c> below <see cref="Bands"/>, after the coordinates themselves
/// when <see cref="IncludeInput"/> is set. A row is <c>[x₀ … | sin, coordinate-major then band | cos, the same order]</c>.</summary>
public readonly record struct FourierEmbedSpec(int InputDims, int Bands, bool IncludeInput)
{
    /// <summary>Elements per embedded row.</summary>
    public int Width => (IncludeInput ? InputDims : 0) + 2 * InputDims * Bands;
}
