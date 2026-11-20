namespace HartsyInference.Diffusion.Utilities;

/// <summary>A signal-to-noise calibration for a model that was trained on one representation and is run on another that
/// carries <c>c</c> times as much signal per unit of noise (AsymFlow; PixVid's pixel patches against Wan2.1's VAE
/// latents). At noise level σ the pixel sample projects to <c>c·(1 − σ)·z + σ·ε</c>, which has the SNR of the latent flow at
/// <see cref="Sigma"/>, and scaling the input by <see cref="InputScale"/> matches its magnitude as well.
/// <para>Two calibrations compose by multiplying their scales, which is how a 2x2 average-pooled input (scale 2) stacks on a
/// patch-embedding calibration. <c>default</c> is the identity (scale 1) and equals <c>new SnrCalibration(1f)</c>.</para></summary>
public readonly struct SnrCalibration : IEquatable<SnrCalibration>
{
    private readonly float _scale;

    /// <summary>Creates a calibration with scale <paramref name="scale"/> (c). Must be positive and finite.</summary>
    public SnrCalibration(float scale)
    {
        if (!(scale > 0f) || !float.IsFinite(scale))
            throw new ArgumentOutOfRangeException(nameof(scale), "The calibration scale must be positive and finite.");
        _scale = scale;
    }

    /// <summary>The scale c. 1.0 is the identity, and the value of <c>default</c>.</summary>
    public float Scale => _scale > 0f ? _scale : 1f;

    /// <summary>The noise level the model is conditioned on for sample noise level <paramref name="sigma"/>:
    /// <c>σ / (c·(1 − σ) + σ)</c>. 0 maps to 0 and 1 to 1.</summary>
    public float Sigma(float sigma) => sigma * InputScale(sigma);

    /// <summary>The factor the model input is multiplied by: <c>1 / (c·(1 − σ) + σ)</c>.</summary>
    public float InputScale(float sigma) => 1f / (Scale * (1f - sigma) + sigma);

    /// <summary>The calibration that applies this one and then <paramref name="other"/>: the scales multiply.</summary>
    public SnrCalibration Compose(SnrCalibration other) => new(Scale * other.Scale);

    /// <inheritdoc/>
    public bool Equals(SnrCalibration other) => Scale == other.Scale;

    /// <inheritdoc/>
    public override bool Equals(object? obj) => obj is SnrCalibration other && Equals(other);

    /// <inheritdoc/>
    public override int GetHashCode() => Scale.GetHashCode();

    /// <summary>Whether two calibrations have the same scale.</summary>
    public static bool operator ==(SnrCalibration left, SnrCalibration right) => left.Equals(right);

    /// <summary>Whether two calibrations differ in scale.</summary>
    public static bool operator !=(SnrCalibration left, SnrCalibration right) => !left.Equals(right);
}
