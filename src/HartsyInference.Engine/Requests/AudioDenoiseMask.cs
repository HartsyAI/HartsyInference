namespace HartsyInference.Engine.Requests;

/// <summary>Continuous audio-latent denoise values and the source waveform preserved below one.</summary>
public sealed record AudioDenoiseMask
{
    /// <summary>Mask samples in [0,1], where one generates and zero preserves.</summary>
    public required IReadOnlyList<float> Values { get; init; }

    /// <summary>Mask sample cadence in hertz. H3's native audio-latent cadence is 40 Hz.</summary>
    public float Rate { get; init; } = 40f;

    /// <summary>Source audio required whenever any resampled mask value is below one.</summary>
    public AudioClip? Source { get; init; }
}
