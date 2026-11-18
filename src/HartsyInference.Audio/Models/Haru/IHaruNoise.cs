namespace HartsyInference.Audio.Models.Haru;

/// <summary>Where Haru's random draws come from: the codec posterior of every user frame and of the voice reference, and the noise the
/// MeanFlow head starts from. A seam, because the upstream torch generator cannot be reproduced from C#; a test replays recorded draws.</summary>
internal interface IHaruNoise
{
    /// <summary>Fills <paramref name="destination"/> with independent standard normal samples.</summary>
    void FillGaussian(Span<float> destination);
}
