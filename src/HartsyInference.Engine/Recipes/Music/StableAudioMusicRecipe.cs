using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Engine.Recipes.Music;

/// <summary>Stable Audio Open Small: the ARC-distilled DiT runs 8 ping-pong steps; its trained window caps the clip (about 11.9 s).</summary>
public sealed class StableAudioMusicRecipe : IMusicRecipe
{
    /// <inheritdoc/>
    public string Name => "stableaudio";

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, Name, StringComparison.OrdinalIgnoreCase);

    /// <inheritdoc/>
    public MusicDefaults Defaults { get; } = new MusicDefaults { Duration = 10, Steps = 8 };

    /// <inheritdoc/>
    public MusicConstraints Constraints { get; } = new MusicConstraints
    {
        MaxDurationSeconds = StableAudioDitConfig.OpenSmall.MaxLatentTokens
            * (double)StableAudioDitConfig.OpenSmall.VaeDownsample / StableAudioDitConfig.OpenSmall.SampleRate,
    };
}
