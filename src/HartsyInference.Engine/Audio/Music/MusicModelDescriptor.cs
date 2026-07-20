using HartsyInference.Core.Backends;

namespace HartsyInference.Engine.Audio;

/// <summary>Per-model specifics for the generic music path.</summary>
internal sealed class MusicModelDescriptor
{
    /// <summary>True for HF-auto-downloaded models (MusicGen/AudioGen/HeartMuLa); false for placed checkpoints.</summary>
    internal required bool ManagesOwnWeights { get; init; }

    /// <summary>Stable cache key for a resolved selector (an HF repo id, or the local checkpoint path).</summary>
    internal required Func<AudioModelSelector, string> CacheKey { get; init; }

    /// <summary>Loads the model into a runner. The backend is needed by pipelines that bind to a device at
    /// construction (ACE-Step); the others ignore it.</summary>
    internal required Func<IBackend, AudioModelSelector, CancellationToken, Task<IMusicRunner>> LoadAsync { get; init; }
}
