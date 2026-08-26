using HartsyInference.Audio.Cache;

namespace HartsyInference.Engine.Audio;

/// <summary>Per-model specifics for the generic music path.</summary>
internal sealed class MusicModelDescriptor
{
    /// <summary>The files this model pulls, or null for a family whose weight list is still implicit in its
    /// load path. Set it and the model becomes pre-downloadable; leave it and prefetch reports the family as
    /// unsupported rather than pretending to install it.</summary>
    internal Func<AudioModelSelector, CancellationToken, Task<IReadOnlyList<AudioModelFile>>>? ResolveFiles { get; init; }

    /// <summary>True for HF-auto-downloaded models (MusicGen/AudioGen/HeartMuLa); false for placed checkpoints.</summary>
    internal required bool ManagesOwnWeights { get; init; }

    /// <summary>Stable cache key for a resolved selector (an HF repo id, or the local checkpoint path).</summary>
    internal required Func<AudioModelSelector, string> CacheKey { get; init; }

    /// <summary>Loads the model into a runner. The context carries the primary backend (pipelines that bind to a device at construction — ACE-Step) plus the shard/quant placement the big-LM families (YuE) consume; the others ignore it.</summary>
    internal required Func<MusicLoadContext, AudioModelSelector, CancellationToken, Task<IMusicRunner>> LoadAsync { get; init; }
}
