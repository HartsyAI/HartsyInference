using HartsyInference.Audio.Cache;

namespace HartsyInference.Engine.Audio;

/// <summary>Per-model specifics for the generic speech path: how to turn a variant hint into a HuggingFace repo (a constant when the pipeline resolves its own weights), and how to load it into an <see cref="ITtsRunner"/>.</summary>
internal sealed class TtsModelDescriptor
{
    /// <summary>Maps the request's variant hint to a HuggingFace repo id.</summary>
    internal required Func<string, string> ResolveRepo { get; init; }

    /// <summary>Loads the model (downloading on first use) into a uniform runner. The context carries the primary backend plus the shard placement CosyVoice's Qwen2 LM consumes; every other TTS family ignores it.</summary>
    internal required Func<TtsLoadContext, string, CancellationToken, Task<ITtsRunner>> LoadAsync { get; init; }

    /// <summary>True when the requested VOICE selects which weights to load (Piper ships one .onnx per voice), so the voice must take part in the cache key and be passed as the load variant. False for models whose single checkpoint carries every voice.</summary>
    internal bool VoiceSelectsWeights { get; init; }

    /// <summary>The files this model pulls, or null for a family whose weight list is still implicit in its
    /// load path. Set it and the model becomes pre-downloadable; leave it and prefetch reports the family as
    /// unsupported rather than pretending to install it. Async because a sharded family has to read its shard
    /// index before it can name its own files.</summary>
    internal Func<string, CancellationToken, Task<IReadOnlyList<AudioModelFile>>>? ResolveFiles { get; init; }
}
