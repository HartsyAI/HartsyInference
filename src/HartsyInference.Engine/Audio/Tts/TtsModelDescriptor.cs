using HartsyInference.Audio.Cache;

namespace HartsyInference.Engine.Audio;

/// <summary>Per-model specifics for the generic speech path: how to turn a variant hint into a HuggingFace repo (a constant when the pipeline resolves its own weights), and how to load it into an <see cref="ITtsRunner"/>.</summary>
internal sealed class TtsModelDescriptor
{
    /// <summary>Maps the request's variant hint to a HuggingFace repo id.</summary>
    internal required Func<string, string> ResolveRepo { get; init; }

    /// <summary>Maps the variant hint to what tells two loads of one repo apart in the runner cache (Fish Audio Drama 3's
    /// precision), or to an empty string when the repo alone names the runner. Null for a model with one load per repo.</summary>
    internal Func<string, string>? ResolveCacheTag { get; init; }

    /// <summary>Loads the model (downloading on first use) into a uniform runner. The context carries the primary backend plus the shard placement CosyVoice's Qwen2 LM consumes; every other TTS family ignores it.</summary>
    internal required Func<TtsLoadContext, string, CancellationToken, Task<ITtsRunner>> LoadAsync { get; init; }

    /// <summary>Checks the length of the request's voice reference before the job is queued and throws when this family cannot
    /// clone from it, so a rejected request never takes the device gate, evicts a model or pays the download and load. Null
    /// when any length is acceptable.</summary>
    internal Action<double>? ValidateReferenceSeconds { get; init; }

    /// <summary>Why this family takes no voice reference at all, or null when it accepts one. Unlike
    /// <see cref="ValidateReferenceSeconds"/>, this is checked on any reference clip whatever its container, before the job is
    /// queued, so a clip whose length the WAV header cannot give is still refused.</summary>
    internal string? ReferenceRejection { get; init; }

    /// <summary>True when the requested VOICE selects which weights to load (Piper ships one .onnx per voice), so the voice must take part in the cache key and be passed as the load variant. False for models whose single checkpoint carries every voice.</summary>
    internal bool VoiceSelectsWeights { get; init; }

    /// <summary>True when the runner widens F16/BF16 weights to F32 as it loads them, so a half-precision checkpoint
    /// takes twice its file size on the device. Read only to size the model for the engine's switch check.</summary>
    internal bool PromotesHalfToF32 { get; init; }

    /// <summary>The files this model pulls, or null for a family whose weight list is still implicit in its
    /// load path. Set it and the model becomes pre-downloadable; leave it and prefetch reports the family as
    /// unsupported rather than pretending to install it. Async because a sharded family has to read its shard
    /// index before it can name its own files.</summary>
    internal Func<string, CancellationToken, Task<IReadOnlyList<AudioModelFile>>>? ResolveFiles { get; init; }
}
