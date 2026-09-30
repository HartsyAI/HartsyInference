namespace HartsyInference.Engine.Audio;

/// <summary>Identifies the resident runner an audio job is about to use: the category cache it lives in plus its bare cache key. <see cref="AudioRuntime"/> needs both, because the memory-pressure sweep keeps the incoming model by asking ITS cache for the bare key while every other cache is cleared. A single prefixed string could not say which cache the key belongs to, and comparing a prefixed job key against bare cache keys is exactly the mismatch that used to evict the incoming model as well.</summary>
internal readonly record struct AudioJob(IAudioRunnerCache Cache, string Key)
{
    /// <summary>The prefixed identity used for same-model detection and log lines, e.g. <c>tts:hexgrad/Kokoro-82M</c>. Category prefixes may themselves contain colons (<c>fx:demucs</c>), so nothing splits this string back apart.</summary>
    public string ModelKey => $"{Cache.Category}:{Key}";
}
