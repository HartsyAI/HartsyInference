namespace HartsyInference.Engine.Audio;

/// <summary>Identifies the resident runner an audio job is about to use: the category cache it lives in plus its bare cache key. <see cref="AudioRuntime"/> needs both, because the memory-pressure sweep keeps the incoming model by asking ITS cache for the bare key while every other cache is cleared. A single prefixed string could not say which cache the key belongs to, and comparing a prefixed job key against bare cache keys is exactly the mismatch that used to evict the incoming model as well.</summary>
/// <param name="Cache">The category cache the runner lives in.</param>
/// <param name="Key">The runner's bare cache key.</param>
/// <param name="EstimateWeightBytes">The device bytes the runner's weights take once loaded, sized from the files already on
/// disk; 0 when nothing is there to size. Called only when a switch has to decide whether a model that is not loaded yet
/// fits, so it must stay local and cheap. Null for a family with nothing to size from.</param>
internal readonly record struct AudioJob(IAudioRunnerCache Cache, string Key, Func<long>? EstimateWeightBytes = null)
{
    /// <summary>The prefixed identity used for same-model detection and log lines, e.g. <c>tts:hexgrad/Kokoro-82M</c>. Category prefixes may themselves contain colons (<c>fx:demucs</c>), so nothing splits this string back apart.</summary>
    public string ModelKey => $"{Cache.Category}:{Key}";
}
