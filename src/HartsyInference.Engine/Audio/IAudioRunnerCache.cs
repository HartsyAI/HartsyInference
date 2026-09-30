namespace HartsyInference.Engine.Audio;

/// <summary>A per-category cache of resident audio pipelines that <see cref="AudioRuntime"/> can drop when a model switch would not fit in host RAM or VRAM.</summary>
internal interface IAudioRunnerCache
{
    /// <summary>Category prefix that names this cache in job keys and log lines, e.g. <c>tts</c> or <c>fx:demucs</c>.</summary>
    string Category { get; }

    /// <summary>Drops every resident runner except the one keyed <paramref name="keepKey"/> (the incoming model) and, unless <paramref name="includePinned"/> is set, any runner a caller holds a <see cref="Pin"/> on.</summary>
    void UnloadAllExcept(string? keepKey, bool includePinned = false);

    /// <summary>Holds <paramref name="key"/> resident through memory-pressure eviction until the returned handle is disposed. Pins nest, so each handle releases one hold, and they are name-level: pinning a key before it loads protects it once it has. An engine release or backend switch still unloads pinned runners — a pin survives memory pressure, not the device.</summary>
    IDisposable Pin(string key);

    /// <summary>Whether at least one <see cref="Pin"/> is currently held on <paramref name="key"/>.</summary>
    bool IsPinned(string key);
}
