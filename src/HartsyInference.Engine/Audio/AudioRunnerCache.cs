using System.Collections.Concurrent;
using HartsyInference.Core.Logging;

namespace HartsyInference.Engine.Audio;

/// <summary>Caches loaded audio runners by resolved model key so a repeated request reuses the resident pipeline. One instance per category per <see cref="AudioRuntime"/> (i.e. per engine), which owns it and sweeps it on memory pressure — runners bind device state, so caches must never be shared across engines/devices. A <see cref="Pin"/> exempts a key from that sweep for callers that must keep a model warm across other jobs (a voice session alternating STT and TTS, an "always resident" setting); the engine's own release path ignores pins because a runner bound to a device that is going away cannot stay.</summary>
internal sealed class AudioRunnerCache<TRunner>(string category) : IAudioRunnerCache
    where TRunner : class, IDisposable
{
    private readonly ConcurrentDictionary<string, TRunner> _entries = new(StringComparer.Ordinal);
    private readonly ConcurrentDictionary<string, int> _pins = new(StringComparer.Ordinal);
    private readonly object _loadLock = new();

    /// <inheritdoc/>
    public string Category { get; } = category;

    /// <summary>Returns the cached runner for <paramref name="key"/>, loading it when absent. The double-check keeps two concurrent callers from keeping two copies of the same model resident.</summary>
    internal async Task<TRunner> GetOrLoadAsync(string key, Func<CancellationToken, Task<TRunner>> load, CancellationToken cancel)
    {
        if (_entries.TryGetValue(key, out TRunner? existing))
        {
            return existing;
        }
        TRunner loaded = await load(cancel).ConfigureAwait(false);
        lock (_loadLock)
        {
            if (_entries.TryGetValue(key, out TRunner? raced))
            {
                loaded.Dispose();
                return raced;
            }
            _entries[key] = loaded;
            return loaded;
        }
    }

    /// <summary>Whether a runner for <paramref name="key"/> is currently resident. Diagnostics and test seam; callers that need the runner use <see cref="GetOrLoadAsync"/>.</summary>
    internal bool IsResident(string key) => _entries.ContainsKey(key);

    /// <inheritdoc/>
    public IDisposable Pin(string key)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(key);
        _pins.AddOrUpdate(key, 1, static (_, holds) => holds + 1);
        return new PinHandle(this, key);
    }

    /// <inheritdoc/>
    public bool IsPinned(string key) => _pins.TryGetValue(key, out int holds) && holds > 0;

    /// <inheritdoc/>
    public void UnloadAllExcept(string? keepKey, bool includePinned = false)
    {
        foreach (string key in _entries.Keys)
        {
            if (string.Equals(key, keepKey, StringComparison.Ordinal))
            {
                continue;
            }
            if (!includePinned && IsPinned(key))
            {
                Logs.Debug($"[Audio] Keeping pinned {Category} model '{key}' resident through eviction.");
                continue;
            }
            if (!_entries.TryRemove(key, out TRunner? runner))
            {
                continue;
            }
            try
            {
                runner.Dispose();
            }
            catch (Exception ex)
            {
                Logs.Warning($"[Audio] Unloading resident model '{key}' failed: {ex.Message}");
            }
        }
    }

    /// <summary>Releases one hold on <paramref name="key"/>, removing the entry at zero so <see cref="IsPinned"/> stays a plain lookup.</summary>
    private void Unpin(string key)
    {
        while (_pins.TryGetValue(key, out int holds))
        {
            if (holds <= 1)
            {
                if (_pins.TryRemove(new KeyValuePair<string, int>(key, holds)))
                {
                    return;
                }
            }
            else if (_pins.TryUpdate(key, holds - 1, holds))
            {
                return;
            }
        }
    }

    /// <summary>One hold on one key; disposing twice releases once.</summary>
    private sealed class PinHandle(AudioRunnerCache<TRunner> owner, string key) : IDisposable
    {
        private int _released;

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _released, 1) == 0)
            {
                owner.Unpin(key);
            }
        }
    }
}
