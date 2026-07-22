using System.Collections.Concurrent;
using HartsyInference.Core.Logging;

namespace HartsyInference.Engine.Audio;

/// <summary>Caches loaded audio runners by resolved model key so a repeated request reuses the resident pipeline.
/// One instance per category; it registers itself with <see cref="AudioRuntime"/> so a model switch under memory
/// pressure can drop everything that is not the incoming model.</summary>
internal sealed class AudioRunnerCache<TRunner> : IAudioRunnerCache
    where TRunner : class, IDisposable
{
    private readonly ConcurrentDictionary<string, TRunner> _entries = new(StringComparer.Ordinal);
    private readonly object _loadLock = new();

    /// <summary>Creates the cache and registers it for memory-pressure eviction.</summary>
    internal AudioRunnerCache() => AudioRuntime.Register(this);

    /// <summary>Returns the cached runner for <paramref name="key"/>, loading it when absent. The double-check keeps
    /// two concurrent callers from keeping two copies of the same model resident.</summary>
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

    /// <inheritdoc/>
    public void UnloadAllExcept(string? keepKey)
    {
        foreach (string key in _entries.Keys)
        {
            if (string.Equals(key, keepKey, StringComparison.Ordinal))
            {
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
}
