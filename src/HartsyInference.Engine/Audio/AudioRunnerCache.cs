using System.Collections.Concurrent;
using HartsyInference.Core.Logging;

namespace HartsyInference.Engine.Audio;

/// <summary>Caches loaded audio runners by resolved model key so a repeated request reuses the resident pipeline.
/// One instance per category per <see cref="AudioRuntime"/> (i.e. per engine), which owns it and sweeps it on
/// memory pressure — runners bind device state, so caches must never be shared across engines/devices.</summary>
internal sealed class AudioRunnerCache<TRunner> : IAudioRunnerCache
    where TRunner : class, IDisposable
{
    private readonly ConcurrentDictionary<string, TRunner> _entries = new(StringComparer.Ordinal);
    private readonly object _loadLock = new();

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
