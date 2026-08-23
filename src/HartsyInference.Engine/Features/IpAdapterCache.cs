namespace HartsyInference.Engine.Features;

/// <summary>Per-pipeline residency cache for <see cref="IpAdapterCacheEntry"/>, keyed case-insensitively by adapter file path; entries live until the owning pipeline is disposed.</summary>
internal sealed class IpAdapterCache : IDisposable
{
    private readonly Dictionary<string, IpAdapterCacheEntry> _entries = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>Cached IP-Adapter entry for <paramref name="path"/>, or null when not loaded yet.</summary>
    internal IpAdapterCacheEntry? Lookup(string path) =>
        _entries.TryGetValue(path, out IpAdapterCacheEntry? entry) ? entry : null;

    /// <summary>Stores a freshly loaded IP-Adapter entry for reuse across generations on this model.</summary>
    internal void Cache(IpAdapterCacheEntry entry)
    {
        ArgumentNullException.ThrowIfNull(entry);
        _entries[entry.FilePath] = entry;
    }

    /// <summary>Disposes every cached entry and empties the cache.</summary>
    public void Dispose()
    {
        foreach (IpAdapterCacheEntry entry in _entries.Values)
        {
            entry.Dispose();
        }
        _entries.Clear();
    }
}
