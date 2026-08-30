using System.Collections.Concurrent;
using System.Security.Cryptography;

namespace HartsyInference.Engine.Planning;

/// <summary>Caches full-file SHA-256 by canonical path, length, and last-write time without retaining file handles.</summary>
internal static class VideoCheckpointHashCache
{
    private static readonly ConcurrentDictionary<HashCacheKey, Lazy<Task<string>>> _hashes = new();

    /// <summary>Returns a lowercase SHA-256, sharing one read among concurrent planners for the same immutable file state.</summary>
    public static Task<string> GetSha256Async(string path, CancellationToken cancel)
    {
        string canonical = CanonicalPath(path);
        FileInfo info = new FileInfo(canonical);
        if (!info.Exists)
        {
            throw new FileNotFoundException("Checkpoint artifact was not found.", canonical);
        }
        HashCacheKey key = new HashCacheKey(canonical, info.Length, info.LastWriteTimeUtc.Ticks);
        Lazy<Task<string>> lazy = _hashes.GetOrAdd(key,
            static cacheKey => new Lazy<Task<string>>(() => ComputeAsync(cacheKey), LazyThreadSafetyMode.ExecutionAndPublication));
        return AwaitWithCancellationAsync(lazy.Value, cancel);
    }

    /// <summary>Number of immutable file states retained, exposed for deterministic cache tests.</summary>
    internal static int Count => _hashes.Count;

    /// <summary>Clears only hash metadata; no file or model state is changed.</summary>
    internal static void Clear() => _hashes.Clear();

    private static async Task<string> ComputeAsync(HashCacheKey key)
    {
        await using FileStream stream = new FileStream(key.CanonicalPath, FileMode.Open, FileAccess.Read, FileShare.Read,
            1024 * 1024, FileOptions.Asynchronous | FileOptions.SequentialScan);
        byte[] hash = await SHA256.HashDataAsync(stream).ConfigureAwait(false);
        return Convert.ToHexString(hash).ToLowerInvariant();
    }

    private static async Task<string> AwaitWithCancellationAsync(Task<string> hashTask, CancellationToken cancel) =>
        await hashTask.WaitAsync(cancel).ConfigureAwait(false);

    /// <summary>Resolves a path through its final symbolic-link target for hash and execution bindings.</summary>
    internal static string CanonicalPath(string path)
    {
        string full = Path.GetFullPath(path);
        FileSystemInfo? target = File.ResolveLinkTarget(full, returnFinalTarget: true);
        return target?.FullName ?? full;
    }

    private readonly record struct HashCacheKey(string CanonicalPath, long Length, long LastWriteTicks);
}
