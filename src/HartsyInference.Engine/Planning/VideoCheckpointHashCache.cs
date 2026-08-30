using System.Buffers;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Security;
using System.Security.Cryptography;
using System.Text;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Logging;
using HartsyInference.Core.MemoryManagement;

namespace HartsyInference.Engine.Planning;

/// <summary>Caches full-file SHA-256 by canonical path, length, and last-write time without retaining file handles.</summary>
internal static class VideoCheckpointHashCache
{
    private const int HashBufferBytes = 4 * 1024 * 1024;
    private const int PersistentFormatVersion = 1;
    private const long ProgressLogThresholdBytes = 1024L * 1024 * 1024;

    private static readonly ConcurrentDictionary<HashCacheKey, Lazy<Task<string>>> _hashes =
        new(new HashCacheKeyComparer());
    private static readonly ConcurrentDictionary<string, int> _computeCounts = new(VideoArtifactPath.Comparer);

    /// <summary>Returns a lowercase SHA-256, sharing one read among concurrent planners for the same immutable file state.</summary>
    public static Task<string> GetSha256Async(string path, CancellationToken cancel)
    {
        string canonical = VideoArtifactPath.Canonicalize(path);
        FileInfo info = new FileInfo(canonical);
        if (!info.Exists)
        {
            throw new FileNotFoundException("Checkpoint artifact was not found.", canonical);
        }
        HashCacheKey key = new HashCacheKey(canonical, info.Length, info.LastWriteTimeUtc.Ticks);
        PruneOtherStates(key);
        Lazy<Task<string>> lazy = _hashes.GetOrAdd(key,
            static cacheKey => new Lazy<Task<string>>(() => GetOrComputeAsync(cacheKey),
                LazyThreadSafetyMode.ExecutionAndPublication));
        return AwaitWithCancellationAsync(lazy.Value, cancel);
    }

    /// <summary>Seeds an exact hash already verified while a downloader streamed the same immutable file state.</summary>
    internal static Task RecordVerifiedSha256Async(string path, string sha256)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentException.ThrowIfNullOrWhiteSpace(sha256);
        string normalized = sha256.ToLowerInvariant();
        if (!IsSha256(normalized))
        {
            throw new ArgumentException("A verified checkpoint hash must contain exactly 64 hexadecimal characters.",
                nameof(sha256));
        }

        string canonical = VideoArtifactPath.Canonicalize(path);
        FileInfo info = new FileInfo(canonical);
        if (!info.Exists)
        {
            throw new FileNotFoundException("Verified checkpoint artifact was not found.", canonical);
        }
        HashCacheKey key = new HashCacheKey(canonical, info.Length, info.LastWriteTimeUtc.Ticks);
        PruneOtherStates(key);
        _hashes[key] = Completed(normalized);
        return TryPersistAsync(key, normalized);
    }

    /// <summary>Number of immutable file states retained, exposed for deterministic cache tests.</summary>
    internal static int Count => _hashes.Count;

    /// <summary>Clears only hash metadata; no file or model state is changed.</summary>
    internal static void Clear() => _hashes.Clear();

    /// <summary>Number of full-file reads performed for one canonical path, exposed for restart-cache tests.</summary>
    internal static int ComputeCountFor(string path)
    {
        string canonical = VideoArtifactPath.Canonicalize(path);
        return _computeCounts.TryGetValue(canonical, out int count) ? count : 0;
    }

    /// <summary>Number of retained stamps for one canonical file, exposed for stale-state eviction tests.</summary>
    internal static int StateCountFor(string path)
    {
        string canonical = VideoArtifactPath.Canonicalize(path);
        return _hashes.Keys.Count(key => VideoArtifactPath.Comparer.Equals(key.CanonicalPath, canonical));
    }

    /// <summary>Deletes one durable cache record without changing the checkpoint.</summary>
    internal static void RemovePersistent(string path)
    {
        string canonical = VideoArtifactPath.Canonicalize(path);
        try
        {
            string cachePath = PersistentPath(canonical);
            File.Delete(cachePath);
        }
        catch (Exception ex) when (IsPersistenceFailure(ex))
        {
            Logs.Warning($"[VideoPlan] Could not remove checkpoint hash metadata for '{Path.GetFileName(canonical)}': "
                + ex.Message);
        }
    }

    private static async Task<string> GetOrComputeAsync(HashCacheKey key)
    {
        if (TryReadPersistent(key, out string? persisted))
        {
            RequireMatchingStamp(key);
            return persisted!;
        }

        string computed = await ComputeAsync(key).ConfigureAwait(false);
        RequireMatchingStamp(key);
        await TryPersistAsync(key, computed).ConfigureAwait(false);
        return computed;
    }

    private static async Task<string> ComputeAsync(HashCacheKey key)
    {
        _computeCounts.AddOrUpdate(key.CanonicalPath, 1, static (_, count) => count + 1);
        string fileName = Path.GetFileName(key.CanonicalPath);
        Logs.Info($"[VideoPlan] Computing exact SHA-256 for '{fileName}' ({ByteFormat.MbF1(key.Length)}). "
            + "This one-time pass is cached across process restarts.");
        Stopwatch elapsed = Stopwatch.StartNew();
        await using FileStream stream = new FileStream(key.CanonicalPath, FileMode.Open, FileAccess.Read, FileShare.Read,
            HashBufferBytes, FileOptions.Asynchronous | FileOptions.SequentialScan);
        using IncrementalHash hasher = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        byte[] buffer = ArrayPool<byte>.Shared.Rent(HashBufferBytes);
        long readTotal = 0;
        int nextMilestone = 25;
        try
        {
            while (true)
            {
                int read = await stream.ReadAsync(buffer.AsMemory(0, HashBufferBytes)).ConfigureAwait(false);
                if (read == 0)
                {
                    break;
                }
                hasher.AppendData(buffer, 0, read);
                readTotal += read;
                if (key.Length >= ProgressLogThresholdBytes && nextMilestone < 100
                    && readTotal * 100L >= key.Length * nextMilestone)
                {
                    Logs.Info($"[VideoPlan] SHA-256 for '{fileName}': {nextMilestone}%.");
                    nextMilestone += 25;
                }
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }
        string hash = Convert.ToHexString(hasher.GetHashAndReset()).ToLowerInvariant();
        Logs.Info($"[VideoPlan] Exact SHA-256 for '{fileName}' finished in {elapsed.Elapsed.TotalSeconds:F1}s.");
        return hash;
    }

    private static void PruneOtherStates(HashCacheKey current)
    {
        foreach (HashCacheKey candidate in _hashes.Keys)
        {
            if (VideoArtifactPath.Comparer.Equals(candidate.CanonicalPath, current.CanonicalPath)
                && (candidate.Length != current.Length || candidate.LastWriteTicks != current.LastWriteTicks))
            {
                _hashes.TryRemove(candidate, out _);
            }
        }
    }

    private static void RequireMatchingStamp(HashCacheKey expected)
    {
        FileInfo current = new FileInfo(expected.CanonicalPath);
        if (current.Exists && current.Length == expected.Length
            && current.LastWriteTimeUtc.Ticks == expected.LastWriteTicks)
        {
            return;
        }

        // A hash read can take minutes for the BF16 checkpoint. Never publish bytes read across a concurrent
        // replacement under the stamp captured before that read; the caller must plan the new file state afresh.
        _hashes.TryRemove(expected, out _);
        throw new IOException($"Checkpoint '{Path.GetFileName(expected.CanonicalPath)}' changed while its exact "
            + "SHA-256 was being resolved; retry planning after the file is stable.");
    }

    private static async Task<string> AwaitWithCancellationAsync(Task<string> hashTask, CancellationToken cancel) =>
        await hashTask.WaitAsync(cancel).ConfigureAwait(false);

    private static bool TryReadPersistent(HashCacheKey key, out string? sha256)
    {
        sha256 = null;
        try
        {
            string path = PersistentPath(key.CanonicalPath);
            if (!File.Exists(path))
            {
                return false;
            }
            string[] lines = File.ReadAllLines(path);
            if (lines.Length != 4 || !int.TryParse(lines[0], out int version)
                || !long.TryParse(lines[1], out long length) || !long.TryParse(lines[2], out long ticks)
                || version != PersistentFormatVersion || length != key.Length || ticks != key.LastWriteTicks
                || !IsSha256(lines[3]))
            {
                return false;
            }
            sha256 = lines[3].ToLowerInvariant();
            return true;
        }
        catch (Exception ex) when (IsPersistenceFailure(ex))
        {
            Logs.Warning($"[VideoPlan] Could not read checkpoint hash metadata for "
                + $"'{Path.GetFileName(key.CanonicalPath)}'; the exact hash will be recomputed: {ex.Message}");
            return false;
        }
    }

    private static async Task TryPersistAsync(HashCacheKey key, string sha256)
    {
        string? temporary = null;
        try
        {
            string path = PersistentPath(key.CanonicalPath);
            string? directory = Path.GetDirectoryName(path);
            temporary = path + $".{Guid.NewGuid():N}.tmp";
            Directory.CreateDirectory(directory!);
            string contents = $"{PersistentFormatVersion}\n{key.Length}\n{key.LastWriteTicks}\n{sha256}\n";
            await File.WriteAllTextAsync(temporary, contents).ConfigureAwait(false);
            File.Move(temporary, path, overwrite: true);
        }
        catch (Exception ex) when (IsPersistenceFailure(ex))
        {
            Logs.Warning($"[VideoPlan] Could not persist checkpoint hash metadata for "
                + $"'{Path.GetFileName(key.CanonicalPath)}'; this process will still reuse it: {ex.Message}");
            if (temporary is not null)
            {
                TryDeleteTemporary(temporary);
            }
        }
    }

    private static void TryDeleteTemporary(string path)
    {
        try
        {
            File.Delete(path);
        }
        catch (Exception ex) when (IsPersistenceFailure(ex))
        {
            Logs.Debug($"[VideoPlan] Could not remove temporary hash metadata '{Path.GetFileName(path)}': "
                + ex.Message);
        }
    }

    private static string PersistentPath(string canonicalPath)
    {
        string identity = VideoArtifactPath.Identity(canonicalPath);
        byte[] digest = SHA256.HashData(Encoding.UTF8.GetBytes(identity));
        string fileName = Convert.ToHexString(digest).ToLowerInvariant() + ".sha256";
        return Path.Combine(PersistentRoot(), "video-checkpoint-hashes", fileName);
    }

    private static string PersistentRoot()
    {
        string? configured = EngineKnobs.ModelCacheRoot.Value;
        if (!string.IsNullOrWhiteSpace(configured))
        {
            return Path.GetFullPath(configured);
        }
        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        return Path.Combine(home, ".cache", "hartsyinference");
    }

    private static Lazy<Task<string>> Completed(string sha256) => new(
        () => Task.FromResult(sha256), LazyThreadSafetyMode.ExecutionAndPublication);

    private static bool IsSha256(string value) => value.Length == 64
        && value.All(character => character is >= '0' and <= '9' or >= 'a' and <= 'f' or >= 'A' and <= 'F');

    private static bool IsPersistenceFailure(Exception exception) => exception is IOException
        or UnauthorizedAccessException or NotSupportedException or ArgumentException or SecurityException;

    private readonly record struct HashCacheKey(string CanonicalPath, long Length, long LastWriteTicks);

    private sealed class HashCacheKeyComparer : IEqualityComparer<HashCacheKey>
    {
        public bool Equals(HashCacheKey x, HashCacheKey y) =>
            VideoArtifactPath.Comparer.Equals(x.CanonicalPath, y.CanonicalPath)
            && x.Length == y.Length && x.LastWriteTicks == y.LastWriteTicks;

        public int GetHashCode(HashCacheKey value) => HashCode.Combine(
            VideoArtifactPath.Comparer.GetHashCode(value.CanonicalPath), value.Length, value.LastWriteTicks);
    }
}
