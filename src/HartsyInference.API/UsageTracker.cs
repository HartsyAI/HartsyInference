using System.Collections.Concurrent;

namespace HartsyInference.API;

/// <summary>Read-only point-in-time view of one caller identity's usage, returned by <see cref="UsageTracker.Snapshot"/>.</summary>
public sealed record UsageSnapshot(long TotalRequests, long ErrorCount, DateTimeOffset LastSeenUtc, IReadOnlyDictionary<string, long> ByModality);

/// <summary>In-memory per-caller request counters, keyed by the resolved <see cref="ApiKeyIdentity.Name"/> (or
/// <c>"anonymous"</c> when auth is disabled). Recorded from one place — the usage-tracking middleware in
/// <c>HartsyInferenceServiceExtensions.MapHartsyInferenceEndpoints</c> — and read back via <c>GET /admin/usage</c>.
/// Resets on restart, matching this project's in-memory-only state design; not intended as a billing ledger.</summary>
public sealed class UsageTracker
{
    private readonly ConcurrentDictionary<string, KeyUsage> _byIdentity = new(StringComparer.Ordinal);

    public void Record(string identityName, string modality, bool isError)
    {
        KeyUsage usage = _byIdentity.GetOrAdd(identityName, static _ => new KeyUsage());
        usage.Record(modality, isError);
    }

    public IReadOnlyDictionary<string, UsageSnapshot> Snapshot() =>
        _byIdentity.ToDictionary(kv => kv.Key, kv => kv.Value.ToSnapshot(), StringComparer.Ordinal);

    private sealed class KeyUsage
    {
        private long _totalRequests;
        private long _errorCount;
        private long _lastSeenUnixMs;
        private readonly ConcurrentDictionary<string, long> _byModality = new(StringComparer.Ordinal);

        public void Record(string modality, bool isError)
        {
            Interlocked.Increment(ref _totalRequests);
            if (isError)
                Interlocked.Increment(ref _errorCount);
            _byModality.AddOrUpdate(modality, 1, static (_, count) => count + 1);
            Interlocked.Exchange(ref _lastSeenUnixMs, DateTimeOffset.UtcNow.ToUnixTimeMilliseconds());
        }

        public UsageSnapshot ToSnapshot() => new(
            Interlocked.Read(ref _totalRequests),
            Interlocked.Read(ref _errorCount),
            DateTimeOffset.FromUnixTimeMilliseconds(Interlocked.Read(ref _lastSeenUnixMs)),
            new Dictionary<string, long>(_byModality, StringComparer.Ordinal));
    }
}
