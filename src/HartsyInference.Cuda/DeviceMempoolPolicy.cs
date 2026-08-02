using HartsyInference.Core.Logging;

namespace HartsyInference.Cuda;

/// <summary>Set-once-per-device policy for the default mempool's release threshold, refcounted by live backends.
/// The threshold is DEVICE state (one default pool per ordinal, whatever contexts or backends exist), so letting
/// each backend write it directly meant backend B's construction momentarily flipped backend A's live pool to
/// release-everything — re-introducing the alloc/free thrash the keep-threshold exists to stop. The first backend
/// on a device decides; later backends refcount and must agree (a mismatch warns and keeps the first policy).</summary>
internal static class DeviceMempoolPolicy
{
    private sealed class Entry
    {
        public int RefCount;
        public bool Keep;
    }

    private static readonly Dictionary<int, Entry> _devices = new();
    private static readonly object _lock = new();

    /// <summary>Applies (first caller) or joins (later callers) the device's threshold policy.
    /// <paramref name="keep"/> true = hold freed blocks in the pool (fast reuse); false = release eagerly.</summary>
    internal static void Acquire(int deviceOrdinal, bool keep)
    {
        lock (_lock)
        {
            if (_devices.TryGetValue(deviceOrdinal, out Entry? entry))
            {
                entry.RefCount++;
                if (entry.Keep != keep)
                {
                    Logs.Warning($"[Cuda] Backend requested mempool keep={keep} on device {deviceOrdinal}, but a live "
                        + $"backend already set keep={entry.Keep} — keeping the first policy (it owns the live pool).");
                }
                return;
            }
            _devices[deviceOrdinal] = new Entry { RefCount = 1, Keep = keep };
            Apply(deviceOrdinal, keep);
        }
    }

    /// <summary>Drops one backend's hold; the device entry disappears at zero so the NEXT first backend re-decides.
    /// The live threshold is deliberately left as-is — there is nobody left to care, and touching it could stall a
    /// racing construction.</summary>
    internal static void Release(int deviceOrdinal)
    {
        lock (_lock)
        {
            if (_devices.TryGetValue(deviceOrdinal, out Entry? entry) && --entry.RefCount <= 0)
            {
                _devices.Remove(deviceOrdinal);
            }
        }
    }

    private static void Apply(int deviceOrdinal, bool keep)
    {
        try
        {
            CudaDriverApi.cuDeviceGetDefaultMemPool(out nint pool, deviceOrdinal).ThrowOnError();
            unsafe
            {
                ulong threshold = keep ? ulong.MaxValue : 0;
                CudaDriverApi.cuMemPoolSetAttribute(pool, CudaDriverApi.CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &threshold).ThrowOnError();
            }
        }
        catch (Exception ex)
        {
            Logs.Warning($"[Cuda] could not set mempool release threshold on device {deviceOrdinal}: {ex.Message}");
        }
    }
}
