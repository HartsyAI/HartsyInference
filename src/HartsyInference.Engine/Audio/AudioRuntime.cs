using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;

namespace HartsyInference.Engine.Audio;

/// <summary>Device management for audio inference on the Engine's shared backend: one generation at a time, other
/// models evicted when a switch would not fit, and activations plus the memory pool trimmed after every run.
/// Lifted from the extension's in-process audio engine, minus its private backend (the Engine owns one).</summary>
internal static class AudioRuntime
{
    /// <summary>Serializes inference across requests — one shared device per process.</summary>
    private static readonly SemaphoreSlim _genLock = new(1, 1);

    private static readonly List<IAudioRunnerCache> _caches = [];
    private static readonly object _cacheLock = new();

    /// <summary>Model that ran most recently — switches trigger the memory-pressure eviction check.</summary>
    private static string? _lastKey;

    /// <summary>Free-host-RAM floor (KiB) below which switching models evicts every OTHER resident audio pipeline
    /// first. Runners otherwise accumulate (each holds multi-GB weight copies) until the kernel OOM-kills the
    /// process — observed at 21.8 GB RSS on a 32 GB box, and again at 11.5 GB with a desktop session sharing the
    /// machine, so the floor is generous. Override via HARTSY_AUDIO_EVICT_BELOW_GB.</summary>
    private static readonly long EvictBelowAvailableKb =
        (long.TryParse(Environment.GetEnvironmentVariable("HARTSY_AUDIO_EVICT_BELOW_GB"), out long gb) && gb > 0 ? gb : 14L) * 1024 * 1024;

    /// <summary>Free-VRAM floor: a prior model's promoted weights can hold most of the card, so the next model
    /// would OOM at load even with plenty of host RAM free.</summary>
    private const long EvictBelowFreeVramBytes = 3L * 1024 * 1024 * 1024;

    /// <summary>Registers a category cache so eviction sweeps can reach its resident runners.</summary>
    internal static void Register(IAudioRunnerCache cache)
    {
        lock (_cacheLock)
        {
            _caches.Add(cache);
        }
    }

    /// <summary>Drops every resident audio pipeline. Called when the engine releases its backend or frees memory, since
    /// a pipeline that bound to the old device (ACE-Step) must not survive into the next one. Waits up to
    /// <paramref name="waitSeconds"/> for an in-flight generation so the release does not dispose a pipeline out from
    /// under it; on timeout it proceeds anyway — teardown must never hang.</summary>
    internal static void UnloadAll(int waitSeconds = 0)
    {
        bool held = _genLock.Wait(TimeSpan.FromSeconds(Math.Max(0, waitSeconds)));
        if (!held)
        {
            Logs.Warning($"[Audio] Unloading with a generation still in flight after {waitSeconds}s — proceeding anyway.");
        }
        try
        {
            UnloadAllCore();
        }
        finally
        {
            if (held)
            {
                _genLock.Release();
            }
        }
    }

    private static void UnloadAllCore()
    {
        lock (_cacheLock)
        {
            foreach (IAudioRunnerCache cache in _caches)
            {
                cache.UnloadAllExcept(null);
            }
        }
        _lastKey = null;
    }

    /// <summary>Runs one audio job under the shared generation lock: evicts other models first when memory is tight,
    /// then frees leftover activations and trims the pool afterwards so a finished generation leaves nothing behind.</summary>
    internal static async Task<T> RunAsync<T>(IBackend backend, string modelKey, Func<CancellationToken, Task<T>> work, CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(work);
        await _genLock.WaitAsync(cancel).ConfigureAwait(false);
        try
        {
            EvictOthersUnderMemoryPressure(backend, modelKey);
            return await work(cancel).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            Logs.Debug($"[Audio] '{modelKey}' was cancelled.");
            throw;
        }
        catch (Exception ex)
        {
            Logs.Error($"[Audio] '{modelKey}' failed: {ex.Message}", ex);
            throw;
        }
        finally
        {
            // Post-generation device hygiene: drop leftover GPU activations and return pool-reserved-but-free blocks
            // to the driver. Cached (promoted) weights stay resident, so warm latency is unaffected — without this,
            // finished generations left multi-GB of dead activations + pool reservations on the card.
            try
            {
                backend.FreeActivations();
                backend.TrimMemoryPool();
            }
            catch (Exception ex)
            {
                Logs.Warning($"[Audio] Post-generation device cleanup failed: {ex.Message}");
            }
            _genLock.Release();
        }
    }

    /// <summary>When switching to a different model with low free host RAM or VRAM, drops every other resident
    /// pipeline (runner disposal also releases their auto-promoted GPU weights). Same-model repeat requests never
    /// evict, so warm generation stays warm.</summary>
    private static void EvictOthersUnderMemoryPressure(IBackend backend, string modelKey)
    {
        if (string.Equals(_lastKey, modelKey, StringComparison.Ordinal))
        {
            return;
        }
        long availableKb = ReadAvailableMemoryKb();
        long freeVramBytes = SafeFreeMemoryBytes(backend);
        bool hostLow = availableKb > 0 && availableKb < EvictBelowAvailableKb;
        bool vramLow = freeVramBytes > 0 && freeVramBytes < EvictBelowFreeVramBytes;
        if (hostLow || vramLow)
        {
            Logs.Info($"[Audio] Memory pressure (host {availableKb / 1024 / 1024.0:0.0} GB free, VRAM "
                + $"{freeVramBytes / 1024.0 / 1024 / 1024:0.0} GB free) — unloading other resident audio models before '{modelKey}'.");
            lock (_cacheLock)
            {
                foreach (IAudioRunnerCache cache in _caches)
                {
                    cache.UnloadAllExcept(modelKey);
                }
            }
            // Disposal only drops host references; the finalizer queue that frees the promoted GPU copies is drained
            // lazily on the compute thread, so without forcing it here the card stays full across a model switch and
            // the incoming model OOMs on load. This is the one place the collect is load-bearing.
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            try
            {
                backend.FreeAllDeviceMemory();
            }
            catch (Exception ex)
            {
                Logs.Warning($"[Audio] Releasing device memory on eviction failed: {ex.Message}");
            }
        }
        _lastKey = modelKey;
    }

    /// <summary>MemAvailable from <c>/proc/meminfo</c> in KiB, or 0 when unavailable (non-Linux → no host eviction).</summary>
    private static long ReadAvailableMemoryKb()
    {
        try
        {
            foreach (string line in File.ReadLines("/proc/meminfo"))
            {
                if (line.StartsWith("MemAvailable:", StringComparison.Ordinal))
                {
                    string[] parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                    return parts.Length > 1 && long.TryParse(parts[1], out long kb) ? kb : 0;
                }
            }
        }
        catch (Exception ex)
        {
            Logs.Debug($"[Audio] Could not read /proc/meminfo ({ex.Message}) — host-memory eviction disabled.");
        }
        return 0;
    }

    private static long SafeFreeMemoryBytes(IBackend backend)
    {
        try
        {
            return backend.FreeMemoryBytes();
        }
        catch (Exception ex)
        {
            Logs.Debug($"[Audio] Backend free-memory query failed ({ex.Message}) — VRAM eviction disabled.");
            return 0;
        }
    }
}
