using System.Runtime.CompilerServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Logging;

namespace HartsyInference.Engine.Audio;

/// <summary>Device management for audio inference on ONE engine's backend: one generation at a time on that engine, other models evicted when a switch would not fit, and activations plus the memory pool trimmed after every run. One instance per <see cref="InferenceEngine"/> — the caches and the generation lock used to be process-wide statics, which serialized audio across every GPU and handed engine B a runner whose weights lived on engine A's device.</summary>
internal sealed class AudioRuntime
{
    /// <summary>Serializes inference across this engine's requests — one device per engine.</summary>
    private readonly SemaphoreSlim _genLock = new(1, 1);

    private readonly IAudioRunnerCache[] _caches;

    /// <summary>Model that ran most recently — switches trigger the memory-pressure eviction check.</summary>
    private string? _lastKey;

    /// <summary>Open runner leases; <see cref="UnloadAll"/> revokes them before it drops their runners.</summary>
    private readonly List<AudioRunnerLease> _leases = [];
    private readonly object _leaseLock = new();

    /// <summary>Bumped by every <see cref="UnloadAll"/>, so a lease whose open straddled a release is refused instead of
    /// holding a dropped runner or a disposed backend.</summary>
    private int _releaseEpoch;

    /// <summary>Free-host-RAM floor (KiB) below which switching models evicts every OTHER resident audio pipeline first. Runners otherwise accumulate (each holds multi-GB weight copies) until the kernel OOM-kills the process — observed at 21.8 GB RSS on a 32 GB box, and again at 11.5 GB with a desktop session sharing the machine, so the floor is generous. Override via vram.audioEvictBelowGb. Host RAM is process-wide, so with several engines each judges the floor independently — acceptable: the floor is generous and the VRAM floor below is genuinely per-device.</summary>
    private static long EvictBelowAvailableKb =>
        EngineKnobs.AudioEvictBelowGb.Value * 1024 * 1024;

    /// <summary>Free-VRAM floor: a prior model's promoted weights can hold most of the card, so the next model would OOM at load even with plenty of host RAM free.</summary>
    private const long EvictBelowFreeVramBytes = 3L * 1024 * 1024 * 1024;

    /// <summary>Resident speech pipelines, keyed by resolved repo.</summary>
    internal AudioRunnerCache<ITtsRunner> Tts { get; }

    /// <summary>Resident transcription pipelines, keyed by resolved repo.</summary>
    internal AudioRunnerCache<ISttRunner> Stt { get; }

    /// <summary>Resident music pipelines, keyed by resolved model key.</summary>
    internal AudioRunnerCache<IMusicRunner> Music { get; }

    /// <summary>Resident voice-conversion pipelines, keyed by resolved model key.</summary>
    internal AudioRunnerCache<IVcRunner> Vc { get; }

    /// <summary>Resident Demucs stem-separation pipelines, keyed by model name.</summary>
    internal AudioRunnerCache<DemucsRunner> Demucs { get; }

    /// <summary>Resident Resemble-Enhance pipelines.</summary>
    internal AudioRunnerCache<EnhanceRunner> Enhance { get; }

    internal AudioRuntime()
    {
        Tts = new AudioRunnerCache<ITtsRunner>("tts");
        Stt = new AudioRunnerCache<ISttRunner>("stt");
        Music = new AudioRunnerCache<IMusicRunner>("music");
        Vc = new AudioRunnerCache<IVcRunner>("vc");
        Demucs = new AudioRunnerCache<DemucsRunner>("fx:demucs");
        Enhance = new AudioRunnerCache<EnhanceRunner>("fx:enhance");
        _caches = [Tts, Stt, Music, Vc, Demucs, Enhance];
    }

    /// <summary>Drops every resident audio pipeline of THIS engine, pinned ones included, and revokes every open lease. Called when the engine releases its backend or frees memory, since a pipeline that bound to the old device (ACE-Step) must not survive into the next one — a pin protects a runner from memory pressure, not from losing its device. Waits up to <paramref name="waitSeconds"/> in total for an in-flight generation and for lease calls in flight, so the release does not dispose a pipeline out from under either; on timeout it proceeds anyway — teardown must never hang.</summary>
    internal void UnloadAll(int waitSeconds = 0)
    {
        TimeSpan budget = TimeSpan.FromSeconds(Math.Max(0, waitSeconds));
        long deadline = Environment.TickCount64 + (long)budget.TotalMilliseconds;
        bool held = _genLock.Wait(budget);
        if (!held)
        {
            Logs.Warning($"[Audio] Unloading with a generation still in flight after {waitSeconds}s — proceeding anyway.");
        }
        try
        {
            RevokeLeases(deadline);
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

    private void UnloadAllCore()
    {
        foreach (IAudioRunnerCache cache in _caches)
        {
            cache.UnloadAllExcept(null, includePinned: true);
        }
        _lastKey = null;
    }

    /// <summary>Closes every open lease, each waiting what is left of the release budget for its call in flight.</summary>
    private void RevokeLeases(long deadline)
    {
        AudioRunnerLease[] open;
        lock (_leaseLock)
        {
            _releaseEpoch++;
            open = [.. _leases];
            _leases.Clear();
        }
        foreach (AudioRunnerLease lease in open)
        {
            TimeSpan wait = TimeSpan.FromMilliseconds(Math.Max(0, deadline - Environment.TickCount64));
            if (!lease.Revoke(wait))
            {
                Logs.Warning($"[Audio] Unloading '{lease.ModelKey}' with a lease call still in flight — proceeding anyway.");
            }
        }
    }

    /// <summary>Runs one audio job under this engine's generation lock: evicts other models first when memory is tight, then frees leftover activations and trims the pool afterwards so a finished generation leaves nothing behind. <paramref name="job"/> names the cache and bare key the work is about to load through <c>GetOrLoadAsync</c>, so the eviction sweep can keep exactly that runner.</summary>
    internal async Task<T> RunAsync<T>(IBackend backend, AudioJob job, Func<CancellationToken, Task<T>> work, CancellationToken cancel,
        IReadOnlyList<IBackend>? stageBackends = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(work);
        await _genLock.WaitAsync(cancel).ConfigureAwait(false);
        // Device gate INSIDE the engine's audio lock (gate is always innermost, process-wide lock order).
        // A layer-split job passes its stage backends so EVERY stage device is gated, not just the primary.
        IDisposable gate = stageBackends is { Count: > 0 }
            ? await DeviceGate.AcquireAllAsync([backend, .. stageBackends], cancel).ConfigureAwait(false)
            : await DeviceGate.AcquireAsync(backend, cancel).ConfigureAwait(false);
        try
        {
            EvictOthersUnderMemoryPressure(backend, job);
            return await work(cancel).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            Logs.Debug($"[Audio] '{job.ModelKey}' was cancelled.");
            throw;
        }
        catch (Exception ex)
        {
            Logs.Error($"[Audio] '{job.ModelKey}' failed: {ex.Message}", ex);
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
            gate.Dispose();
            _genLock.Release();
        }
    }

    /// <summary>Streaming counterpart to <see cref="RunAsync{T}"/>: holds this engine's generation lock and device gate for the ENTIRE consumption of the returned stream, not just until <paramref name="work"/> returns — the underlying generation loop keeps running on the device until every item has been yielded. Releases when the consumer finishes draining, breaks early, or cancels; the async-iterator's <c>finally</c> below runs exactly once either way. (C# forbids <c>yield return</c> inside a <c>try</c> with a <c>catch</c> clause, so unlike <see cref="RunAsync{T}"/> this does not itself log-and-rethrow — exceptions still propagate to the caller, they just aren't logged at this layer.)</summary>
    internal async IAsyncEnumerable<T> RunStreamAsync<T>(IBackend backend, AudioJob job,
        Func<CancellationToken, IAsyncEnumerable<T>> work, [EnumeratorCancellation] CancellationToken cancel,
        IReadOnlyList<IBackend>? stageBackends = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(work);
        await _genLock.WaitAsync(cancel).ConfigureAwait(false);
        IDisposable gate = stageBackends is { Count: > 0 }
            ? await DeviceGate.AcquireAllAsync([backend, .. stageBackends], cancel).ConfigureAwait(false)
            : await DeviceGate.AcquireAsync(backend, cancel).ConfigureAwait(false);
        try
        {
            EvictOthersUnderMemoryPressure(backend, job);
            await foreach (T item in work(cancel).WithCancellation(cancel).ConfigureAwait(false))
            {
                yield return item;
            }
        }
        finally
        {
            // Same post-generation device hygiene as RunAsync — see its comment for why this matters.
            try
            {
                backend.FreeActivations();
                backend.TrimMemoryPool();
            }
            catch (Exception ex)
            {
                Logs.Warning($"[Audio] Post-generation device cleanup failed: {ex.Message}");
            }
            gate.Dispose();
            _genLock.Release();
        }
    }

    /// <summary>Opens a lease on the runner for <paramref name="key"/>: loads or reuses it through <see cref="RunAsync{T}"/>, the same locked, gated and eviction-checked path a service call takes, then builds the lease, which pins the key, and registers it for revocation. Pinning inside the generation lock means no sweep can run between the load and the pin.</summary>
    internal Task<TLease> OpenLeaseAsync<TRunner, TLease>(IBackend backend, AudioRunnerCache<TRunner> cache, string key,
        Func<CancellationToken, Task<TRunner>> load, Func<TRunner, TLease> create, CancellationToken cancel,
        IReadOnlyList<IBackend>? stageBackends = null)
        where TRunner : class, IDisposable
        where TLease : AudioRunnerLease
    {
        ArgumentNullException.ThrowIfNull(cache);
        ArgumentNullException.ThrowIfNull(load);
        ArgumentNullException.ThrowIfNull(create);
        int epoch = Volatile.Read(ref _releaseEpoch);
        return RunAsync(backend, new AudioJob(cache, key), async ct =>
        {
            ct.ThrowIfCancellationRequested();
            TRunner runner = await cache.GetOrLoadAsync(key, load, ct).ConfigureAwait(false);
            TLease lease = create(runner);
            Register(lease, epoch);
            return lease;
        }, cancel, stageBackends);
    }

    /// <summary>Forgets a disposed lease; a no-op for one a release already revoked.</summary>
    internal void Unregister(AudioRunnerLease lease)
    {
        lock (_leaseLock)
        {
            _leases.Remove(lease);
        }
    }

    /// <summary>Tracks <paramref name="lease"/> for revocation, or disposes it and throws when a release ran after its open began.</summary>
    private void Register(AudioRunnerLease lease, int epoch)
    {
        lock (_leaseLock)
        {
            if (_releaseEpoch == epoch)
            {
                _leases.Add(lease);
                return;
            }
        }
        lease.Dispose();
        throw new ObjectDisposedException(lease.GetType().Name,
            $"The engine released its audio models while the lease on '{lease.ModelKey}' was opening.");
    }

    /// <summary>When switching to a different model with low free host RAM or VRAM, drops every other resident pipeline (runner disposal also releases their auto-promoted GPU weights), keeping the incoming runner by asking ITS cache for the bare key and every pinned runner. Same-model repeat requests never evict, so warm generation stays warm. The keep decision has to go through <see cref="AudioJob.Cache"/>: the caches store bare keys, and handing them the prefixed <see cref="AudioJob.ModelKey"/> used to match nothing, so under pressure every switch evicted the model about to run and reloaded it.</summary>
    private void EvictOthersUnderMemoryPressure(IBackend backend, AudioJob job)
    {
        string modelKey = job.ModelKey;
        if (string.Equals(_lastKey, modelKey, StringComparison.Ordinal))
        {
            return;
        }
        long availableKb = ReadAvailableMemoryKb();
        (long freeVramBytes, long totalVramBytes) = SafeVramInfo(backend);
        bool hostLow = availableKb > 0 && availableKb < EvictBelowAvailableKb;
        // Gated on the TOTAL, not on free being non-zero: a backend that reports honestly can say zero free, and
        // testing free itself skipped eviction in precisely the case it was written for.
        bool vramLow = totalVramBytes > 0 && freeVramBytes < EvictBelowFreeVramBytes;
        if (hostLow || vramLow)
        {
            Logs.Info($"[Audio] Memory pressure (host {availableKb / 1024 / 1024.0:0.0} GB free, VRAM "
                + $"{freeVramBytes / 1024.0 / 1024 / 1024:0.0} GB free) — unloading other resident audio models before '{modelKey}'.");
            foreach (IAudioRunnerCache cache in _caches)
            {
                cache.UnloadAllExcept(ReferenceEquals(cache, job.Cache) ? job.Key : null);
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

    /// <summary>Free and total device memory, or <c>(0, 0)</c> when this backend does not report it.</summary>
    /// <remarks>Both halves, because free alone cannot say whether zero means "nothing left" or "no report" — and
    /// on a backend that reports honestly, nothing left is exactly the state eviction exists for. A total above
    /// zero is what marks the number as real.</remarks>
    private static (long FreeBytes, long TotalBytes) SafeVramInfo(IBackend backend)
    {
        try
        {
            return backend.GetVramInfo();
        }
        catch (Exception ex)
        {
            Logs.Debug($"[Audio] Backend VRAM query failed ({ex.Message}) — VRAM eviction disabled.");
            return (0, 0);
        }
    }
}
