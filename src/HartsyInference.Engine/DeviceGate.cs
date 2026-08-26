using System.Collections.Concurrent;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;

namespace HartsyInference.Engine;

/// <summary>Process-wide per-GPU generation gate. Two backends on one device are state-isolated (per-backend CUDA
/// caches/streams) but their concurrent kernel/allocator use is not yet audited, so generations on the SAME device
/// ordinal serialize here. No cost in the common cases: a device with one backend never has two concurrent
/// generations trying the gate (each backend already serializes its own), and different devices use different
/// gates. <c>HARTSY_SAME_GPU_CONCURRENT=1</c> disables the gate once the concurrency milestone lands/soaks.</summary>
public static class DeviceGate
{
    private static readonly bool _concurrent = EngineKnobs.SameGpuConcurrent.Value;
    private static readonly ConcurrentDictionary<int, SemaphoreSlim> _gates = new();
    private static readonly IDisposable _noop = new NoopReleaser();

    /// <summary>Acquires the generation slot for <paramref name="backend"/>'s device; dispose the returned token to
    /// release. No-op (returns immediately) for CPU backends or when same-GPU concurrency is enabled. Acquire this
    /// INNERMOST — after any engine/slot locks — so lock order stays consistent process-wide.</summary>
    public static async Task<IDisposable> AcquireAsync(IBackend backend, CancellationToken cancel = default)
    {
        if (_concurrent || backend is null || backend.Device.Type != DeviceType.Cuda)
        {
            return _noop;
        }
        SemaphoreSlim gate = _gates.GetOrAdd(backend.Device.Ordinal, static _ => new SemaphoreSlim(1, 1));
        await gate.WaitAsync(cancel).ConfigureAwait(false);
        return new Releaser(gate);
    }

    /// <summary>Synchronous form for non-async generation paths.</summary>
    public static IDisposable Acquire(IBackend backend, CancellationToken cancel = default)
    {
        if (_concurrent || backend is null || backend.Device.Type != DeviceType.Cuda)
        {
            return _noop;
        }
        return AcquireForCudaOrdinal(backend.Device.Ordinal, cancel);
    }

    /// <summary>Acquires a device's slot by ORDINAL, for callers that know the device before they have a backend
    /// object (a model load that has yet to construct one). Deadlock-freedom rule: take either exactly ONE gate,
    /// or several via the Acquire*All* forms — those always acquire in ascending-ordinal order, and a single gate
    /// is a degenerate case of that order, so the two styles compose safely.</summary>
    public static IDisposable AcquireForCudaOrdinal(int ordinal, CancellationToken cancel = default)
    {
        if (_concurrent || ordinal < 0)
        {
            return _noop;
        }
        SemaphoreSlim gate = _gates.GetOrAdd(ordinal, static _ => new SemaphoreSlim(1, 1));
        gate.Wait(cancel);
        return new Releaser(gate);
    }

    /// <summary>Acquires the slots for EVERY distinct CUDA ordinal among <paramref name="backends"/> (nulls and
    /// non-CUDA backends ignored), in ascending-ordinal order; dispose to release in reverse. Use when one
    /// generation's placement spans devices (CFG-parallel, TE/VAE, shard stages) — gating only the primary left
    /// the other devices open to concurrent generations from sibling engines gated on THOSE ordinals.</summary>
    public static IDisposable AcquireAll(IEnumerable<IBackend?> backends, CancellationToken cancel = default) =>
        _concurrent ? _noop : AcquireAllOrdinals(CudaOrdinalsOf(backends), cancel);

    /// <summary>Async form of <see cref="AcquireAll"/> for the audio path.</summary>
    public static async Task<IDisposable> AcquireAllAsync(IEnumerable<IBackend?> backends, CancellationToken cancel = default)
    {
        if (_concurrent)
        {
            return _noop;
        }
        List<int> ordinals = SortedDistinct(CudaOrdinalsOf(backends));
        if (ordinals.Count == 0)
        {
            return _noop;
        }
        List<SemaphoreSlim> acquired = new(ordinals.Count);
        try
        {
            foreach (int ordinal in ordinals)
            {
                SemaphoreSlim gate = _gates.GetOrAdd(ordinal, static _ => new SemaphoreSlim(1, 1));
                await gate.WaitAsync(cancel).ConfigureAwait(false);
                acquired.Add(gate);
            }
        }
        catch
        {
            for (int i = acquired.Count - 1; i >= 0; i--)
            {
                acquired[i].Release();
            }
            throw;
        }
        return new MultiReleaser(acquired);
    }

    /// <summary>Ordinal-list form of <see cref="AcquireAll"/> (negatives = CPU, ignored).</summary>
    public static IDisposable AcquireAllOrdinals(IEnumerable<int> ordinals, CancellationToken cancel = default)
    {
        if (_concurrent)
        {
            return _noop;
        }
        List<int> sorted = SortedDistinct(ordinals);
        if (sorted.Count == 0)
        {
            return _noop;
        }
        List<SemaphoreSlim> acquired = new(sorted.Count);
        try
        {
            foreach (int ordinal in sorted)
            {
                SemaphoreSlim gate = _gates.GetOrAdd(ordinal, static _ => new SemaphoreSlim(1, 1));
                gate.Wait(cancel);
                acquired.Add(gate);
            }
        }
        catch
        {
            for (int i = acquired.Count - 1; i >= 0; i--)
            {
                acquired[i].Release();
            }
            throw;
        }
        return new MultiReleaser(acquired);
    }

    private static IEnumerable<int> CudaOrdinalsOf(IEnumerable<IBackend?> backends)
    {
        foreach (IBackend? backend in backends)
        {
            if (backend is not null && backend.Device.Type == DeviceType.Cuda)
            {
                yield return backend.Device.Ordinal;
            }
        }
    }

    private static List<int> SortedDistinct(IEnumerable<int> ordinals)
    {
        List<int> sorted = [];
        foreach (int ordinal in ordinals)
        {
            if (ordinal >= 0 && !sorted.Contains(ordinal))
            {
                sorted.Add(ordinal);
            }
        }
        sorted.Sort();
        return sorted;
    }

    private sealed class Releaser : IDisposable
    {
        private SemaphoreSlim? _gate;

        public Releaser(SemaphoreSlim gate) => _gate = gate;

        public void Dispose() => Interlocked.Exchange(ref _gate, null)?.Release();
    }

    private sealed class MultiReleaser(List<SemaphoreSlim> gates) : IDisposable
    {
        private List<SemaphoreSlim>? _gates = gates;

        public void Dispose()
        {
            List<SemaphoreSlim>? gates = Interlocked.Exchange(ref _gates, null);
            if (gates is null)
            {
                return;
            }
            for (int i = gates.Count - 1; i >= 0; i--)
            {
                gates[i].Release();
            }
        }
    }

    private sealed class NoopReleaser : IDisposable
    {
        public void Dispose()
        {
        }
    }
}
