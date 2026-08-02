using System.Collections.Concurrent;
using HartsyInference.Core.Backends;

namespace HartsyInference.Engine;

/// <summary>Process-wide per-GPU generation gate. Two backends on one device are state-isolated (per-backend CUDA
/// caches/streams) but their concurrent kernel/allocator use is not yet audited, so generations on the SAME device
/// ordinal serialize here. No cost in the common cases: a device with one backend never has two concurrent
/// generations trying the gate (each backend already serializes its own), and different devices use different
/// gates. <c>HARTSY_SAME_GPU_CONCURRENT=1</c> disables the gate once the concurrency milestone lands/soaks.</summary>
public static class DeviceGate
{
    private static readonly bool _concurrent = Environment.GetEnvironmentVariable("HARTSY_SAME_GPU_CONCURRENT") == "1";
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
        SemaphoreSlim gate = _gates.GetOrAdd(backend.Device.Ordinal, static _ => new SemaphoreSlim(1, 1));
        gate.Wait(cancel);
        return new Releaser(gate);
    }

    private sealed class Releaser : IDisposable
    {
        private SemaphoreSlim? _gate;

        public Releaser(SemaphoreSlim gate) => _gate = gate;

        public void Dispose() => Interlocked.Exchange(ref _gate, null)?.Release();
    }

    private sealed class NoopReleaser : IDisposable
    {
        public void Dispose()
        {
        }
    }
}
