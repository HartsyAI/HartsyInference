using System.Diagnostics;
using System.Runtime.CompilerServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Gpu;

/// <summary>What every GPU backend does the same way, once.
///
/// <para>Two things live here. The op scope — the safe point at which a backend drains the cleanup a finalizer
/// could not run and reclaims what the previous op displaced — which both backends had written separately and
/// which is easy to get subtly wrong, since it is defined by what has ALREADY happened rather than by what the op
/// is about to do. And the <see cref="IBackend"/> members that are pure delegation to the residency cache, which
/// were duplicated for no reason beyond each backend owning its own cache object.</para>
///
/// <para>Deliberately NOT here: anything a backend does differently. The base declares hooks for those and states
/// what each is for, rather than absorbing one backend's answer and making the other bend to it.</para></summary>
public abstract class GpuBackendBase
{
    private int _opDepth;
    private int _dispatchesThisOp;
    private int _disposed;

    /// <summary>This backend's device-residency cache.</summary>
    protected abstract IGpuResidency Residency { get; }

    /// <summary>Whether to time each op. Read once per scope, so turning it off costs a bool.</summary>
    /// <remarks>A hook rather than a shared profile object: the two backends still keep their own — CUDA's is NVTX
    /// ranges, Vulkan's is <c>VulkanProfiler</c> — and unifying those is its own change. What the scope needs is
    /// only the question "should I time this", and somewhere to hand the answer.</remarks>
    protected virtual bool ProfilingEnabled => false;

    /// <summary>Receives one completed op's host time and dispatch count.</summary>
    protected virtual void OnOpRecorded(string opName, long elapsedTicks, int dispatches) { }

    /// <summary>Dispatches recorded in the op currently running. Zero outside an op.</summary>
    protected int DispatchesThisOp => _dispatchesThisOp;

    /// <summary>True while an op is in progress, at any nesting depth.</summary>
    protected bool InOp => _opDepth > 0;

    /// <summary>Counts one device dispatch against the running op.</summary>
    protected void CountDispatch() => _dispatchesThisOp++;

    /// <summary>Opens an op scope. Dispose it — <c>using OpScope _ = EnterOp();</c> — at the end of the op.</summary>
    /// <remarks>Nesting is expected and only the OUTERMOST scope does the work: an op built out of other ops must
    /// not drain or sweep in the middle of itself, because the buffers it would reclaim are the ones its own later
    /// dispatches still read.</remarks>
    protected OpScope EnterOp([CallerMemberName] string opName = "") => new(this, opName);

    /// <summary>Called as the outermost op begins, before the drain and sweep.</summary>
    /// <remarks>Where a backend makes itself current, binds its ambient state, or opens a profiling range.</remarks>
    protected virtual void OnOpBegin(string opName) { }

    /// <summary>Called as the outermost op ends.</summary>
    /// <remarks>Where a backend flushes a command stream, frees per-op transients, or closes a profiling range.</remarks>
    protected virtual void OnOpEnd(string opName, int dispatches) { }

    /// <summary>The op scope. A struct so an op that does nothing else allocates nothing.</summary>
    public readonly struct OpScope : IDisposable
    {
        private readonly GpuBackendBase _backend;
        private readonly string _opName;
        private readonly long _startTicks;

        internal OpScope(GpuBackendBase backend, string opName)
        {
            _backend = backend;
            _opName = opName;
            _startTicks = backend.ProfilingEnabled ? Stopwatch.GetTimestamp() : 0;
            if (backend._opDepth == 0)
            {
                backend._dispatchesThisOp = 0;
                backend.OnOpBegin(opName);
                // A tensor finalized rather than disposed cannot free its device buffer from the finalizer thread,
                // which has no business touching a device context, so the work was queued. This is the safe point
                // that runs it, on the owning thread.
                backend.Residency.DrainFinalizerCleanup();
                // Every previous op's finally has run by now, so anything a rebind displaced and nobody claimed is
                // provably ownerless. Reclaiming it here is what makes the non-in-place case safe.
                backend.Residency.SweepOrphans();
            }
            backend._opDepth++;
        }

        public void Dispose()
        {
            _backend._opDepth--;
            if (_backend._opDepth != 0)
            {
                return;
            }
            int dispatches = _backend._dispatchesThisOp;
            _backend.OnOpEnd(_opName, dispatches);
            if (_startTicks != 0)
            {
                _backend.OnOpRecorded(_opName, Stopwatch.GetTimestamp() - _startTicks, dispatches);
            }
        }
    }

    // ── IBackend members that are the residency cache, spelled once ──────────────────────────────────────

    /// <summary>Device-to-host syncs since the last reset. A fully resident step reports zero.</summary>
    /// <remarks>Inside an op scope like any other entry point, because a backend that resolves its state ambiently
    /// must bind it before touching the cache at all — reading a counter included.</remarks>
    public long GetD2hSyncCount()
    {
        using OpScope _ = EnterOp();
        return Residency.D2hSyncCount;
    }

    /// <summary>Resets the D2H counter, so a caller can measure one region rather than a whole run.</summary>
    public void ResetD2hSyncCount()
    {
        using OpScope _ = EnterOp();
        Residency.ResetD2hSyncCount();
    }

    /// <summary>Uploads weights ahead of first use, so no op pays a cache-miss transfer mid-generation.</summary>
    /// <remarks>Expands low-rank adjuncts first, and both backends did: a weight carrying an adjunct is read as
    /// its factors, so preloading the weight alone leaves the factors to miss one at a time on the hot path.
    /// Subclasses take the expanded set and decide what else uploading means for them.</remarks>
    public void PreloadWeights(IEnumerable<Tensor> weights)
    {
        using OpScope _ = EnterOp();
        PreloadExpandedWeights(LowRankAdjunct.ExpandWeights(weights));
    }

    /// <summary>Uploads an already-expanded weight set.</summary>
    protected virtual void PreloadExpandedWeights(IEnumerable<Tensor> weights) => Residency.PreloadWeights(weights);

    /// <summary>Releases the device copies of these weights, adjunct factors included.</summary>
    public void FreeWeights(IEnumerable<Tensor> weights)
    {
        using OpScope _ = EnterOp();
        FreeExpandedWeights([.. LowRankAdjunct.ExpandWeights(weights)]);
    }

    /// <summary>Releases an already-expanded weight set.</summary>
    protected virtual void FreeExpandedWeights(IReadOnlyList<Tensor> weights) => Residency.FreeWeights(weights);

    /// <summary>Marks an activation as surviving the bulk free — cross-step state whose only copy is on device.</summary>
    public void PinActivation(Tensor tensor)
    {
        using OpScope _ = EnterOp();
        Residency.PinActivation(tensor);
    }

    /// <summary>Removes a <see cref="PinActivation"/> mark.</summary>
    public void UnpinActivation(Tensor tensor)
    {
        using OpScope _ = EnterOp();
        Residency.UnpinActivation(tensor);
    }

    /// <summary>Materializes activations to host, largest first, until <paramref name="targetBytes"/> is freed.</summary>
    public virtual long OffloadActivations(long targetBytes)
    {
        using OpScope _ = EnterOp();
        return Residency.OffloadActivations(targetBytes);
    }

    /// <summary>Free and total device memory, as the driver reports it.</summary>
    public abstract (long FreeBytes, long TotalBytes) GetVramInfo();

    // ── Reclaiming device memory ─────────────────────────────────────────────────────────────────────────
    //
    // These four are what an engine calls at a phase, generation or model-swap boundary, and they are the
    // difference between VRAM coming back when asked and coming back whenever the GC happens to reach each
    // tensor. They were no-ops on IBackend and implemented only by CUDA, so every one of those call sites did
    // nothing at all on any other GPU backend.

    /// <summary>Drops cached activations and returns what the pool was holding, keeping resident weights.</summary>
    public void FreeActivations() => FreeActivations(trimPool: true);

    /// <summary>As <see cref="FreeActivations()"/>, with control over the pool trim.</summary>
    /// <remarks>Hot per-step callers pass false: the next iteration re-uses the reservation directly, and a trim
    /// there costs a multi-gigabyte driver release and re-map every iteration for memory that is about to be
    /// asked for again.</remarks>
    public void FreeActivations(bool trimPool)
    {
        using OpScope _ = EnterOp();
        OnActivationsFreeing();
        Residency.FreeActivations();
        if (trimPool)
        {
            TrimMemoryPoolCore();
        }
    }

    /// <summary>Returns pool-reserved-but-free device memory to the driver, keeping every cache intact.</summary>
    public void TrimMemoryPool()
    {
        using OpScope _ = EnterOp();
        TrimMemoryPoolCore();
    }

    /// <summary>Hands back whatever this backend's allocator is holding but not using.</summary>
    /// <remarks>Abstract rather than a virtual no-op: a backend that genuinely has nothing to trim should say so
    /// with an empty body, because inheriting silence here is exactly how this whole set came to do nothing.</remarks>
    protected abstract void TrimMemoryPoolCore();

    /// <summary>Releases every cached device allocation — weights, conversions and activations — for a clean slate.</summary>
    /// <remarks>Each step runs even if an earlier one failed, and the first failure is rethrown at the end: these
    /// are independent native resources, and letting one failure strand the rest would leave the card full for a
    /// reason the caller cannot see.</remarks>
    public void FreeAllDeviceMemory()
    {
        using OpScope _ = EnterOp();
        Exception? failure = null;
        long freeBefore = Probe();
        Attempt(OnActivationsFreeing);
        Attempt(Residency.FreeAllCached);
        Attempt(OnAllDeviceMemoryFreed);
        Attempt(TrimMemoryPoolCore);
        long freeAfter = Probe();
        if (freeBefore >= 0 && freeAfter >= 0)
        {
            Logs.Info($"[{GetType().Name}] FreeAllDeviceMemory: free {freeBefore >> 20} MB → {freeAfter >> 20} MB");
        }
        if (failure is not null)
        {
            throw new InvalidOperationException(
                $"One or more device resources failed to release during {GetType().Name}'s full memory sweep.",
                failure);
        }

        long Probe()
        {
            try { return GetVramInfo().FreeBytes; }
            catch (Exception ex) { Logs.Debug($"[{GetType().Name}] VRAM probe failed: {ex.Message}"); return -1; }
        }

        void Attempt(Action step)
        {
            try { step(); }
            catch (Exception ex) { failure ??= ex; }
        }
    }

    /// <summary>Called before activations are released, at the phase boundary that releases them.</summary>
    /// <remarks>Where a backend invalidates anything that baked an activation's device address. A captured step
    /// graph is the case that exists: it holds the addresses of the buffers about to be freed, so replaying it
    /// afterwards reads memory the driver has taken back.</remarks>
    protected virtual void OnActivationsFreeing() { }

    /// <summary>Called during <see cref="FreeAllDeviceMemory"/>, after the caches are dropped.</summary>
    /// <remarks>Where a backend releases device memory its residency cache never owned — a vendor library's plan
    /// cache and workspaces, a side table keyed by weight.</remarks>
    protected virtual void OnAllDeviceMemoryFreed() { }

    /// <summary>Idempotent teardown.</summary>
    /// <remarks>Idempotent because a backend is disposed from more than one place in practice — a pipeline's
    /// <c>finally</c> and a test's <c>using</c> reaching the same object — and the second call must be a no-op
    /// rather than a double free.</remarks>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }
        DisposeCore();
        GC.SuppressFinalize(this);
    }

    /// <summary>Backend teardown, run exactly once.</summary>
    protected abstract void DisposeCore();
}
