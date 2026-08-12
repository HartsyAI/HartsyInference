using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Utilities;

/// <summary>Device-resident across-step feature cache for GPU-resident DiT forwards (First-Block-Cache /
/// TeaCache family — docs/Research/STEP_ACCELERATION.md §2). The CPU-side <see cref="FeatureCache"/> reads
/// <c>DataPointer</c>, which on CUDA evicts cached activations mid-forward (the stream-drain pathology); this
/// variant keeps every tensor on-device by routing all math through <see cref="IBackend"/> ops — the gate
/// metric via <see cref="IBackend.RelativeL1Distance"/>, residual store via Scale(−1)+Add, residual apply via
/// Add — so the only host traffic is the metric's 8-byte scalar readback.
///
/// <para><b>Wiring (transformer level).</b> Run block 0 as normal; call <see cref="ShouldCompute"/> with its
/// output stream as the indicator. Miss: run blocks 1..N, then <see cref="StoreResidual"/>(block0Out, finalOut).
/// Hit: skip blocks 1..N and take <see cref="ApplyResidual"/>(block0Out) as the final hidden state. The final
/// norm/projection always runs. Each CFG stream (cond / uncond) needs its OWN instance — their hidden states
/// differ. Gate creation on <see cref="IBackend.SupportsDeviceStepCacheGate"/>.</para>
///
/// <para>The gate accumulates the indicator's relative-L1 drift across skipped steps and forces a recompute
/// once it crosses <see cref="Threshold"/> or after <see cref="MaxConsecutiveReuse"/> reuses. Works in the
/// stream's native dtype (F32/F16 — Scale/Add/RelativeL1Distance all dispatch per dtype).</para></summary>
public sealed class DeviceFeatureCache : IDisposable
{
    /// <summary>Last-resort lowvram lever: <c>HARTSY_STEP_CACHE_OFFLOAD=1</c> pages the cross-step residual and
    /// indicator snapshot to host as they are produced, trading one PCIe round trip per step for their full device
    /// size. Off by default. Defensible only because both are read at most ONCE PER STEP — the same treatment applied
    /// to a per-block tensor loses outright (docs/Research/MEMORY_SCHEDULING_SERVING.md §9).</summary>
    private static readonly bool OffloadEnabled =
        Environment.GetEnvironmentVariable("HARTSY_STEP_CACHE_OFFLOAD") == "1";

    private readonly float _threshold;
    private readonly int _maxConsecutiveReuse;
    private readonly float[]? _polyCoeffs;      // TeaCache-style gate calibration: drift ← Σ cᵢ·relⁱ (c0 first)
    private readonly string? _calibFile;        // when set (HARTSY_STEP_CACHE_CALIB), logs "indicatorRel,residualRel" pairs
    private float _lastIndicatorRel = -1f;
    private float _accumulatedDistance;
    private int _consecutiveReuse;
    private Tensor? _prevIndicator;
    private Tensor? _cachedResidual;
    private bool _disposed;

    /// <summary>Accumulated relative-L1 drift budget; crossing it forces the next step to recompute. Lower =
    /// more recomputes = higher fidelity. Literature range 0.05–0.25 for DiT residual caching.</summary>
    public float Threshold => _threshold;

    /// <summary>Steps served from the cached residual so far (diagnostic, logged for benchmark records).</summary>
    public int Reuses { get; private set; }

    /// <summary>Full block-stack computes so far (diagnostic).</summary>
    public int Computes { get; private set; }

    /// <summary>Creates a device-resident step cache.</summary>
    /// <param name="threshold">Accumulated relative-L1 drift budget before forcing a recompute.</param>
    /// <param name="maxConsecutiveReuse">Hard cap on consecutive reuses regardless of drift, bounding worst-case error.</param>
    public DeviceFeatureCache(float threshold = 0.10f, int maxConsecutiveReuse = 3,
        float[]? polyCoeffs = null, string? calibFile = null)
    {
        if (!float.IsFinite(threshold) || threshold <= 0.0f)
            throw new ArgumentOutOfRangeException(nameof(threshold), "Threshold must be finite and positive.");
        if (maxConsecutiveReuse < 1) throw new ArgumentOutOfRangeException(nameof(maxConsecutiveReuse));
        if (polyCoeffs is not null && Array.Exists(polyCoeffs, coefficient => !float.IsFinite(coefficient)))
            throw new ArgumentException("Calibration polynomial coefficients must be finite.", nameof(polyCoeffs));
        _threshold = threshold;
        _maxConsecutiveReuse = maxConsecutiveReuse;
        _polyCoeffs = polyCoeffs is null ? null : (float[])polyCoeffs.Clone();
        _calibFile = calibFile;
    }

    /// <summary>Maps the raw indicator drift through the calibration polynomial (identity when none).</summary>
    private float MapDrift(float rel)
    {
        if (_polyCoeffs is null) return rel;
        float acc = 0f, pow = 1f;
        foreach (float c in _polyCoeffs) { acc += c * pow; pow *= rel; }
        return acc < 0f ? 0f : acc;   // a fitted poly can dip negative near 0 — drift budgets are non-negative
    }

    /// <summary>Decides whether this step must run the full block stack, updating drift state from
    /// <paramref name="indicator"/> (typically the first block's output stream). True on the first step, when no
    /// residual is cached, when accumulated drift crosses <see cref="Threshold"/>, or at the reuse cap; false =
    /// reuse via <see cref="ApplyResidual"/>. The indicator is snapshotted device-side; the caller keeps ownership.</summary>
    public bool ShouldCompute(IBackend backend, Tensor indicator)
    {
        if (backend is null) throw new ArgumentNullException(nameof(backend));
        if (indicator is null) throw new ArgumentNullException(nameof(indicator));
        ThrowIfDisposed();

        if (_prevIndicator is null || _cachedResidual is null || !_prevIndicator.Shape.Equals(indicator.Shape)
            || _prevIndicator.DType != indicator.DType)
        {
            SnapshotIndicator(backend, indicator);
            return MarkCompute();
        }

        float rel = backend.RelativeL1Distance(indicator, _prevIndicator);
        _lastIndicatorRel = rel;
        SnapshotIndicator(backend, indicator);

        // A nonzero tensor relative to an all-zero snapshot has infinite relative drift. Polynomial
        // evaluation can turn that into NaN through mixed-sign terms; either case must invalidate the cache.
        if (!float.IsFinite(rel))
        {
            _lastIndicatorRel = -1f;
            return MarkCompute();
        }

        float drift = MapDrift(rel);
        if (!float.IsFinite(drift))
            return MarkCompute();
        _accumulatedDistance += drift;

        if (!float.IsFinite(_accumulatedDistance) || _accumulatedDistance >= _threshold
            || _consecutiveReuse >= _maxConsecutiveReuse)
        {
            return MarkCompute();
        }

        // Keep the drift added above so it builds across consecutive reused steps and
        // eventually trips the threshold, forcing a refresh.
        _consecutiveReuse++;
        Reuses++;
        return false;
    }

    /// <summary>Stores the cached-region residual (<paramref name="output"/> − <paramref name="input"/>) after a
    /// full compute, entirely device-side (Scale −1 into a scratch, then Add). Shapes and dtypes must match.</summary>
    public void StoreResidual(IBackend backend, Tensor input, Tensor output)
    {
        if (backend is null) throw new ArgumentNullException(nameof(backend));
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (output is null) throw new ArgumentNullException(nameof(output));
        if (!input.Shape.Equals(output.Shape))
            throw new ArgumentException($"StoreResidual shape mismatch: input={input.Shape}, output={output.Shape}.");
        if (input.DType != output.DType)
            throw new ArgumentException($"StoreResidual dtype mismatch: input={input.DType}, output={output.DType}.");
        ThrowIfDisposed();

        using Tensor negInput = new Tensor(input.Shape, input.DType);
        backend.Scale(negInput, input, -1.0f);

        // Calibration mode: before replacing the residual, log (indicator drift → true residual drift) —
        // the pair the TeaCache-style polynomial is fitted on. Costs one extra reduction per compute step,
        // only when HARTSY_STEP_CACHE_CALIB is set.
        if (_calibFile is not null && _cachedResidual is not null && _lastIndicatorRel >= 0f
            && _cachedResidual.Shape.Equals(output.Shape))
        {
            using Tensor newResidual = new Tensor(output.Shape, output.DType);
            backend.Add(newResidual, output, negInput);
            float residualRel = backend.RelativeL1Distance(newResidual, _cachedResidual);
            File.AppendAllText(_calibFile, $"{_lastIndicatorRel:G9},{residualRel:G9}\n");
        }

        _cachedResidual?.Dispose();
        _cachedResidual = new Tensor(output.Shape, output.DType);
        backend.Add(_cachedResidual, output, negInput);
        // Cross-step state: the residual's only copy is on-device — survive the video pipelines' per-step
        // FreeActivations (its own Dispose still reclaims it; pin is a no-op on host backends).
        backend.PinActivation(_cachedResidual);
        OffloadIfEnabled(backend, _cachedResidual);
    }

    /// <summary>Reconstructs the cached-region output on a hit: returns a new tensor = <paramref name="input"/> +
    /// cached residual (device Add). Requires a prior <see cref="StoreResidual"/>; caller owns the returned tensor.</summary>
    public Tensor ApplyResidual(IBackend backend, Tensor input)
    {
        if (backend is null) throw new ArgumentNullException(nameof(backend));
        if (input is null) throw new ArgumentNullException(nameof(input));
        ThrowIfDisposed();
        if (_cachedResidual is null)
            throw new InvalidOperationException("ApplyResidual called before any StoreResidual.");
        if (!input.Shape.Equals(_cachedResidual.Shape))
            throw new ArgumentException($"ApplyResidual shape mismatch: input={input.Shape}, residual={_cachedResidual.Shape}.");
        if (input.DType != _cachedResidual.DType)
            throw new ArgumentException($"ApplyResidual dtype mismatch: input={input.DType}, residual={_cachedResidual.DType}.");

        Tensor output = new Tensor(input.Shape, input.DType);
        backend.Add(output, input, _cachedResidual);
        return output;
    }

    /// <summary>Clears all drift and cached state (call between independent generations sharing the instance).</summary>
    public void Reset()
    {
        _accumulatedDistance = 0.0f;
        _consecutiveReuse = 0;
        Reuses = 0;
        Computes = 0;
        _prevIndicator?.Dispose();
        _prevIndicator = null;
        _cachedResidual?.Dispose();
        _cachedResidual = null;
    }

    private bool MarkCompute()
    {
        _accumulatedDistance = 0.0f;
        _consecutiveReuse = 0;
        Computes++;
        return true;
    }

    private void SnapshotIndicator(IBackend backend, Tensor indicator)
    {
        // Fresh tensor per snapshot: Scale allocates a new device buffer and re-caching a reused tensor is
        // exactly the in-place-op pitfall (PHASE_3_DEVIATIONS #17) — allocation churn rides the async pool.
        Tensor snapshot = new Tensor(indicator.Shape, indicator.DType);
        backend.Scale(snapshot, indicator, 1.0f);
        backend.PinActivation(snapshot);   // cross-step gate state — survive per-step FreeActivations
        _prevIndicator?.Dispose();
        _prevIndicator = snapshot;
        OffloadIfEnabled(backend, snapshot);
    }

    /// <summary>Pages <paramref name="justProduced"/> out to host under <see cref="OffloadEnabled"/>, sized so the
    /// bulk walk frees roughly this one tensor. The walk is largest-first (pinned first within a size class), so an
    /// equally large live activation can be taken instead — bounded to one tensor's worth per call, and the entry
    /// point is deliberately the bulk one so the policy stays in one place.</summary>
    private static void OffloadIfEnabled(IBackend backend, Tensor justProduced)
    {
        if (!OffloadEnabled) return;
        // A captured step graph bakes activation pointers and a reload returns a new one — which is exactly why the
        // HunyuanVideo/Kandinsky5 pipelines suppress their per-step FreeActivations while their graph is live.
        if (backend.StepGraphReady || backend.StepGraphOwner is not null) return;
        backend.OffloadActivations(justProduced.DType.ComputeByteCount(justProduced.ElementCount));
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(DeviceFeatureCache));
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _prevIndicator?.Dispose();
        _prevIndicator = null;
        _cachedResidual?.Dispose();
        _cachedResidual = null;
    }
}
