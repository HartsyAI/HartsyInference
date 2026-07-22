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
    private readonly float _threshold;
    private readonly int _maxConsecutiveReuse;
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
    public DeviceFeatureCache(float threshold = 0.10f, int maxConsecutiveReuse = 3)
    {
        if (threshold <= 0.0f) throw new ArgumentOutOfRangeException(nameof(threshold), "Threshold must be positive.");
        if (maxConsecutiveReuse < 1) throw new ArgumentOutOfRangeException(nameof(maxConsecutiveReuse));
        _threshold = threshold;
        _maxConsecutiveReuse = maxConsecutiveReuse;
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
        _accumulatedDistance += rel;
        SnapshotIndicator(backend, indicator);

        if (_accumulatedDistance >= _threshold || _consecutiveReuse >= _maxConsecutiveReuse)
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

        _cachedResidual?.Dispose();
        _cachedResidual = new Tensor(output.Shape, output.DType);
        backend.Add(_cachedResidual, output, negInput);
        // Cross-step state: the residual's only copy is on-device — survive the video pipelines' per-step
        // FreeActivations (its own Dispose still reclaims it; pin is a no-op on host backends).
        backend.PinActivation(_cachedResidual);
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
