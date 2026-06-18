using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Utilities;

/// <summary>Training-free across-step feature cache for diffusion backbones (the TeaCache / First-Block-Cache family). Adjacent denoise steps produce near-identical deep features, so the heavy block stack is run only when a cheap proxy signal has drifted enough; otherwise the previous step's <b>residual</b> (output − input of the cached region) is reused.
///
/// <para><b>How a backbone wires it.</b> At the start of a step, compute a cheap proxy (TeaCache: the timestep-modulated input; FBCache: the first transformer block's output) and call <see cref="ShouldCompute"/>. On a cache hit (returns false), skip the remaining blocks and call <see cref="ApplyResidual"/> to reconstruct the output from the step input. On a miss (returns true), run the full stack and call <see cref="StoreResidual"/> with the region's input and output.</para>
///
/// <para>The gate accumulates the relative-L1 distance of the proxy across skipped steps and forces a recompute once it crosses <see cref="Threshold"/> or after <see cref="MaxConsecutiveReuse"/> reuses, bounding error drift. Operates on F32 CPU-accessible tensors; feed it the small proxy (modulation/timestep signal) rather than full hidden states to keep the gate cheap.</para></summary>
public sealed unsafe class FeatureCache : IDisposable
{
    private readonly float _threshold;
    private readonly int _maxConsecutiveReuse;
    private float _accumulatedDistance;
    private int _consecutiveReuse;
    private Tensor? _prevProxy;
    private Tensor? _cachedResidual;
    private bool _disposed;

    /// <summary>Relative-L1 drift budget; once the accumulated proxy distance exceeds this, the next step recomputes. Lower = more recomputes = higher fidelity. Typical 0.1-0.25.</summary>
    public float Threshold => _threshold;

    /// <summary>Number of steps reused so far (diagnostic).</summary>
    public int Reuses { get; private set; }

    /// <summary>Number of full recomputes so far (diagnostic).</summary>
    public int Computes { get; private set; }

    /// <summary>Creates a feature cache.</summary>
    /// <param name="threshold">Accumulated relative-L1 drift budget before forcing a recompute.</param>
    /// <param name="maxConsecutiveReuse">Hard cap on consecutive reuses regardless of the gate, bounding worst-case drift.</param>
    public FeatureCache(float threshold = 0.15f, int maxConsecutiveReuse = 3)
    {
        if (threshold <= 0.0f) throw new ArgumentOutOfRangeException(nameof(threshold), "Threshold must be positive.");
        if (maxConsecutiveReuse < 1) throw new ArgumentOutOfRangeException(nameof(maxConsecutiveReuse));
        _threshold = threshold;
        _maxConsecutiveReuse = maxConsecutiveReuse;
    }

    /// <summary>Decides whether this step must run the full block stack. Returns true (recompute) on the first step, when no residual is cached yet, when the accumulated proxy drift crosses <see cref="Threshold"/>, or when the consecutive-reuse cap is hit; returns false (reuse the cached residual) otherwise. Updates internal drift state from <paramref name="proxy"/>.</summary>
    public bool ShouldCompute(Tensor proxy)
    {
        if (proxy is null) throw new ArgumentNullException(nameof(proxy));
        ThrowIfDisposed();

        if (_prevProxy is null || _cachedResidual is null)
        {
            StorePrevProxy(proxy);
            return MarkCompute();
        }

        float rel = RelativeL1(proxy, _prevProxy);
        _accumulatedDistance += rel;
        StorePrevProxy(proxy);

        if (_accumulatedDistance >= _threshold || _consecutiveReuse >= _maxConsecutiveReuse)
        {
            return MarkCompute();
        }

        // Reuse: keep the accumulated drift (added above) so it builds across consecutive
        // reused steps and eventually trips the threshold, forcing a refresh.
        _consecutiveReuse++;
        Reuses++;
        return false;
    }

    /// <summary>Stores the residual (<paramref name="output"/> − <paramref name="input"/>) of the cached region after a full compute, for reuse on subsequent cache hits. Both tensors must be F32 with identical shape.</summary>
    public void StoreResidual(Tensor input, Tensor output)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (output is null) throw new ArgumentNullException(nameof(output));
        if (!input.Shape.Equals(output.Shape))
            throw new ArgumentException($"StoreResidual shape mismatch: input={input.Shape}, output={output.Shape}.");
        if (input.DType != DType.F32 || output.DType != DType.F32)
            throw new ArgumentException("FeatureCache operates on F32 tensors.");
        ThrowIfDisposed();

        _cachedResidual?.Dispose();
        _cachedResidual = new Tensor(output.Shape, DType.F32);
        int count = (int)output.ElementCount;
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        float* resPtr = (float*)_cachedResidual.DataPointer;
        for (int i = 0; i < count; i++)
        {
            resPtr[i] = outPtr[i] - inPtr[i];
        }
    }

    /// <summary>Reconstructs the region output on a cache hit: <paramref name="output"/> = <paramref name="input"/> + cached residual. Requires a prior <see cref="StoreResidual"/>.</summary>
    public void ApplyResidual(Tensor input, Tensor output)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (output is null) throw new ArgumentNullException(nameof(output));
        ThrowIfDisposed();
        if (_cachedResidual is null)
            throw new InvalidOperationException("ApplyResidual called before any StoreResidual.");
        if (!input.Shape.Equals(_cachedResidual.Shape))
            throw new ArgumentException($"ApplyResidual shape mismatch: input={input.Shape}, residual={_cachedResidual.Shape}.");

        int count = (int)input.ElementCount;
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        float* resPtr = (float*)_cachedResidual.DataPointer;
        for (int i = 0; i < count; i++)
        {
            outPtr[i] = inPtr[i] + resPtr[i];
        }
    }

    /// <summary>Resets all drift and cached state, e.g. between independent generations sharing the cache instance.</summary>
    public void Reset()
    {
        _accumulatedDistance = 0.0f;
        _consecutiveReuse = 0;
        Reuses = 0;
        Computes = 0;
        _prevProxy?.Dispose();
        _prevProxy = null;
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

    private void StorePrevProxy(Tensor proxy)
    {
        if (_prevProxy is null || !_prevProxy.Shape.Equals(proxy.Shape))
        {
            _prevProxy?.Dispose();
            _prevProxy = new Tensor(proxy.Shape, DType.F32);
        }
        int count = (int)proxy.ElementCount;
        Buffer.MemoryCopy((void*)proxy.DataPointer, (void*)_prevProxy.DataPointer,
            (long)count * sizeof(float), (long)count * sizeof(float));
    }

    private static float RelativeL1(Tensor a, Tensor b)
    {
        int count = (int)a.ElementCount;
        float* ap = (float*)a.DataPointer;
        float* bp = (float*)b.DataPointer;
        double diff = 0.0, denom = 0.0;
        for (int i = 0; i < count; i++)
        {
            diff += Math.Abs(ap[i] - bp[i]);
            denom += Math.Abs(bp[i]);
        }
        return denom > 0.0 ? (float)(diff / denom) : 0.0f;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(FeatureCache));
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _prevProxy?.Dispose();
        _prevProxy = null;
        _cachedResidual?.Dispose();
        _cachedResidual = null;
    }
}
