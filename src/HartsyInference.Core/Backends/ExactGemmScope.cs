namespace HartsyInference.Core.Backends;

/// <summary>
/// Turns <see cref="IBackend.HighPrecisionGemm"/> on for a model's forward pass and puts the previous setting back, even when the pass throws. For models whose
/// reference is plain F32 and whose output is a measured quantity (metric depth): on CUDA the F32 GEMMs and attention then skip TF32, which is about
/// 1e-3 off a plain-F32 lane, instead of the engine-wide TF32 default. Backends without TF32 ignore the switch.
/// <para>On CUDA the switch also makes a BF16 weight upcast to F32 on every call (no cached cast), so a BF16 or F16 model would not get half-precision GEMMs under
/// this scope; it is meant for models whose weights are F32.</para>
/// <para>Thread safety: the switch is one property of the backend, saved and restored with no synchronisation. Two threads that open scopes on one backend can
/// restore each other's value, and work another thread runs on that backend while a scope is open also gets exact GEMMs. Give each concurrent request its own
/// backend, or serialise the requests that use scopes.</para>
/// </summary>
public readonly struct ExactGemmScope : IDisposable
{
    private readonly IBackend _backend;
    private readonly bool _previous;

    /// <summary>Switches <paramref name="backend"/> to exact GEMMs until the scope is disposed.</summary>
    public ExactGemmScope(IBackend backend)
    {
        ArgumentNullException.ThrowIfNull(backend);
        _backend = backend;
        _previous = backend.HighPrecisionGemm;
        backend.HighPrecisionGemm = true;
    }

    /// <summary>Restores the setting the backend had when the scope began.</summary>
    public void Dispose() => _backend.HighPrecisionGemm = _previous;
}
