using BenchmarkDotNet.Attributes;
using HartsyInference.Core.Configuration;

namespace HartsyInference.Benchmarks;

/// <summary>What one knob read costs, which is the number that decides whether knob values may be read live.
///
/// <para>Runtime knobs are read at the point of use so a per-request profile can reach them. The obvious worry is
/// that a resolve on a hot path is too expensive and the values should be snapshotted once per generation instead.
/// This exists so that argument is settled with a measurement rather than an intuition, and so the next person to
/// change <see cref="KnobStore"/>'s internals can see what they moved.</para>
///
/// <para>The scoped case is the one that matters: the image and video services push a
/// <see cref="KnobProfileScope"/> for every request, so a production read pays the profile lookup as well. Against
/// it, weigh how often the hottest reader actually runs — the per-op orphan sweep is read about ten thousand times
/// in a 1024x1024 eight-step image, so a cost of tens of nanoseconds is under a millisecond across the whole
/// generation.</para></summary>
[MemoryDiagnoser]
[SimpleJob]
public sealed class KnobResolutionBenchmarks
{
    private IDisposable? _scope;

    /// <summary>Forces the declarations to run so the first timed read is not paying for registration.</summary>
    [GlobalSetup]
    public void Setup() => _ = EngineKnobs.OrphanSweep.Value;

    [Benchmark(Baseline = true, Description = "Knob.Value, no profile scope")]
    public bool Unscoped() => EngineKnobs.OrphanSweep.Value;

    [Benchmark(Description = "Knob.Value, request profile scope live")]
    public bool Scoped()
    {
        _scope ??= KnobProfileScope.Push(KnobProfiles.Reference);
        return EngineKnobs.OrphanSweep.Value;
    }

    /// <summary>An explicit override, the path a <c>--set</c> or a test takes.</summary>
    [Benchmark(Description = "Knob.Value, explicit override set")]
    public bool Overridden()
    {
        KnobStore.Set(EngineKnobs.OrphanSweep, true);
        return EngineKnobs.OrphanSweep.Value;
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        KnobStore.Clear(EngineKnobs.OrphanSweep);
        _scope?.Dispose();
        _scope = null;
    }
}
