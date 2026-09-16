using HartsyInference.Core.Backends;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Audio;

/// <summary>Wraps a synth delegate plus the disposables a loaded model owns, so each model is a descriptor.</summary>
internal sealed class MusicRunner(
    int sampleRate,
    Func<IBackend, MusicRequest, CancellationToken, MusicAudio> synth,
    params IDisposable?[] disposables) : IMusicRunner
{
    /// <summary>Optional score-planning delegate; left null by models that render straight from the prompt.</summary>
    internal Func<IBackend, MusicRequest, CancellationToken, ScorePlanResult>? Planner { get; init; }

    /// <summary>Optional context-budget delegate; left null by models with no token budget to report.</summary>
    internal Func<MusicRequest, ScorePlanResult>? Budgeter { get; init; }

    /// <inheritdoc/>
    public int SampleRate => sampleRate;

    /// <inheritdoc/>
    public MusicAudio Synthesize(IBackend backend, MusicRequest request, CancellationToken cancel) => synth(backend, request, cancel);

    /// <inheritdoc/>
    public ScorePlanResult? PlanScore(IBackend backend, MusicRequest request, CancellationToken cancel)
        => Planner?.Invoke(backend, request, cancel);

    /// <inheritdoc/>
    public ScorePlanResult? Budget(MusicRequest request) => Budgeter?.Invoke(request);

    /// <inheritdoc/>
    public void Dispose()
    {
        foreach (IDisposable? disposable in disposables)
        {
            disposable?.Dispose();
        }
    }
}
