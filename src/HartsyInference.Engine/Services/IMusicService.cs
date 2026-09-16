using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Typed text-to-music surface (MusicGen/AudioGen/ACE-Step/YuE/HeartMuLa), including continuation/repaint/cover editing modes.</summary>
public interface IMusicService
{
    /// <summary>Generates music for <paramref name="request"/>.</summary>
    Task<AudioResult> GenerateAsync(ModelSpec spec, MusicRequest request, IProgress<StepPreview>? progress = null, CancellationToken cancel = default);

    /// <summary>Writes the symbolic score for <paramref name="request"/> without rendering it — seconds rather
    /// than minutes, so a score can be planned, edited and fed back before any audio is paid for.</summary>
    /// <exception cref="NotSupportedException">The selected model does not plan a score.</exception>
    Task<ScorePlanResult> PlanScoreAsync(ModelSpec spec, MusicRequest request, CancellationToken cancel = default);

    /// <summary>Reports what the context leaves for audio behind this request's prompt and score.</summary>
    /// <exception cref="NotSupportedException">The selected model has no token budget to report.</exception>
    Task<ScorePlanResult> BudgetAsync(ModelSpec spec, MusicRequest request, CancellationToken cancel = default);
}
