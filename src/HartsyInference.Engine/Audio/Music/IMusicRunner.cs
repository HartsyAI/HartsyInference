using HartsyInference.Core.Backends;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Audio;

/// <summary>A loaded music model reduced to: prompt → PCM at <see cref="SampleRate"/>. The cancellation token is observed inside the synth loop so a stop request interrupts long autoregressive decodes mid-flight.</summary>
internal interface IMusicRunner : IDisposable
{
    /// <summary>Output sample rate in Hz.</summary>
    int SampleRate { get; }

    /// <summary>Generates audio for <paramref name="request"/>.</summary>
    MusicAudio Synthesize(IBackend backend, MusicRequest request, CancellationToken cancel);

    /// <summary>Writes the symbolic plan without rendering it; null on models that do not plan one.</summary>
    ScorePlanResult? PlanScore(IBackend backend, MusicRequest request, CancellationToken cancel);

    /// <summary>Reports what the context leaves for audio; null on models with no such budget.</summary>
    ScorePlanResult? Budget(MusicRequest request);
}
