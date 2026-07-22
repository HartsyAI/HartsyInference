using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Typed text-to-music surface (MusicGen/AudioGen/ACE-Step/YuE/HeartMuLa), including
/// continuation/repaint/cover editing modes.</summary>
public interface IMusicService
{
    /// <summary>Generates music for <paramref name="request"/>.</summary>
    Task<AudioResult> GenerateAsync(ModelSpec spec, MusicRequest request, IProgress<StepPreview>? progress = null, CancellationToken cancel = default);
}
