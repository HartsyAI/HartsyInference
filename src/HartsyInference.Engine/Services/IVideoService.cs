using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Typed video-generation surface: returns the decoded frames together with the soundtrack that belongs with them.</summary>
public interface IVideoService
{
    /// <summary>Generates a video for <paramref name="request"/>, including any audio the model produced or the request supplied for mux.</summary>
    Task<VideoGenerationResult> GenerateAsync(ModelSpec spec, VideoRequest request, IProgress<StepPreview>? progress = null, CancellationToken cancel = default);
}
