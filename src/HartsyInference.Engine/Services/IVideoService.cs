using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Typed video-generation surface: streams decoded frames as they are produced.</summary>
public interface IVideoService
{
    /// <summary>Generates a video for <paramref name="request"/>, yielding each frame as it decodes.</summary>
    IAsyncEnumerable<VideoFrame> GenerateAsync(ModelSpec spec, VideoRequest request, IProgress<StepPreview>? progress = null, CancellationToken cancel = default);
}
