using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Video-generation service. Wired by the architecture-recipe phase (E-IMG-3); not yet available.</summary>
public sealed class VideoService : IVideoService
{
    private readonly InferenceEngine _engine;

    /// <summary>Creates the service bound to its owning engine.</summary>
    internal VideoService(InferenceEngine engine) => _engine = engine;

    /// <inheritdoc/>
    public IAsyncEnumerable<VideoFrame> GenerateAsync(ModelSpec spec, VideoRequest request, IProgress<StepPreview>? progress = null, CancellationToken cancel = default) =>
        throw new NotSupportedException("Video generation is wired by the architecture-recipe phase (E-IMG-3); not yet available.");
}
