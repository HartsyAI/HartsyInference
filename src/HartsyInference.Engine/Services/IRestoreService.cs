using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Video/image restoration (SeedVR2): degraded media in, restored frames out at the area-normalized target size. Frame-streaming shape matches <see cref="IVideoService"/> so the CLI/SSE frame plumbing is shared.</summary>
public interface IRestoreService
{
    /// <summary>Restores the request's video/frames/image and streams the restored frames in order.</summary>
    IAsyncEnumerable<VideoFrame> RestoreAsync(ModelSpec spec, RestoreRequest request,
        IProgress<StepPreview>? progress = null, CancellationToken cancel = default);
}
