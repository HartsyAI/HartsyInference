using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Typed vision surface: single-image operations (embed, detect, segment, maps, geometry, pose, faces, matting,
/// upscale), multi-view operations over several views of one scene, and streamed operations over video frames.</summary>
public interface IVisionService
{
    /// <summary>Runs the single-image operation in <paramref name="request"/>.</summary>
    Task<VisionResult> RunAsync(ModelSpec spec, VisionRequest request, CancellationToken cancel = default);

    /// <summary>Runs a multi-view operation (<see cref="VisionSequenceKind.Views"/>): one result whose
    /// <see cref="VisionResult.Views"/> holds a result per view, in input order. Depth Pro 2 MetricDepth is the one
    /// implementation in this build.</summary>
    Task<VisionResult> RunViewsAsync(ModelSpec spec, VisionBatchRequest request, CancellationToken cancel = default) =>
        throw new NotSupportedException($"{GetType().Name} does not run multi-view requests.");

    /// <summary>Streams a frame operation (<see cref="VisionSequenceKind.Frames"/>: tracking, interpolation, video metric depth), one
    /// <see cref="VisionFrameResult"/> per output frame in time order. Stopping the enumeration early, or cancelling,
    /// stops the model and frees what it holds. Depth Pro 2 MetricDepth is the one implementation in this build.</summary>
    IAsyncEnumerable<VisionFrameResult> RunFramesAsync(ModelSpec spec, VisionBatchRequest request, CancellationToken cancel = default) =>
        throw new NotSupportedException($"{GetType().Name} does not run frame sequences.");
}
