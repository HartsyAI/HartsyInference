using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Typed vision surface: embed, detect (RT-DETR/GroundingDINO/YOLO), and segment (SAM/ClipSeg).</summary>
public interface IVisionService
{
    /// <summary>Runs the embed/detect/segment operation in <paramref name="request"/>.</summary>
    Task<VisionResult> RunAsync(ModelSpec spec, VisionRequest request, CancellationToken cancel = default);
}
