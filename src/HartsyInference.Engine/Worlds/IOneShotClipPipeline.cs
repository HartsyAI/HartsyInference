using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Worlds;

/// <summary>A batch world model: one call renders the whole clip its plan describes. The engine wraps it in a session whose
/// frames are delivered once the call returns, so the model only has to produce the clip.</summary>
public interface IOneShotClipPipeline
{
    /// <summary>Renders the clip and returns exactly <see cref="WorldPlan.Frames"/> frames, the seed frame first, indexed 0 to n-1.
    /// Checks <paramref name="cancel"/> between stages; a run that observes it throws <see cref="OperationCanceledException"/>.</summary>
    IReadOnlyList<VideoFrame> GenerateClip(OneShotClipRequest request, CancellationToken cancel);
}
