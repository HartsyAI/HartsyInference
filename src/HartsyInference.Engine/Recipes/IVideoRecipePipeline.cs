using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.Engine.Recipes;

/// <summary>A constructed, ready-to-run video pipeline for one architecture family. Owns the loaded components and
/// drives that family's bespoke encode + denoise + decode, returning the decoded frame sequence. Cached per loaded
/// model and reused across requests.</summary>
public interface IVideoRecipePipeline : IDisposable
{
    /// <summary>Generates the frame sequence for <paramref name="request"/>, reporting step progress.</summary>
    IReadOnlyList<VideoFrame> Generate(VideoRequest request, IProgress<StepPreview>? progress, CancellationToken cancel);
}
