using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.Engine.Recipes;

/// <summary>A constructed, ready-to-run video pipeline for one architecture family. Owns the loaded components and
/// drives that family's bespoke encode + denoise + decode, returning the decoded frames and any soundtrack that
/// belongs with them. Cached per loaded model and reused across requests.</summary>
public interface IVideoRecipePipeline : IDisposable
{
    /// <summary>Generates for <paramref name="request"/>, reporting step progress. Attach audio to the result only when it is meant to be heard — a family that consumes audio purely as conditioning leaves it null.</summary>
    VideoGenerationResult Generate(VideoRequest request, IProgress<StepPreview>? progress, CancellationToken cancel);
}
