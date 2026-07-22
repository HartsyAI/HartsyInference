using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Video-generation service: resolves the checkpoint's family recipe from the
/// <see cref="VideoRecipeRegistry"/>, constructs (and caches) the pipeline, and streams the decoded frames as they
/// become available. Composition features are applied by the feature-resolver phase (E-IMG-4); a request that sets
/// one is rejected rather than silently ignored.</summary>
public sealed class VideoService : IVideoService
{
    private readonly InferenceEngine _engine;

    /// <summary>Creates the service bound to its owning engine.</summary>
    internal VideoService(InferenceEngine engine) => _engine = engine;

    /// <inheritdoc/>
    public async IAsyncEnumerable<VideoFrame> GenerateAsync(ModelSpec spec, VideoRequest request, IProgress<StepPreview>? progress = null,
        [EnumeratorCancellation] CancellationToken cancel = default)
    {
        if (request.Loras is not null)
        {
            throw new NotSupportedException(
                "Video composition features (LoRA) are applied by the feature-resolver phase (E-IMG-4); not yet available.");
        }

        IReadOnlyList<VideoFrame> frames = await Task.Run(
            () =>
            {
                IVideoRecipePipeline pipeline = _engine.GetOrConstructVideoRecipe(spec);
                VideoRequest resolved = InferenceEngine.VideoDefaultsFor(spec).Apply(request);
                return pipeline.Generate(resolved, progress, cancel);
            },
            cancel).ConfigureAwait(false);

        foreach (VideoFrame frame in frames)
        {
            cancel.ThrowIfCancellationRequested();
            yield return frame;
        }
    }
}
