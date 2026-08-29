using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.Engine.Services;

namespace HartsyInference.Engine.Recipes;

/// <summary>Preserves diffusion progress metadata while adapting borrowed pipeline latents into consumer-safe RGB previews.</summary>
internal static class RecipeProgressAdapter
{
    /// <summary>Creates the synchronous bridge recipes pass into image and video pipelines.</summary>
    internal static Action<GenerationProgress> Create(IProgress<StepPreview>? target, CancellationToken cancel,
        int stepOffset = 0, int? totalSteps = null)
    {
        return source =>
        {
            cancel.ThrowIfCancellationRequested();
            if (target is null)
            {
                return;
            }

            byte[]? previewRgb = null;
            int previewWidth = 0;
            int previewHeight = 0;
            if (source.Latent is not null)
            {
                previewRgb = LatentPreview.DecodeLatent2Rgb(
                    source.Latent, source.LatentArch, out previewWidth, out previewHeight);
            }

            target.Report(new StepPreview
            {
                Step = stepOffset + source.Step,
                TotalSteps = totalSteps ?? source.TotalSteps,
                PreviewRgb = previewRgb,
                PreviewWidth = previewWidth,
                PreviewHeight = previewHeight,
            });
        };
    }
}
