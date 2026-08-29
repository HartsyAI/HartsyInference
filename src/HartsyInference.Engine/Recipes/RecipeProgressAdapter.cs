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
            byte[][]? previewFramesRgb = null;
            int previewWidth = 0;
            int previewHeight = 0;
            if (source.Latent is not null)
            {
                if (source.Latent.Shape.Rank == 5 && source.Latent.Shape[2] > 1)
                {
                    previewFramesRgb = LatentPreview.DecodeVideoLatent2RgbFrames(
                        source.Latent, source.LatentArch, out previewWidth, out previewHeight);
                    if (previewFramesRgb is { Length: > 0 })
                    {
                        previewRgb = previewFramesRgb[previewFramesRgb.Length / 2];
                    }
                }
                else
                {
                    previewRgb = LatentPreview.DecodeLatent2Rgb(
                        source.Latent, source.LatentArch, out previewWidth, out previewHeight);
                }
            }

            target.Report(new StepPreview
            {
                Step = stepOffset + source.Step,
                TotalSteps = totalSteps ?? source.TotalSteps,
                PreviewRgb = previewRgb,
                PreviewFramesRgb = previewFramesRgb,
                PreviewWidth = previewWidth,
                PreviewHeight = previewHeight,
            });
        };
    }
}
