using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Services;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Contract coverage for the recipe boundary that previously discarded pipeline-supplied image and video latents.</summary>
public sealed class RecipeProgressAdapterTests
{
    [Fact]
    public void ImageLatent_PopulatesRgbPreview()
    {
        using Tensor latent = new Tensor(new TensorShape(1, 16, 4, 6), DType.F32);
        CapturingProgress target = new CapturingProgress();
        Action<GenerationProgress> bridge = RecipeProgressAdapter.Create(target, CancellationToken.None);

        bridge(new GenerationProgress(3, 8, 10.0)
        {
            Latent = latent,
            LatentArch = LatentArchitecture.ZImage,
        });

        Assert.Equal(3, target.Value.Step);
        Assert.Equal(8, target.Value.TotalSteps);
        Assert.Equal(6, target.Value.PreviewWidth);
        Assert.Equal(4, target.Value.PreviewHeight);
        Assert.NotNull(target.Value.PreviewRgb);
        Assert.Equal(6 * 4 * 3, target.Value.PreviewRgb!.Length);
    }

    [Fact]
    public void VideoLatent_PopulatesRgbPreviewAndPreservesChunkProgress()
    {
        using Tensor latent = new Tensor(new TensorShape([1, 128, 3, 4, 6]), DType.F32);
        CapturingProgress target = new CapturingProgress();
        Action<GenerationProgress> bridge = RecipeProgressAdapter.Create(
            target, CancellationToken.None, stepOffset: 8, totalSteps: 24);

        bridge(new GenerationProgress(2, 8, 10.0)
        {
            Latent = latent,
            LatentArch = LatentArchitecture.Ltx,
        });

        Assert.Equal(10, target.Value.Step);
        Assert.Equal(24, target.Value.TotalSteps);
        Assert.Equal(6, target.Value.PreviewWidth);
        Assert.Equal(4, target.Value.PreviewHeight);
        Assert.NotNull(target.Value.PreviewRgb);
        Assert.Equal(6 * 4 * 3, target.Value.PreviewRgb!.Length);
        Assert.NotNull(target.Value.PreviewFramesRgb);
        Assert.Equal(3, target.Value.PreviewFramesRgb!.Length);
    }

    [Fact]
    public void MissingLatent_StillReportsStepWithoutPreview()
    {
        CapturingProgress target = new CapturingProgress();
        Action<GenerationProgress> bridge = RecipeProgressAdapter.Create(target, CancellationToken.None);

        bridge(new GenerationProgress(1, 5, 10.0));

        Assert.Equal(1, target.Value.Step);
        Assert.Equal(5, target.Value.TotalSteps);
        Assert.Null(target.Value.PreviewRgb);
        Assert.Equal(0, target.Value.PreviewWidth);
        Assert.Equal(0, target.Value.PreviewHeight);
    }

    private sealed class CapturingProgress : IProgress<StepPreview>
    {
        public StepPreview Value { get; private set; }

        public void Report(StepPreview value)
        {
            Value = value;
        }
    }
}
