using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>MiniMax-H3-Image transformer checks on a tiny geometry: the loader's audio rule, the stacked-head combine
/// against ComfyUI's <c>_pdd_head</c> arithmetic, and the schedule's grid rules.</summary>
public sealed class MiniMaxH3ImageTransformerTests
{
    private static MiniMaxH3Config TinyConfig(bool hasAudio) => new()
    {
        HiddenSize = 8,
        NumLayers = 1,
        TokenRefinerNumLayers = 0,
        NumAttentionHeads = 2,
        AttentionHeadDim = 4,
        FfnHiddenSize = 16,
        LatentsDim = 24,
        HasAudio = hasAudio,
        TextDim = 5,
        OutputHeadCount = 4,
    };

    [Fact]
    public void ImageLoaderNeedsNoAudioProjection()
    {
        using MiniMaxH3Transformer image = new(TinyConfig(hasAudio: false));

        image.LoadWeights(VideoOnlyWeights());
    }

    [Fact]
    public void AudioBuildLoaderStillRequiresAudioProjection()
    {
        using MiniMaxH3Transformer video = new(TinyConfig(hasAudio: true));

        Assert.Throws<KeyNotFoundException>(() => video.LoadWeights(VideoOnlyWeights()));
    }

    [Fact]
    public void StackedHeadCombineMatchesReferenceArithmetic()
    {
        // Four heads of two rows over three inputs. Block n holds the constant (n + 1) * 0.1 plus an input-dependent term,
        // so every block is distinct and a wrong block or weight shows up in the result.
        const int Heads = 4, OutRows = 2, InDim = 3;
        double[] grid = [1.0, 0.8, 0.5, 0.3, 0.0];
        StackedOutputHeads heads = new(grid);
        OutputHeadStep step = heads.Resolve(sigma: 0.8, sigmaNext: 0.3);
        using IBackend backend = new CpuBackend();
        using Tensor stackedWeight = new(new TensorShape(Heads * OutRows, InDim), DType.F32);
        using Tensor stackedBias = new(new TensorShape(Heads * OutRows), DType.F32);
        float[] weightValues = new float[Heads * OutRows * InDim];
        float[] biasValues = new float[Heads * OutRows];
        for (int row = 0; row < Heads * OutRows; row++)
        {
            biasValues[row] = (row + 1) * 0.25f;
            for (int col = 0; col < InDim; col++)
            {
                weightValues[row * InDim + col] = (row + 1) * 0.1f + col;
            }
        }
        weightValues.AsSpan().CopyTo(stackedWeight.AsSpan<float>());
        biasValues.AsSpan().CopyTo(stackedBias.AsSpan<float>());

        (Tensor weight, Tensor bias) = StackedOutputHeadWeights.Combine(backend, stackedWeight, stackedBias, OutRows, step);
        using (weight)
        using (bias)
        {
            // Span [1, 3): block 0 at weight 1, blocks 1 and 2 at dt[1] / S and dt[2] / S with S = dt[0] + dt[1] + dt[2]
            // in the offset total, which here is dt[1] + dt[2] because the span starts at block 1.
            double dt1 = grid[1] - grid[2], dt2 = grid[2] - grid[3];
            double total = dt1 + dt2;
            double w1 = dt1 / total, w2 = dt2 / total;
            for (int row = 0; row < OutRows; row++)
            {
                for (int col = 0; col < InDim; col++)
                {
                    double expected = weightValues[row * InDim + col]
                        + w1 * weightValues[(OutRows + row) * InDim + col]
                        + w2 * weightValues[(2 * OutRows + row) * InDim + col];
                    Assert.Equal(expected, weight.AsSpan<float>()[row * InDim + col], 5);
                }
                double expectedBias = biasValues[row] + w1 * biasValues[OutRows + row] + w2 * biasValues[2 * OutRows + row];
                Assert.Equal(expectedBias, bias.AsSpan<float>()[row], 5);
            }
        }
    }

    [Fact]
    public void StackedHeadCombineRefusesMismatchedRows()
    {
        using IBackend backend = new CpuBackend();
        using Tensor stackedWeight = new(new TensorShape(7, 3), DType.F32);
        using Tensor stackedBias = new(new TensorShape(7), DType.F32);
        OutputHeadStep step = new StackedOutputHeads([1.0, 0.5, 0.0]).Resolve(1.0, 0.0);

        Assert.Throws<ArgumentException>(() => StackedOutputHeadWeights.Combine(backend, stackedWeight, stackedBias, 2, step));
    }

    [Fact]
    public void ImageSigmasKeepTheCheckpointGridUnshifted()
    {
        double[] grid = [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0];

        Assert.Equal(grid, MiniMaxH3Schedule.ImageSigmas(grid));
    }

    [Fact]
    public void ImageSigmasRefuseAGridThatDoesNotEndAtZero()
    {
        Assert.Throws<ArgumentException>(() => MiniMaxH3Schedule.ImageSigmas([1.0, 0.5]));
    }

    private static Dictionary<string, Tensor> VideoOnlyWeights() => new()
    {
        ["video_patch_proj.weight"] = new Tensor(new TensorShape(8, 96), DType.F32),
        ["final_layer.video_out.weight"] = new Tensor(new TensorShape(576, 8), DType.F32),
    };
}
