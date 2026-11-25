using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines.LongCat;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>A guided video run projects the positive and the negative prompt once each, not once per pass: the pipeline keeps the prompt tensors alive
/// for the run, so the identity-keyed text projection cache hits on every later pass.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatTextProjectionPipelineTests
{
    private const int Frames = 3, Height = 4, Width = 4, TextLength = 6, Steps = 3;

    [Fact]
    public void AGuidedRunProjectsEachPromptOnce()
    {
        LongCatConfig config = LongCatTestData.Tiny;
        Dictionary<string, Tensor> weights = LongCatTestData.Weights(config, seed: 83);
        try
        {
            using LongCatTransformer transformer = new(config);
            transformer.LoadWeights(weights);
            using CpuBackend backend = new();
            Random rng = new(9);
            LongCatPrompt positive = new(Values(rng, TextLength * config.CaptionChannels), TextLength, TextLength);
            LongCatPrompt negative = new(Values(rng, TextLength * config.CaptionChannels), TextLength, TextLength);
            LongCatRun run = new(config.InChannels, Frames, Height, Width, Values(rng, config.InChannels * Frames * Height * Width), positive,
                negative, Steps, Distill: false, Shift: 3f, Guidance: 4f);

            new LongCatVideoPipeline(backend, transformer).Denoise(run);

            // Two guidance passes per step; the prompt tensors are shared, so only the first pass of each prompt projects.
            Assert.Equal(2, transformer.TextProjectionBuilds);
        }
        finally
        {
            LongCatTestData.DisposeAll(weights);
        }
    }

    private static float[] Values(Random rng, int count)
    {
        float[] values = new float[count];
        for (int i = 0; i < count; i++)
            values[i] = (float)(rng.NextDouble() * 2 - 1);
        return values;
    }
}
