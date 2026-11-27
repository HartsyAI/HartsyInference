using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines.LongCat;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The LongCat 720p refinement on the tiny random transformer: its schedule, the block-sparse self-attention wiring
/// against dense attention, and the refinement run's invariants. Tiny blocks of 2×2×2 latent tokens let a 4×4 token grid select
/// some but not all key blocks. Parity against the reference's own pipeline is <c>LongCatRefinementParityTests</c>.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatRefinementTests
{
    private const int Frames = 6, Cond = 2, Height = 8, Width = 8, Text = 5;

    [Fact]
    public void TheScheduleIsFiveStepsFromTheThresholdDown()
    {
        LongCatSchedule schedule = LongCatSchedule.Refinement(50, shift: 12f, threshold: 0.5f);

        float[] expected = [500f, 443.0f, 343.5f, 207.8f, 11.9f];
        Assert.Equal(expected.Length, schedule.Steps);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.InRange(schedule.Timesteps[i], expected[i] - 0.05f, expected[i] + 0.05f);
            Assert.Equal(schedule.Timesteps[i] / 1000f, schedule.Sigmas[i]);
        }
        Assert.Equal(0f, schedule.Sigmas[^1]);
    }

    [Theory]
    [InlineData(0f)]
    [InlineData(1.5f)]
    [InlineData(float.NaN)]
    public void TheScheduleRefusesAThresholdOutsideItsRange(float threshold) =>
        Assert.Throws<ArgumentOutOfRangeException>(() => LongCatSchedule.Refinement(50, 12f, threshold));

    [Theory]
    [InlineData(0)]
    [InlineData(Cond)]
    public void ReadingEveryKeyBlockIsDenseSelfAttention(int cond)
    {
        WithTiny(sparsity: 0.0, (transformer, backend, config) =>
        {
            float[] dense = Forward(transformer, backend, config, cond, blockSparse: false);
            float[] sparse = Forward(transformer, backend, config, cond, blockSparse: true);

            Assert.True(LongCatTestData.RelRms(sparse, dense) <= 1e-5, $"relRMS {LongCatTestData.RelRms(sparse, dense):E3}");
        });
    }

    [Theory]
    [InlineData(0)]
    [InlineData(Cond)]
    public void ReadingHalfTheKeyBlocksChangesTheResultAndHeadGroupsDoNot(int cond)
    {
        WithTiny(sparsity: 0.5, (transformer, backend, config) =>
        {
            int noisy = Frames - cond;
            BlockSparseAttentionOptions options = config.SparseAttention.Options(noisy, Frames, Height / 2, Width / 2);
            Assert.InRange(options.SelectedBlocks, 1, options.KeyBlocks - 1);
            float[] dense = Forward(transformer, backend, config, cond, blockSparse: false);
            float[] together = Forward(transformer, backend, config, cond, blockSparse: true);
            Assert.True(LongCatTestData.RelRms(together, dense) > 1e-3, "The selection left the result dense.");

            foreach (int group in (int[])[1, 3])
            {
                transformer.SetSparseHeadGroup(group);
                float[] grouped = Forward(transformer, backend, config, cond, blockSparse: true);
                Assert.True(LongCatTestData.RelRms(grouped, together) <= 1e-6, $"heads in groups of {group}");
            }
        });
    }

    [Fact]
    public void AOneFrameClipStaysDense()
    {
        WithTiny(sparsity: 0.5, (transformer, backend, config) =>
        {
            float[] dense = Forward(transformer, backend, config, cond: 0, blockSparse: false, frames: 1);

            Assert.Equal(dense, Forward(transformer, backend, config, cond: 0, blockSparse: true, frames: 1));
        });
    }

    [Theory]
    [InlineData(Frames, 1, 8, "query token grids")]
    [InlineData(5, 0, 8, "query token grids")]
    [InlineData(Frames, 0, 10, "multiples of 2x2x2")]
    public void AGridThatDoesNotTileIntoBlocksIsRefused(int frames, int cond, int height, string named)
    {
        WithTiny(sparsity: 0.5, (transformer, backend, config) =>
        {
            ArgumentException refused = Assert.Throws<ArgumentException>(
                () => Forward(transformer, backend, config, cond, blockSparse: true, frames, height));

            Assert.Contains(named, refused.Message, StringComparison.Ordinal);
        });
    }

    [Fact]
    public void ARefinementStartsAtTheMixedLatentKeepsItsConditionAndStepsFiveTimes()
    {
        WithTiny(sparsity: 0.5, (transformer, backend, config) =>
        {
            LongCatVideoPipeline pipeline = new(backend, transformer);
            LongCatRefineRun run = RefineRun(config, Cond);

            LongCatResult result = pipeline.Refine(run);

            Assert.Equal(5, result.Steps.Count);
            int hw = Height * Width;
            for (int c = 0; c < config.InChannels; c++)
                for (int e = 0; e < Cond * hw; e++)
                    Assert.Equal(run.CondLatents![c * Cond * hw + e], result.Latents[c * Frames * hw + e]);
            // The first step's sample is the mixed latent plus its step: x₁ = x₀ + (σ₁ − σ₀)·v, v the recorded (negated) velocity.
            float[] first = result.Steps[0].SampleOut;
            float dt = result.Schedule.Sigmas[1] - result.Schedule.Sigmas[0];
            int noisy = Frames - Cond;
            for (int c = 0; c < config.InChannels; c++)
                for (int e = 0; e < noisy * hw; e++)
                {
                    int at = c * Frames * hw + Cond * hw + e, row = c * noisy * hw + e;
                    float start = 0.5f * run.Latent[at] + 0.5f * run.Noise[at];
                    Assert.InRange(first[row] - (start - dt * -result.Steps[0].Velocity[row]), -1e-5f, 1e-5f);
                }
        });
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    public void ARefinementRefusesAConditionThatDoesNotTileIntoBlocks(int cond)
    {
        WithTiny(sparsity: 0.5, (transformer, backend, config) =>
        {
            LongCatVideoPipeline pipeline = new(backend, transformer);

            Assert.Throws<ArgumentException>(() => pipeline.Refine(RefineRun(config, cond)));
        });
    }

    [Fact]
    public void TheConfigReadsTheBlockSparseParametersAndRefusesOnesItDoesNotRun()
    {
        LongCatConfig read = LongCatConfig.FromJson(
            """{"bsa_params":{"sparsity":0.5,"chunk_3d_shape_q":[2,4,4],"chunk_3d_shape_k":[4,2,2]}}""");

        Assert.Equal(new LongCatSparseAttention(0.5, new TokenGrid3D(2, 4, 4), new TokenGrid3D(4, 2, 2)), read.SparseAttention);
        Assert.Equal(LongCatSparseAttention.Released, LongCatConfig.FromJson("{}").SparseAttention);
        Assert.Throws<InvalidDataException>(() => LongCatConfig.FromJson("""{"bsa_params":{"cdf_threshold":0.9}}"""));
        Assert.Throws<InvalidDataException>(() => LongCatConfig.FromJson("""{"bsa_params":{"chunk_3d_shape_q":[4,4]}}"""));
        Assert.Throws<InvalidDataException>(() => LongCatConfig.FromJson("""{"bsa_params":{"sparsity":2}}"""));
    }

    private static void WithTiny(double sparsity, Action<LongCatTransformer, CpuBackend, LongCatConfig> body)
    {
        LongCatConfig config = LongCatTestData.Tiny with
        {
            SparseAttention = new LongCatSparseAttention(sparsity, new TokenGrid3D(2, 2, 2), new TokenGrid3D(2, 2, 2)),
        };
        Dictionary<string, Tensor> weights = LongCatTestData.Weights(config, seed: 23);
        try
        {
            using LongCatTransformer transformer = new(config);
            transformer.LoadWeights(weights);
            using CpuBackend backend = new();
            body(transformer, backend, config);
        }
        finally
        {
            LongCatTestData.DisposeAll(weights);
        }
    }

    private static float[] Forward(LongCatTransformer transformer, CpuBackend backend, LongCatConfig config, int cond, bool blockSparse,
        int frames = Frames, int height = Height)
    {
        using Tensor latent = LongCatTestData.Latent(config, frames, height, Width, seed: 31);
        using Tensor text = LongCatTestData.Encoder(config, Text, seed: 37);
        LongCatConditioning conditioning = new(cond, 0, -1, 0);
        using Tensor output = transformer.Forward(backend, latent, text, conditioning.FrameTimesteps(500f, frames), in conditioning,
            blockSparse: blockSparse);
        return output.AsReadOnlySpan<float>().ToArray();
    }

    private static LongCatRefineRun RefineRun(LongCatConfig config, int cond)
    {
        int values = config.InChannels * Frames * Height * Width;
        Random random = new(41);
        float[] embeds = LongCatTestData.RandomValues(random, Text * config.CaptionChannels);
        return new LongCatRefineRun(config.InChannels, Frames, Height, Width, LongCatTestData.RandomValues(random, values),
            LongCatTestData.RandomValues(random, values), new LongCatPrompt(embeds, Text, Text), cond,
            LongCatTestData.RandomValues(random, config.InChannels * cond * Height * Width), RecordTrace: true);
    }
}
