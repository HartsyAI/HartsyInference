using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines.LongCat;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The LongCat pipelines' schedule math and their invariants on the tiny random transformer: the conditioned frames are written and
/// kept, the continuation chain hands each segment's context on, and identical prompts make the guidance a no-op. The parity against the
/// reference's own pipeline loops is <c>LongCatPipelineParityTests</c>.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatPipelineTests
{
    private const int Channels = 4, Height = 4, Width = 4, Text = 5;

    [Fact]
    public void LinearScheduleIsTheShiftedRampWithATerminalZero()
    {
        LongCatSchedule schedule = LongCatSchedule.Build(4, distill: false, shift: 1f);

        // torch.linspace(1, 0.001, 4): 1, 0.667, 0.334, 0.001 (step -0.333), then the terminal zero.
        Assert.Equal(new[] { 1f, 0.667f, 0.334f, 0.001f, 0f }, schedule.Sigmas, new FloatComparer(1e-5f));
        Assert.Equal(4, schedule.Steps);
        Assert.Equal(1000f, schedule.Timesteps[0], 3);
    }

    [Fact]
    public void ShiftAndDistilledIndicesMatchTheReference()
    {
        // The reference, shift 3, four distilled steps: sigmas 1, 0.9048, 0.75, 0.5132, 0.
        LongCatSchedule schedule = LongCatSchedule.Build(4, distill: true, shift: 3f);

        Assert.Equal(new[] { 1f, 0.9048f, 0.75f, 0.5132f, 0f }, schedule.Sigmas, new FloatComparer(1e-4f));
    }

    [Fact]
    public void EnhanceHfKeepsTheStepsAboveFiveHundredAndAddsTheUniformTail()
    {
        // The reference's continuation schedule at shift 3 and four steps: 1000, 857, 601 kept, then 500 down to 50 in ten steps.
        LongCatSchedule enhanced = LongCatSchedule.Build(4, distill: false, shift: 3f).EnhanceHf();

        Assert.Equal(13, enhanced.Steps);
        Assert.Equal(500f, enhanced.Timesteps[3], 3);
        Assert.Equal(50f, enhanced.Timesteps[12], 3);
        Assert.Equal(14, enhanced.Sigmas.Length);
        Assert.Equal(0f, enhanced.Sigmas[^1]);
        Assert.Equal(enhanced.Timesteps[0] / 1000f, enhanced.Sigmas[0], 6);
    }

    [Theory]
    [InlineData(0, 3f)]
    [InlineData(4, 0f)]
    [InlineData(4, float.NaN)]
    public void ScheduleRefusesNoStepsOrANonPositiveShift(int steps, float shift)
    {
        Assert.ThrowsAny<ArgumentException>(() => LongCatSchedule.Build(steps, distill: false, shift));
    }

    [Fact]
    public void ImageConditioningWritesTheFrameAndNeverStepsIt()
    {
        WithTinyPipeline(static (pipeline, config) =>
        {
            float[] cond = Ramp(Channels * Height * Width, 0.3f);
            LongCatRun run = Run(config, LongCatConditioningMode.Image, frames: 3, numCond: 1, cond, Prompt(config, 1), Prompt(config, 2), guidance: 4f);

            LongCatResult result = pipeline.Denoise(run);

            Assert.Equal(cond, Frames(result.Latents, 0, 1), new FloatComparer(0f));
            Assert.All(result.Latents, static v => Assert.True(float.IsFinite(v)));
            Assert.Equal(3, result.Steps.Count);
            Assert.Equal(2 * Height * Width * Channels, result.Steps[0].Velocity.Length);
        });
    }

    [Fact]
    public void ContinuationKeepsItsContextAndChainsTwoSegments()
    {
        WithTinyPipeline(static (pipeline, config) =>
        {
            float[] context = Ramp(Channels * 2 * Height * Width, 0.1f);
            LongCatRun template = Run(config, LongCatConditioningMode.Continuation, frames: 4, numCond: 2, context, Prompt(config, 3),
                Prompt(config, 4), guidance: 4f, enhance: true);
            float[] first = Ramp(Channels * 4 * Height * Width, 0.7f);
            float[] second = Ramp(Channels * 4 * Height * Width, 1.3f);

            IReadOnlyList<float[]> outputs = pipeline.ContinuationChain(template with { Noise = first }, context,
                [first, second], contextFrames: 2);

            Assert.Equal(context, Frames(outputs[0], 0, 2), new FloatComparer(0f));
            Assert.Equal(LongCatVideoPipeline.LastFrames(outputs[0], Channels, 4, Height * Width, 2), Frames(outputs[1], 0, 2),
                new FloatComparer(0f));
            Assert.Equal(outputs[0].Length, outputs[1].Length);
        });
    }

    [Fact]
    public void IdenticalPromptsMakeTheGuidanceANoOp()
    {
        WithTinyPipeline(static (pipeline, config) =>
        {
            LongCatPrompt prompt = Prompt(config, 5);
            float[] noise = Ramp(Channels * 2 * Height * Width, 0.9f);

            float[] guided = pipeline.Denoise(Run(config, LongCatConditioningMode.Text, 2, 0, null, prompt, prompt, guidance: 4f, noise: noise)).Latents;
            float[] unguided = pipeline.Denoise(Run(config, LongCatConditioningMode.Text, 2, 0, null, prompt, prompt, guidance: 1f, noise: noise)).Latents;

            Assert.Equal(unguided, guided, new FloatComparer(1e-5f));
        });
    }

    [Fact]
    public void ARunRefusesAnUnloadedTransformer()
    {
        LongCatConfig config = LongCatTestData.Tiny;
        using LongCatTransformer unloaded = new(config);
        using CpuBackend backend = new();
        LongCatVideoPipeline pipeline = new(backend, unloaded);

        Assert.Throws<InvalidOperationException>(() => pipeline.Denoise(Run(config, LongCatConditioningMode.Text, 2, 0, null, Prompt(config, 6),
            Prompt(config, 7), guidance: 1f)));
    }

    private static void WithTinyPipeline(Action<LongCatVideoPipeline, LongCatConfig> body)
    {
        LongCatConfig config = LongCatTestData.Tiny;
        Dictionary<string, Tensor> weights = LongCatTestData.Weights(config, seed: 17);
        try
        {
            using LongCatTransformer transformer = new(config);
            transformer.LoadWeights(weights);
            using CpuBackend backend = new();
            body(new LongCatVideoPipeline(backend, transformer), config);
        }
        finally
        {
            LongCatTestData.DisposeAll(weights);
        }
    }

    private static LongCatRun Run(LongCatConfig config, LongCatConditioningMode mode, int frames, int numCond, float[]? cond, LongCatPrompt positive,
        LongCatPrompt negative, float guidance, bool enhance = false, float[]? noise = null)
    {
        float[] initial = noise ?? Ramp(Channels * frames * Height * Width, 0.5f);
        return new LongCatRun(config.InChannels, frames, Height, Width, initial, positive, negative, Steps: 3, Distill: false, Shift: 3f,
            Guidance: guidance, Mode: mode, NumCondLatents: numCond, CondLatents: cond, EnhanceHf: enhance);
    }

    private static LongCatPrompt Prompt(LongCatConfig config, int seed)
    {
        float[] embeds = new float[Text * config.CaptionChannels];
        for (int i = 0; i < embeds.Length; i++)
            embeds[i] = 0.05f * (float)Math.Sin(0.37 * (i + seed * 13));
        return new LongCatPrompt(embeds, Text, ValidTokens: Text - 1);
    }

    private static float[] Ramp(int count, float offset)
    {
        float[] values = new float[count];
        for (int i = 0; i < count; i++)
            values[i] = offset + 0.01f * (i % 17) - 0.05f;
        return values;
    }

    /// <summary>The first <paramref name="count"/> frames of a <c>[channels, frames, hw]</c> latent.</summary>
    private static float[] Frames(float[] latent, int start, int count)
    {
        int frames = latent.Length / (Channels * Height * Width);
        int hw = Height * Width;
        List<float> values = [];
        for (int c = 0; c < Channels; c++)
            for (int f = start; f < start + count; f++)
                for (int e = 0; e < hw; e++)
                    values.Add(latent[(c * frames + f) * hw + e]);
        return [.. values];
    }

    private sealed class FloatComparer(float tolerance) : IEqualityComparer<float>
    {
        public bool Equals(float x, float y) => Math.Abs(x - y) <= tolerance;
        public int GetHashCode(float value) => value.GetHashCode();
    }
}
