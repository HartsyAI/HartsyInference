using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines.LongCat;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The video denoise loop with its state on the device matches the host arithmetic it replaces: the CFG-zero scale from the same sums, the
/// same combine, and the same Euler step on the same forwards. The reference below is that host loop, run against the transformer directly.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatVideoDeviceLoopTests : IDisposable
{
    private const int Frames = 3, Height = 4, Width = 4, TextLength = 6, Steps = 3;
    private static readonly int Hw = Height * Width;
    private readonly CpuBackend _backend = new();
    private readonly LongCatConfig _config = LongCatTestData.Tiny;
    private readonly Dictionary<string, Tensor> _weights;

    public LongCatVideoDeviceLoopTests()
    {
        _weights = LongCatTestData.Weights(_config, 91);
    }

    public void Dispose()
    {
        LongCatTestData.DisposeAll(_weights);
        _backend.Dispose();
    }

    [Theory]
    [InlineData(LongCatConditioningMode.Text, 4f)]
    [InlineData(LongCatConditioningMode.Text, 1f)]
    [InlineData(LongCatConditioningMode.Image, 4f)]
    [InlineData(LongCatConditioningMode.Continuation, 4f)]
    public void TheDeviceLoopMatchesTheHostLoopWithinAMicroFloat(LongCatConditioningMode mode, float guidance)
    {
        int cond = mode == LongCatConditioningMode.Text ? 0 : 1;
        Random rng = new(13);
        float[] noise = RandomArray(rng, _config.InChannels * Frames * Hw);
        float[]? condLatents = cond == 0 ? null : RandomArray(rng, _config.InChannels * cond * Hw);
        LongCatPrompt positive = new(RandomArray(rng, TextLength * _config.CaptionChannels), TextLength, TextLength);
        LongCatPrompt negative = new(RandomArray(rng, TextLength * _config.CaptionChannels), TextLength, TextLength);

        using LongCatTransformer transformer = new(_config);
        transformer.LoadWeights(_weights);
        LongCatRun run = new(_config.InChannels, Frames, Height, Width, noise, positive, negative, Steps, Distill: false, Shift: 3f,
            Guidance: guidance, Mode: mode, NumCondLatents: cond, CondLatents: condLatents);

        float[] device = new LongCatVideoPipeline(_backend, transformer).Denoise(run).Latents;
        float[] host = HostReference(transformer, run, cond);

        Assert.Equal(host.Length, device.Length);
        double worst = 0;
        for (int i = 0; i < host.Length; i++)
            worst = Math.Max(worst, Math.Abs(host[i] - device[i]));
        Assert.True(worst <= 1e-5, $"device loop drifted from the host loop by {worst}");
    }

    /// <summary>The host loop the device version replaces: the same forwards, the CFG-zero combine and the Euler step on host arrays.</summary>
    private float[] HostReference(LongCatTransformer transformer, LongCatRun run, int cond)
    {
        int channels = run.Channels, frames = run.Frames, hw = run.Height * run.Width, stepped = frames - cond;
        LongCatSchedule schedule = LongCatSchedule.Build(run.Steps, run.Distill, run.Shift);
        LongCatConditioning conditioning = new(cond, 0, -1, 0);
        int guidedFrom = run.Mode == LongCatConditioningMode.Continuation ? cond : 0;
        bool guided = run.Guidance > 1f;

        float[] latents = (float[])run.Noise.Clone();
        for (int c = 0; c < channels && cond > 0; c++)
            Array.Copy(run.CondLatents!, c * cond * hw, latents, c * frames * hw, cond * hw);

        for (int i = 0; i < schedule.Steps; i++)
        {
            float timestep = schedule.Timesteps[i], dt = schedule.Sigmas[i + 1] - schedule.Sigmas[i];
            float[] frameTimesteps = conditioning.FrameTimesteps(timestep, frames);
            float[] velocity = Forward(transformer, run, latents, run.Positive, frameTimesteps, conditioning);
            if (guided)
            {
                float[] negative = Forward(transformer, run, latents, run.Negative, frameTimesteps, conditioning);
                double dot = 0, squared = 0;
                for (int c = 0; c < channels; c++)
                    for (int f = guidedFrom; f < frames; f++)
                        for (int e = 0; e < hw; e++)
                        {
                            int index = (c * frames + f) * hw + e;
                            dot += velocity[index] * negative[index];
                            squared += negative[index] * negative[index];
                        }
                float scale = (float)(dot / (squared + 1e-8));
                for (int c = 0; c < channels; c++)
                    for (int f = guidedFrom; f < frames; f++)
                        for (int e = 0; e < hw; e++)
                        {
                            int index = (c * frames + f) * hw + e;
                            float uncond = negative[index] * scale;
                            velocity[index] = uncond + run.Guidance * (velocity[index] - uncond);
                        }
            }
            for (int c = 0; c < channels; c++)
                for (int f = cond; f < frames; f++)
                    for (int e = 0; e < hw; e++)
                    {
                        int index = (c * frames + f) * hw + e;
                        latents[index] += dt * -velocity[index];
                    }
        }
        return latents;
    }

    private float[] Forward(LongCatTransformer transformer, LongCatRun run, float[] latents, LongCatPrompt prompt, float[] frameTimesteps,
        LongCatConditioning conditioning)
    {
        using Tensor input = Shaped((float[])latents.Clone(), [1, run.Channels, run.Frames, run.Height, run.Width]);
        using Tensor text = Shaped((float[])prompt.Embeds.Clone(), [prompt.Length, _config.CaptionChannels]);
        using Tensor output = transformer.Forward(_backend, input, text, frameTimesteps, in conditioning, prompt.ValidTokens);
        return output.AsReadOnlySpan<float>().ToArray();
    }

    private Tensor Shaped(float[] data, int[] dims)
    {
        Tensor tensor = new(new TensorShape(dims.Select(d => (long)d).ToArray()), DType.F32);
        data.AsSpan().CopyTo(tensor.AsSpan<float>());
        return tensor;
    }

    private static float[] RandomArray(Random rng, int count)
    {
        float[] values = new float[count];
        for (int i = 0; i < count; i++)
            values[i] = (float)(rng.NextDouble() * 2 - 1);
        return values;
    }
}
