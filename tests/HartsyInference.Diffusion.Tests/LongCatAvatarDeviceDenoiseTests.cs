using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines.LongCat;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The avatar denoise loop with its state on the device matches the host arithmetic it replaces: the same guidance combine and Euler step on
/// the same velocities, read back from the same forwards. The reference below is that host loop, run against the transformer directly.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatAvatarDeviceDenoiseTests : IDisposable
{
    private const int Frames = 5, Height = 4, Width = 4, TextLength = 6, Steps = 3;
    private static readonly int Hw = Height * Width;
    private readonly CpuBackend _backend = new();
    private readonly LongCatConfig _config = LongCatTestData.TinyAvatar;
    private readonly Dictionary<string, Tensor> _weights;

    public LongCatAvatarDeviceDenoiseTests()
    {
        _weights = LongCatTestData.AvatarWeights(_config, 71);
    }

    public void Dispose()
    {
        LongCatTestData.DisposeAll(_weights);
        _backend.Dispose();
    }

    [Theory]
    [InlineData(LongCatAvatarMode.At2v, false, 2f, 1.5f)]
    [InlineData(LongCatAvatarMode.Ai2v, false, 1f, 1f)]
    [InlineData(LongCatAvatarMode.Ai2v, false, 2f, 1.5f)]
    [InlineData(LongCatAvatarMode.Avc, true, 2f, 1.5f)]
    public void TheDeviceLoopMatchesTheHostLoopWithinAMicroFloat(LongCatAvatarMode mode, bool cached, float textGuidance, float audioGuidance)
    {
        int cond = mode switch { LongCatAvatarMode.At2v => 0, LongCatAvatarMode.Avc => 2, _ => 1 };
        Random rng = new(5);
        float[] noise = RandomArray(rng, _config.InChannels * Frames * Hw);
        float[]? condLatents = cond == 0 ? null : RandomArray(rng, _config.InChannels * cond * Hw);
        float[] audio = RandomArray(rng, Frames * _config.AudioContextTokens * _config.AudioOutputDim);
        float[] unconditionalAudio = RandomArray(rng, audio.Length);
        LongCatPrompt positive = new(RandomArray(rng, TextLength * _config.CaptionChannels), TextLength, TextLength);
        LongCatPrompt negative = new(RandomArray(rng, TextLength * _config.CaptionChannels), TextLength, TextLength);

        using LongCatTransformer transformer = new(_config);
        transformer.LoadWeights(_weights);
        LongCatAvatarRun run = new(_config.InChannels, Frames, Height, Width, noise, positive, negative, audio, unconditionalAudio, Steps,
            LongCatAvatarSchedule.Kind.Linear, 7f, textGuidance, audioGuidance, mode, cond, condLatents, UseKvCache: cached);

        float[] device = new LongCatAvatarPipeline(_backend, transformer).Denoise(run).Latents;
        float[] host = HostReference(transformer, run, cond, cached);

        Assert.Equal(host.Length, device.Length);
        double worst = 0;
        for (int i = 0; i < host.Length; i++)
            worst = Math.Max(worst, Math.Abs(host[i] - device[i]));
        Assert.True(worst <= 1e-5, $"device loop drifted from the host loop by {worst}");
        // The conditioning frames come back exactly as given, and the noisy frames moved, so the parity is not vacuous.
        int hw = Hw;
        for (int c = 0; c < _config.InChannels; c++)
            for (int f = 0; f < cond; f++)
                for (int e = 0; e < hw; e++)
                    Assert.Equal(condLatents![(c * cond + f) * hw + e], device[(c * Frames + f) * hw + e]);
        double moved = 0;
        for (int c = 0; c < _config.InChannels; c++)
            for (int f = cond; f < Frames; f++)
                for (int e = 0; e < hw; e++)
                    moved = Math.Max(moved, Math.Abs(device[(c * Frames + f) * hw + e] - run.Noise[(c * Frames + f) * hw + e]));
        Assert.True(moved > 1e-3, $"the denoise loop left the noisy frames where they started (moved {moved})");
    }

    /// <summary>The host loop the device version replaces: the same forwards, the three-pass combine and the Euler step on host arrays.</summary>
    private float[] HostReference(LongCatTransformer transformer, LongCatAvatarRun run, int cond, bool cached)
    {
        int channels = run.Channels, frames = run.Frames, hw = run.Height * run.Width, noisy = frames - cond, ctx = _config.AudioContextTokens;
        bool guided = run.TextGuidance > 1f || run.AudioGuidance > 1f;
        float[] sigmas = Shifted(LongCatAvatarSchedule.Sigmas(run.Steps, run.Kind), run.Shift);
        float[] sigmasWithZero = [.. sigmas, 0f];
        LongCatConditioning conditioning = run.Mode switch
        {
            LongCatAvatarMode.At2v => new LongCatConditioning(),
            LongCatAvatarMode.Avc => LongCatConditioning.Avatar(cond, run.RefImgIndex),
            _ => new LongCatConditioning(NumCondLatents: 1),
        };
        float[] latents = (float[])run.Noise.Clone();
        for (int c = 0; c < channels && cond > 0; c++)
            Array.Copy(run.CondLatents!, c * cond * hw, latents, c * frames * hw, cond * hw);

        using Tensor condClip = cached ? Shaped(Frame(latents, channels, frames, hw, 0, cond), [1, channels, cond, run.Height, run.Width]) : null!;
        using LongCatKvCache? cache = cached ? transformer.BuildKvCache(_backend, condClip, conditioning) : null;
        for (int i = 0; i < run.Steps; i++)
        {
            float t = sigmas[i] * LongCatAvatarSchedule.NumTimesteps, dt = sigmasWithZero[i + 1] - sigmasWithZero[i];
            float[] cond0 = Pass(transformer, run, latents, true, true, cached, cond, conditioning, cache, t);
            float[] velocity = cond0;
            if (guided)
            {
                float[] uncondText = Pass(transformer, run, latents, false, true, cached, cond, conditioning, cache, t);
                float[] uncond = Pass(transformer, run, latents, false, false, cached, cond, conditioning, cache, t);
                velocity = new float[cond0.Length];
                for (int k = 0; k < velocity.Length; k++)
                    velocity[k] = uncond[k] + run.TextGuidance * (cond0[k] - uncondText[k]) + run.AudioGuidance * (uncondText[k] - uncond[k]);
            }
            for (int c = 0; c < channels; c++)
                for (int f = 0; f < noisy; f++)
                    for (int e = 0; e < hw; e++)
                        latents[(c * frames + cond + f) * hw + e] += dt * -velocity[(c * noisy + f) * hw + e];
        }
        return latents;
    }

    /// <summary>One forward pass of the host loop: the noisy velocity rows, <c>[channels·noisy, hw]</c>.</summary>
    private float[] Pass(LongCatTransformer transformer, LongCatAvatarRun run, float[] latents, bool positive, bool audioOn, bool cached, int cond,
        LongCatConditioning conditioning, LongCatKvCache? cache, float t)
    {
        int channels = run.Channels, frames = run.Frames, hw = run.Height * run.Width, noisy = frames - cond, ctx = _config.AudioContextTokens;
        LongCatPrompt prompt = positive ? run.Positive : run.Negative;
        float[] audioSource = audioOn ? run.AudioTokens : run.UnconditionalAudioTokens!;
        int audioFrom = cached ? cond : 0, audioCount = cached ? noisy : frames;
        float[] audio = new float[audioCount * ctx * _config.AudioOutputDim];
        Array.Copy(audioSource, audioFrom * ctx * _config.AudioOutputDim, audio, 0, audio.Length);
        using Tensor text = Shaped(prompt.Embeds, [prompt.Length, _config.CaptionChannels]);
        using Tensor audioTokens = Shaped(audio, [audioCount * ctx, _config.AudioOutputDim]);

        if (cached)
        {
            float[] noisyFrames = Frame(latents, channels, frames, hw, cond, noisy);
            using Tensor input = Shaped(noisyFrames, [1, channels, noisy, run.Height, run.Width]);
            using Tensor output = transformer.ForwardCached(_backend, input, text, [t], in conditioning, cache!, prompt.ValidTokens, audioTokens);
            return output.AsReadOnlySpan<float>().ToArray();
        }
        using Tensor full = Shaped((float[])latents.Clone(), [1, channels, frames, run.Height, run.Width]);
        using Tensor result = transformer.Forward(_backend, full, text, conditioning.FrameTimesteps(t, frames), in conditioning, prompt.ValidTokens,
            audioTokens);
        float[] all = result.AsReadOnlySpan<float>().ToArray();
        float[] rows = new float[channels * noisy * hw];
        for (int c = 0; c < channels; c++)
            Array.Copy(all, (c * frames + cond) * hw, rows, c * noisy * hw, noisy * hw);
        return rows;
    }

    private static float[] Frame(float[] latents, int channels, int frames, int hw, int from, int count)
    {
        float[] result = new float[channels * count * hw];
        for (int c = 0; c < channels; c++)
            Array.Copy(latents, (c * frames + from) * hw, result, c * count * hw, count * hw);
        return result;
    }

    private Tensor Shaped(float[] data, int[] dims)
    {
        Tensor tensor = new(new TensorShape(dims.Select(d => (long)d).ToArray()), DType.F32);
        data.AsSpan().CopyTo(tensor.AsSpan<float>());
        return tensor;
    }

    private static float[] Shifted(float[] sigmas, float shift)
    {
        float[] result = new float[sigmas.Length];
        for (int i = 0; i < sigmas.Length; i++)
            result[i] = shift * sigmas[i] / (1f + (shift - 1f) * sigmas[i]);
        return result;
    }

    /// <summary>The tiny config's base weights plus the avatar's audio projection and each block's audio step, with seeded values.</summary>
    private static float[] RandomArray(Random rng, int count)
    {
        float[] values = new float[count];
        for (int i = 0; i < count; i++)
            values[i] = (float)(rng.NextDouble() * 2 - 1);
        return values;
    }
}
