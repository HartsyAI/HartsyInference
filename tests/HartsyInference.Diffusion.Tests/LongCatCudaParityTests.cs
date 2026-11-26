using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines.LongCat;
using HartsyInference.Tests.Shared;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The LongCat video and avatar denoise loops on CUDA against the CPU backend, on the tiny config: the same weights, the same inputs, the
/// same final latent to within 1e-4 (the CUDA kernels round differently). Env-gated: it runs only when HARTSY_LONGCAT_CUDA_PARITY is 1 and CUDA is
/// available, so the suite stays green everywhere else.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class LongCatCudaParityTests
{
    private const string EnvName = "HARTSY_LONGCAT_CUDA_PARITY";
    private const int Frames = 3, Height = 4, Width = 4, TextLength = 6;
    private const float Tolerance = 1e-4f;
    private static readonly int Hw = Height * Width;

    [LongCatCudaParityFact]
    public void TheVideoLoopOnCudaMatchesTheCpu()
    {
        LongCatConfig config = LongCatTestData.Tiny;
        Dictionary<string, Tensor> weights = LongCatTestData.Weights(config, 111);
        try
        {
            Random rng = new(3);
            LongCatPrompt positive = new(LongCatTestData.RandomValues(rng, TextLength * config.CaptionChannels), TextLength, TextLength);
            LongCatPrompt negative = new(LongCatTestData.RandomValues(rng, TextLength * config.CaptionChannels), TextLength, TextLength);
            float[] noise = LongCatTestData.RandomValues(rng, config.InChannels * Frames * Hw);
            float[] cpu;
            using (CpuBackend cpuBackend = new())
                cpu = RunVideo(cpuBackend, config, weights, noise, positive, negative);
            using CudaBackend cuda = GpuDevices.OpenCuda();
            float[] gpu = RunVideo(cuda, config, weights, noise, positive, negative);

            AssertMovedFrom(cpu, noise, Frames, fromFrame: 0, config.InChannels, Hw, "video");
            AssertClose(cpu, gpu, "video");
        }
        finally
        {
            LongCatTestData.DisposeAll(weights);
        }
    }

    [LongCatCudaParityFact]
    public void TheAvatarCachedLoopOnCudaMatchesTheCpu()
    {
        LongCatConfig config = LongCatTestData.TinyAvatar;
        Dictionary<string, Tensor> weights = LongCatTestData.AvatarWeights(config, 113);
        try
        {
            Random rng = new(5);
            int cond = 2, ctx = config.AudioContextTokens, width = config.AudioOutputDim;
            LongCatPrompt positive = new(LongCatTestData.RandomValues(rng, TextLength * config.CaptionChannels), TextLength, TextLength);
            LongCatPrompt negative = new(LongCatTestData.RandomValues(rng, TextLength * config.CaptionChannels), TextLength, TextLength);
            float[] noise = LongCatTestData.RandomValues(rng, config.InChannels * Frames * Hw);
            float[] condLatents = LongCatTestData.RandomValues(rng, config.InChannels * cond * Hw);
            float[] audio = LongCatTestData.RandomValues(rng, Frames * ctx * width);
            float[] unconditionalAudio = LongCatTestData.RandomValues(rng, Frames * ctx * width);
            float[] cpu;
            using (CpuBackend cpuBackend = new())
                cpu = RunAvatar(cpuBackend, config, weights, noise, condLatents, audio, unconditionalAudio, positive, negative);
            using CudaBackend cuda = GpuDevices.OpenCuda();
            float[] gpu = RunAvatar(cuda, config, weights, noise, condLatents, audio, unconditionalAudio, positive, negative);

            AssertMovedFrom(cpu, noise, Frames, fromFrame: 2, config.InChannels, Hw, "avatar");
            AssertConditioningExact(cpu, condLatents, 2, config.InChannels, Hw, "avatar");
            AssertClose(cpu, gpu, "avatar");
        }
        finally
        {
            LongCatTestData.DisposeAll(weights);
        }
    }

    private static float[] RunVideo(IBackend backend, LongCatConfig config, Dictionary<string, Tensor> weights, float[] noise,
        LongCatPrompt positive, LongCatPrompt negative)
    {
        using LongCatTransformer transformer = new(config);
        transformer.LoadWeights(weights);
        LongCatRun run = new(config.InChannels, Frames, Height, Width, noise, positive, negative, Steps: 3, Distill: false, Shift: 3f, Guidance: 4f);
        return new LongCatVideoPipeline(backend, transformer).Denoise(run).Latents;
    }

    private static float[] RunAvatar(IBackend backend, LongCatConfig config, Dictionary<string, Tensor> weights, float[] noise, float[] condLatents,
        float[] audio, float[] unconditionalAudio, LongCatPrompt positive, LongCatPrompt negative)
    {
        using LongCatTransformer transformer = new(config);
        transformer.LoadWeights(weights);
        LongCatAvatarRun run = new(config.InChannels, Frames, Height, Width, noise, positive, negative, audio, unconditionalAudio, Steps: 3,
            LongCatAvatarSchedule.Kind.Linear, 7f, 2f, 1.5f, LongCatAvatarMode.Avc, 2, condLatents, UseKvCache: true);
        return new LongCatAvatarPipeline(backend, transformer).Denoise(run).Latents;
    }

    private static void AssertClose(float[] cpu, float[] gpu, string label)
    {
        Assert.Equal(cpu.Length, gpu.Length);
        float worst = 0;
        for (int i = 0; i < cpu.Length; i++)
            worst = MathF.Max(worst, MathF.Abs(cpu[i] - gpu[i]));
        Assert.True(worst <= Tolerance, $"{label}: CUDA drifted from the CPU by {worst}");
    }

    /// <summary>The frames from <paramref name="fromFrame"/> on must have moved away from the noise, so the parity compares real work.</summary>
    private static void AssertMovedFrom(float[] result, float[] noise, int frames, int fromFrame, int channels, int hw, string label)
    {
        float moved = 0;
        for (int c = 0; c < channels; c++)
            for (int f = fromFrame; f < frames; f++)
                for (int e = 0; e < hw; e++)
                {
                    int index = (c * frames + f) * hw + e;
                    moved = MathF.Max(moved, MathF.Abs(result[index] - noise[index]));
                }
        Assert.True(moved > 1e-3f, $"{label}: the loop left the noisy frames where they started (moved {moved})");
    }

    /// <summary>The conditioning frames come back exactly as given.</summary>
    private static void AssertConditioningExact(float[] result, float[] condLatents, int cond, int channels, int hw, string label)
    {
        for (int c = 0; c < channels; c++)
            for (int f = 0; f < cond; f++)
                for (int e = 0; e < hw; e++)
                    Assert.Equal(condLatents[(c * cond + f) * hw + e], result[(c * Frames + f) * hw + e]);
    }

    /// <summary>Runs only when the env var is 1 and a CUDA device opens. Otherwise the test is reported as skipped, never passed.</summary>
    private sealed class LongCatCudaParityFactAttribute : FactAttribute
    {
        public LongCatCudaParityFactAttribute()
        {
            if (Environment.GetEnvironmentVariable(EnvName) != "1")
                Skip = $"SKIPPED: {EnvName} is not 1.";
            else if (GpuDevices.CudaFailure is { } failure)
                Skip = $"SKIPPED: no usable CUDA device ({failure}).";
        }
    }
}
