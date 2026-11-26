using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines.LongCat;
using HartsyInference.Tests.Shared;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The LongCat video and avatar denoise loops on CUDA against the CPU backend, on the tiny config: the same weights, the same inputs, and
/// the same final latent, in two lanes. The exact lane runs under <see cref="ExactGemmScope"/> (plain F32 GEMMs, no TF32) and must agree to
/// <see cref="ExactBound"/>; that lane is the proof that the loops compute the same thing. The default lane runs the engine's default GEMMs (TF32)
/// and must agree to <see cref="DefaultBound"/>, the bound the TF32 error is measured against. Each lane prints its worst absolute difference.
/// Env-gated: it runs only when HARTSY_LONGCAT_CUDA_PARITY is 1 and CUDA is available, so the suite stays green everywhere else.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class LongCatCudaParityTests(ITestOutputHelper output)
{
    private const string EnvName = "HARTSY_LONGCAT_CUDA_PARITY";
    private const int Frames = 3, Height = 4, Width = 4, TextLength = 6;

    /// <summary>The exact lane's bound: plain F32 on both sides, so only the kernels' reduction order differs. Measured on a 4090 at d7cda1f25
    /// (ExactGemmScope on, the gate on): the video worst absolute difference was 2.384e-6 and the avatar's 1.188e-6, both well inside 1e-5.</summary>
    private const float ExactBound = 1e-5f;

    /// <summary>The default lane's bound. Measured on a 4090 at 7dd3c077c (default TF32 GEMMs, the gate on): the video worst absolute difference
    /// was 2.961e-3 and the avatar's 1.633e-3. The bound sits about 3.4 times above the larger of the two.</summary>
    private const float DefaultBound = 1e-2f;

    private static readonly int Hw = Height * Width;

    [LongCatCudaParityFact]
    public void TheVideoLoopWithExactGemmsMatchesTheCpu() => VideoParity(exactGemm: true, ExactBound, "video exact");

    [LongCatCudaParityFact]
    public void TheVideoLoopWithDefaultGemmsStaysWithinTheTf32Bound() => VideoParity(exactGemm: false, DefaultBound, "video default");

    [LongCatCudaParityFact]
    public void TheAvatarCachedLoopWithExactGemmsMatchesTheCpu() => AvatarParity(exactGemm: true, ExactBound, "avatar exact");

    [LongCatCudaParityFact]
    public void TheAvatarCachedLoopWithDefaultGemmsStaysWithinTheTf32Bound() => AvatarParity(exactGemm: false, DefaultBound, "avatar default");

    private void VideoParity(bool exactGemm, float bound, string label)
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
            using IDisposable? scope = exactGemm ? new ExactGemmScope(cuda) : null;
            float[] gpu = RunVideo(cuda, config, weights, noise, positive, negative);

            AssertMovedFrom(cpu, noise, Frames, fromFrame: 0, config.InChannels, Hw, label);
            AssertClose(cpu, gpu, bound, label);
        }
        finally
        {
            LongCatTestData.DisposeAll(weights);
        }
    }

    private void AvatarParity(bool exactGemm, float bound, string label)
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
            using IDisposable? scope = exactGemm ? new ExactGemmScope(cuda) : null;
            float[] gpu = RunAvatar(cuda, config, weights, noise, condLatents, audio, unconditionalAudio, positive, negative);

            AssertMovedFrom(cpu, noise, Frames, fromFrame: 2, config.InChannels, Hw, label);
            AssertConditioningExact(cpu, condLatents, 2, config.InChannels, Hw, label);
            AssertClose(cpu, gpu, bound, label);
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

    /// <summary>Prints the worst absolute difference of the lane, then asserts it is within <paramref name="bound"/>.</summary>
    private void AssertClose(float[] cpu, float[] gpu, float bound, string label)
    {
        Assert.Equal(cpu.Length, gpu.Length);
        float worst = 0;
        for (int i = 0; i < cpu.Length; i++)
            worst = MathF.Max(worst, MathF.Abs(cpu[i] - gpu[i]));
        output.WriteLine($"{label}: worst absolute difference {worst:E3} (bound {bound:E0})");
        Assert.True(worst <= bound, $"{label}: CUDA drifted from the CPU by {worst:E3}, above the bound {bound:E0}");
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
