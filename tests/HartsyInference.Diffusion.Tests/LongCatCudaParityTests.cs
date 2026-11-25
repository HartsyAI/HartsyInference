using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines.LongCat;
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

    [Fact]
    public void TheVideoLoopOnCudaMatchesTheCpu()
    {
        if (!Enabled())
            return;
        LongCatConfig config = LongCatTestData.Tiny;
        Dictionary<string, Tensor> weights = LongCatTestData.Weights(config, 111);
        try
        {
            Random rng = new(3);
            LongCatPrompt positive = new(Values(rng, TextLength * config.CaptionChannels), TextLength, TextLength);
            LongCatPrompt negative = new(Values(rng, TextLength * config.CaptionChannels), TextLength, TextLength);
            float[] noise = Values(rng, config.InChannels * Frames * Hw);
            float[] cpu = RunVideo(new CpuBackend(), config, weights, noise, positive, negative);
            using CudaBackend cuda = new(0, PtxDirectory());
            float[] gpu = RunVideo(cuda, config, weights, noise, positive, negative);

            AssertClose(cpu, gpu, "video");
        }
        finally
        {
            LongCatTestData.DisposeAll(weights);
        }
    }

    [Fact]
    public void TheAvatarCachedLoopOnCudaMatchesTheCpu()
    {
        if (!Enabled())
            return;
        LongCatConfig config = AvatarConfig();
        Dictionary<string, Tensor> weights = AvatarWeights(config, 113);
        try
        {
            Random rng = new(5);
            int cond = 2, ctx = config.AudioContextTokens, width = config.AudioOutputDim;
            LongCatPrompt positive = new(Values(rng, TextLength * config.CaptionChannels), TextLength, TextLength);
            LongCatPrompt negative = new(Values(rng, TextLength * config.CaptionChannels), TextLength, TextLength);
            float[] noise = Values(rng, config.InChannels * Frames * Hw);
            float[] condLatents = Values(rng, config.InChannels * cond * Hw);
            float[] audio = Values(rng, Frames * ctx * width);
            float[] unconditionalAudio = Values(rng, Frames * ctx * width);
            float[] cpu = RunAvatar(new CpuBackend(), config, weights, noise, condLatents, audio, unconditionalAudio, positive, negative);
            using CudaBackend cuda = new(0, PtxDirectory());
            float[] gpu = RunAvatar(cuda, config, weights, noise, condLatents, audio, unconditionalAudio, positive, negative);

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

    private static bool Enabled() => Environment.GetEnvironmentVariable(EnvName) == "1" && CudaContext.IsAvailable();

    /// <summary>The tiny avatar config: the base tiny shape plus the audio projection and the per-block audio step.</summary>
    private static LongCatConfig AvatarConfig() => new()
    {
        HiddenSize = 128, Depth = 2, NumHeads = 4, InChannels = 4, OutChannels = 4, CaptionChannels = 48, AdalnTembedDim = 64,
        FrequencyEmbeddingSize = 64, PatchSize = (1, 2, 2), TextTokensZeroPad = true, AudioOutputDim = 16, AudioWindow = 5, AudioBlock = 3,
        AudioChannel = 6, AudioIntermediateDim = 16, AudioContextTokens = 4, AudioVaeScale = 4,
    };

    /// <summary>The tiny config's base weights plus the audio projection and each block's audio step, with seeded values.</summary>
    private static Dictionary<string, Tensor> AvatarWeights(LongCatConfig config, int seed)
    {
        Dictionary<string, Tensor> weights = LongCatTestData.Weights(config, seed);
        Random rng = new(seed + 1);
        int h = config.HiddenSize, ctx = config.AudioContextTokens, outDim = config.AudioOutputDim, inter = config.AudioIntermediateDim;
        int seqLen = config.AudioWindow, seqLenVf = config.AudioWindow + config.AudioVaeScale - 1, blocks = config.AudioBlock, channels = config.AudioChannel;
        void Add(string key, params int[] dims) => weights[key] = RandomTensor(rng, dims);
        Add("audio_proj.proj1.weight", inter, seqLen * blocks * channels);
        Add("audio_proj.proj1.bias", inter);
        Add("audio_proj.proj1_vf.weight", inter, seqLenVf * blocks * channels);
        Add("audio_proj.proj1_vf.bias", inter);
        Add("audio_proj.proj2.weight", inter, inter);
        Add("audio_proj.proj2.bias", inter);
        Add("audio_proj.proj3.weight", ctx * outDim, inter);
        Add("audio_proj.proj3.bias", ctx * outDim);
        Add("audio_proj.norm.weight", outDim);
        Add("audio_proj.norm.bias", outDim);
        for (int b = 0; b < config.Depth; b++)
        {
            string p = $"blocks.{b}.";
            Add(p + "audio_adaLN_modulation.1.weight", 3 * h, config.AdalnTembedDim);
            Add(p + "audio_adaLN_modulation.1.bias", 3 * h);
            Add(p + "pre_video_crs_attn_norm.weight", h);
            Add(p + "pre_video_crs_attn_norm.bias", h);
            Add(p + "audio_cross_attn.q_linear.weight", h, h);
            Add(p + "audio_cross_attn.q_linear.bias", h);
            Add(p + "audio_cross_attn.kv_linear.weight", 2 * h, outDim);
            Add(p + "audio_cross_attn.kv_linear.bias", 2 * h);
            Add(p + "audio_cross_attn.q_norm.weight", config.HeadDim);
            Add(p + "audio_cross_attn.k_norm.weight", config.HeadDim);
            Add(p + "audio_cross_attn.proj.weight", h, h);
            Add(p + "audio_cross_attn.proj.bias", h);
        }
        return weights;
    }

    private static Tensor RandomTensor(Random rng, params int[] dims)
    {
        Tensor tensor = new(new TensorShape(dims.Select(d => (long)d).ToArray()), DType.F32);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
            values[i] = (float)(rng.NextDouble() * 0.2 - 0.1);
        return tensor;
    }

    private static float[] Values(Random rng, int count)
    {
        float[] values = new float[count];
        for (int i = 0; i < count; i++)
            values[i] = (float)(rng.NextDouble() * 2 - 1);
        return values;
    }

    private static string PtxDirectory()
    {
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate))
                return candidate;
        }

        throw new DirectoryNotFoundException("HartsyInference.Cuda/Ptx was not found above the test binary.");
    }
}
