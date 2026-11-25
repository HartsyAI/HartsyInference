using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines.LongCat;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The video and avatar denoise loops keep their state on the device. On CUDA the device's device-to-host counter is compared across a run
/// of six steps and a run of three, so fixed costs (text projections, the final latent read) cancel. Measured on a CUDA machine, one transformer
/// forward costs one read in steady state, the CFG-zero sums cost one read per guided step, and the loop's own ops cost none. Each guided step
/// therefore budgets its forwards (two for video, three for the avatar's three passes) plus one read of slack for the sums and the combine. The
/// budget is an upper bound, so it catches a plumbing op that reads back per frame or per element. Returns without asserting when CUDA is not
/// available, so the suite stays green on a CPU-only machine.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class LongCatDeviceResidencyGpuTests
{
    private const int Frames = 3, Height = 4, Width = 4, TextLength = 6;
    private static readonly int Hw = Height * Width;

    [Fact]
    public void AVideoRunReadsWithinItsForwardsAndSumsBudget()
    {
        if (!CudaContext.IsAvailable())
            return;
        LongCatConfig config = LongCatTestData.Tiny;
        Dictionary<string, Tensor> weights = LongCatTestData.Weights(config, 97);
        try
        {
            using CudaBackend cuda = new(0, PtxDirectory());
            using LongCatTransformer transformer = new(config);
            transformer.LoadWeights(weights);
            long shortRun = CountVideoReads(cuda, transformer, config, steps: 3);
            long longRun = CountVideoReads(cuda, transformer, config, steps: 6);

            // Two forwards and one sums read per guided step, plus one read of slack: four per step.
            Assert.True(longRun - shortRun <= 3 * 4, $"{longRun - shortRun} reads over three extra steps");
        }
        finally
        {
            LongCatTestData.DisposeAll(weights);
        }
    }

    [Fact]
    public void AnAvatarCachedRunReadsWithinItsForwardsBudget()
    {
        if (!CudaContext.IsAvailable())
            return;
        LongCatConfig config = AvatarConfig();
        Dictionary<string, Tensor> weights = AvatarWeights(config, 101);
        try
        {
            using CudaBackend cuda = new(0, PtxDirectory());
            using LongCatTransformer transformer = new(config);
            transformer.LoadWeights(weights);
            long shortRun = CountAvatarReads(cuda, transformer, config, steps: 3);
            long longRun = CountAvatarReads(cuda, transformer, config, steps: 6);

            // Three forwards per guided step, plus one read of slack: four per step.
            Assert.True(longRun - shortRun <= 3 * 4, $"{longRun - shortRun} reads over three extra steps");
        }
        finally
        {
            LongCatTestData.DisposeAll(weights);
        }
    }

    private static long CountVideoReads(CudaBackend cuda, LongCatTransformer transformer, LongCatConfig config, int steps)
    {
        Random rng = new(23);
        LongCatPrompt positive = new(Values(rng, TextLength * config.CaptionChannels), TextLength, TextLength);
        LongCatPrompt negative = new(Values(rng, TextLength * config.CaptionChannels), TextLength, TextLength);
        LongCatRun run = new(config.InChannels, Frames, Height, Width, Values(rng, config.InChannels * Frames * Hw), positive, negative, steps,
            Distill: false, Shift: 3f, Guidance: 4f);
        cuda.ResetD2hSyncCount();
        new LongCatVideoPipeline(cuda, transformer).Denoise(run);
        return cuda.GetD2hSyncCount();
    }

    private static long CountAvatarReads(CudaBackend cuda, LongCatTransformer transformer, LongCatConfig config, int steps)
    {
        Random rng = new(29);
        int cond = 2, ctx = config.AudioContextTokens, width = config.AudioOutputDim;
        LongCatPrompt positive = new(Values(rng, TextLength * config.CaptionChannels), TextLength, TextLength);
        LongCatPrompt negative = new(Values(rng, TextLength * config.CaptionChannels), TextLength, TextLength);
        LongCatAvatarRun run = new(config.InChannels, Frames, Height, Width, Values(rng, config.InChannels * Frames * Hw), positive, negative,
            Values(rng, Frames * ctx * width), Values(rng, Frames * ctx * width), steps, LongCatAvatarSchedule.Kind.Linear, 7f, 2f, 1.5f,
            LongCatAvatarMode.Avc, cond, Values(rng, config.InChannels * cond * Hw), UseKvCache: true);
        cuda.ResetD2hSyncCount();
        new LongCatAvatarPipeline(cuda, transformer).Denoise(run);
        return cuda.GetD2hSyncCount();
    }

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
