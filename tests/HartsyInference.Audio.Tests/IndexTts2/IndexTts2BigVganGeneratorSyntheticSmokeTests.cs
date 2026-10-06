using HartsyInference.Audio.Models.IndexTts2;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.PyTorch;
using Xunit;

namespace HartsyInference.Audio.Tests.IndexTts2;

/// <summary>Tiny random weights through <see cref="IndexTts2BigVganGenerator"/>: output length (<c>T·hop</c>),
/// finiteness, and determinism only. Says nothing about parity with the real
/// <c>nvidia/bigvgan_v2_22khz_80band_256x</c> checkpoint.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class IndexTts2BigVganGeneratorSyntheticSmokeTests : IDisposable
{
    private const string Prefix = "generator";
    private readonly List<Tensor> _owned = [];
    private readonly CpuBackend _backend = new();

    private static readonly IndexTts2BigVganConfig Cfg = new()
    {
        InputChannels = 8,
        UpsampleInitialChannel = 16,
        UpsampleRates = [2, 2],
        UpsampleKernelSizes = [4, 4],
        ResblockKernelSizes = [3, 5],
        ResblockDilations = [[1, 3], [1, 3]],
    };

    private Tensor Rand(Random rng, double amp, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        foreach (ref float v in t.AsSpan<float>()) v = (float)((rng.NextDouble() * 2 - 1) * amp);
        _owned.Add(t);
        return t;
    }

    private void AddSnakeWeights(Dictionary<string, Tensor> w, string prefix, Random rng, int channels)
    {
        w[$"{prefix}.act.alpha"] = Rand(rng, 0.05, channels);
        w[$"{prefix}.act.beta"] = Rand(rng, 0.05, channels);
        w[$"{prefix}.upsample.filter"] = Rand(rng, 0.3, 12);
        w[$"{prefix}.downsample.lowpass.filter"] = Rand(rng, 0.3, 12);
    }

    private void AddResblock(Dictionary<string, Tensor> w, string prefix, Random rng, int channels, int kernel, int[] dilations)
    {
        for (int i = 0; i < dilations.Length; i++)
        {
            w[$"{prefix}.convs1.{i}.weight"] = Rand(rng, 0.1, channels, channels, kernel);
            w[$"{prefix}.convs1.{i}.bias"] = Rand(rng, 0.02, channels);
            w[$"{prefix}.convs2.{i}.weight"] = Rand(rng, 0.1, channels, channels, kernel);
            w[$"{prefix}.convs2.{i}.bias"] = Rand(rng, 0.02, channels);
        }
        for (int i = 0; i < 2 * dilations.Length; i++) AddSnakeWeights(w, $"{prefix}.activations.{i}", rng, channels);
    }

    private IndexTts2BigVganGenerator BuildGenerator(Random rng)
    {
        Dictionary<string, Tensor> w = [];
        w[$"{Prefix}.conv_pre.weight"] = Rand(rng, 0.1, Cfg.UpsampleInitialChannel, Cfg.InputChannels, 7);
        w[$"{Prefix}.conv_pre.bias"] = Rand(rng, 0.02, Cfg.UpsampleInitialChannel);

        int numStages = Cfg.UpsampleRates.Length;
        for (int i = 0; i < numStages; i++)
        {
            int inCh = Cfg.UpsampleInitialChannel / (1 << i);
            int outCh = Cfg.UpsampleInitialChannel / (1 << (i + 1));
            w[$"{Prefix}.ups.{i}.0.weight"] = Rand(rng, 0.1, inCh, outCh, Cfg.UpsampleKernelSizes[i]);
            w[$"{Prefix}.ups.{i}.0.bias"] = Rand(rng, 0.02, outCh);
            for (int j = 0; j < Cfg.ResblockKernelSizes.Length; j++)
            {
                int idx = i * Cfg.ResblockKernelSizes.Length + j;
                AddResblock(w, $"{Prefix}.resblocks.{idx}", rng, outCh, Cfg.ResblockKernelSizes[j], Cfg.ResblockDilations[j]);
            }
        }

        int finalCh = Cfg.UpsampleInitialChannel / (1 << numStages);
        AddSnakeWeights(w, $"{Prefix}.activation_post", rng, finalCh);
        w[$"{Prefix}.conv_post.weight"] = Rand(rng, 0.1, 1, finalCh, 7);   // no conv_post.bias — use_bias_at_final: false.

        IndexTts2BigVganGenerator gen = new(Cfg);
        gen.LoadWeights(w, Prefix);
        return gen;
    }

    [Fact]
    public void Forward_ProducesFiniteOutput_OfExpectedUpsampledLength()
    {
        Random rng = new(1);
        using IndexTts2BigVganGenerator gen = BuildGenerator(rng);
        const int melLen = 5;
        Tensor mel = Rand(rng, 1.0, 1, Cfg.InputChannels, melLen);

        using Tensor wave = gen.Forward(_backend, mel, melLen);

        int hop = Cfg.UpsampleRates.Aggregate(1, (a, b) => a * b);
        Assert.Equal(new TensorShape(1, 1, melLen * hop), wave.Shape);
        foreach (float v in wave.AsSpan<float>()) Assert.True(float.IsFinite(v));
    }

    [Fact]
    public void Forward_IsDeterministic_ForTheSameInput()
    {
        Random rng = new(2);
        using IndexTts2BigVganGenerator gen = BuildGenerator(rng);
        const int melLen = 4;
        Tensor mel = Rand(rng, 1.0, 1, Cfg.InputChannels, melLen);

        using Tensor a = gen.Forward(_backend, mel, melLen);
        using Tensor b = gen.Forward(_backend, mel, melLen);

        float[] av = a.AsSpan<float>().ToArray(), bv = b.AsSpan<float>().ToArray();
        for (int i = 0; i < av.Length; i++) Assert.Equal(av[i], bv[i], 5);
    }

    public void Dispose()
    {
        foreach (Tensor t in _owned) t.Dispose();
        _backend.Dispose();
    }
}

/// <summary>Loads the real <c>nvidia/bigvgan_v2_22khz_80band_256x</c> <c>bigvgan_generator.pt</c> (too large to
/// bundle — point <c>INDEXTTS2_BIGVGAN_PT_PATH</c> at a local copy). A clean <c>LoadWeights</c> confirms every
/// key name this class expects matches the real checkpoint (including the absence of <c>cond_layer</c>,
/// <c>conds.N</c> and <c>conv_post.bias</c>); one <c>Forward</c> call confirms the full upsample stack runs
/// end to end and produces finite output of the expected length.</summary>
[Trait("Category", "Integration")]
public sealed class IndexTts2BigVganGeneratorRealWeightTests
{
    [Fact]
    public void Forward_SucceedsAgainstRealBigVganCheckpoint()
    {
        string? path = Environment.GetEnvironmentVariable("INDEXTTS2_BIGVGAN_PT_PATH");
        if (string.IsNullOrEmpty(path) || !File.Exists(path)) return;   // resource-gated: skip without the real checkpoint file

        using PytorchPickleLoader loader = new();
        loader.Load(path, recursiveFlatten: true);
        Dictionary<string, Tensor> weights = loader.GetAllTensors();
        try
        {
            IndexTts2BigVganConfig cfg = IndexTts2BigVganConfig.Nvidia22kHz80Band;
            using IndexTts2BigVganGenerator gen = new(cfg);
            gen.LoadWeights(weights, "generator");

            const int melLen = 10;
            Random rng = new(3);
            Tensor mel = new(new TensorShape(1, cfg.InputChannels, melLen), DType.F32);
            foreach (ref float v in mel.AsSpan<float>()) v = (float)(rng.NextDouble() * 2 - 1);

            using CpuBackend backend = new();
            using Tensor wave = gen.Forward(backend, mel, melLen);
            mel.Dispose();

            int hop = cfg.UpsampleRates.Aggregate(1, (a, b) => a * b);
            Assert.Equal(new TensorShape(1, 1, melLen * hop), wave.Shape);
            foreach (float v in wave.AsSpan<float>()) Assert.True(float.IsFinite(v));
        }
        finally
        {
            foreach (Tensor t in weights.Values) t.Dispose();
        }
    }
}
