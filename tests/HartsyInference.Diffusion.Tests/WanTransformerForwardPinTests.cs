using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Pins the plain Wan-Video DiT forward on a tiny random checkpoint, on the CPU. The Wan test suite was removed
/// from the tree on 2026-11-05, and several models share <see cref="WanVideoTransformer"/>, so a change to its input
/// embedding, block loop or output head had nothing that failed on a silent change of the numbers. This does not show the
/// forward matches the Wan reference (the layer-diff against it needs the real weights); it shows a refactor kept the
/// numbers it had. The golden values are what this code produced, not values from a reference. The run is F32 on the CPU
/// backend; the F16 branch path a GPU backend can take has its own numbers, so these do not carry over to it.</summary>
public sealed class WanTransformerForwardPinTests
{
    private const int Heads = 2;
    private const int HeadDim = 128;
    private const int Inner = Heads * HeadDim;
    private const int Layers = 2;
    private const int Ffn = 64;
    private const int Channels = 4;
    private const int TextDim = 32;
    private const int FreqDim = 32;
    private const int TextLength = 5;
    private const int LatentFrames = 3;
    private const int LatentHeight = 4;
    private const int LatentWidth = 4;

    // Fingerprint of the velocity for each forward below: sum, sum of squares, a position-weighted sum (sum of v[i]·(i+1),
    // so a permuted token or channel changes it), and the first, a middle and the last value. Relative tolerance, not exact:
    // the CPU kernels may split a reduction differently on another machine.
    private static readonly double[] SingleTimestep = [-264.61846899986267, 1137.8716598252431, -23226.772809684277, -3.1172897815704346, -0.34743446111679077, -1.3682007789611816];
    private static readonly double[] PerFrameTimestep = [-266.9602277725935, 1075.614074899884, -26666.346339821815, -1.0185980796813965, 2.1148500442504883, -1.9022417068481445];
    private const double RelativeTolerance = 1e-4;

    private static WanVideoConfig TinyConfig => new()
    {
        NumHeads = Heads, HeadDim = HeadDim, NumLayers = Layers, FfnDim = Ffn,
        InChannels = Channels, OutChannels = Channels, VaeLatentChannels = Channels,
        TextDim = TextDim, FreqDim = FreqDim, VaeSpatialCompression = 8,
    };

    [Fact]
    public void DetectorReadsTheTinyCheckpointBackToItsConfig()
    {
        using TinyCheckpoint checkpoint = new();

        WanVideoConfig detected = WanConfigDetector.Detect(checkpoint.Weights);

        Assert.Equal(Inner, detected.InnerDim);
        Assert.Equal(Heads, detected.NumHeads);
        Assert.Equal(Layers, detected.NumLayers);
        Assert.Equal(Ffn, detected.FfnDim);
        Assert.Equal(Channels, detected.InChannels);
        Assert.Equal(Channels, detected.OutChannels);
        Assert.Equal((1, 2, 2), detected.PatchSize);
        Assert.False(detected.HasImageConditioning);
        Assert.False(detected.IsMixtureOfExperts);
    }

    [Fact]
    public void SingleTimestepForwardKeepsItsNumbers()
    {
        AssertPinned(Fingerprint(Forward([600f])), SingleTimestep);
    }

    /// <summary>One timestep per latent frame is the TI2V per-frame-modulation path, which runs different glue in every
    /// block and in the final layer.</summary>
    [Fact]
    public void PerFrameTimestepForwardKeepsItsNumbers()
    {
        AssertPinned(Fingerprint(Forward([0f, 600f, 600f])), PerFrameTimestep);
    }

    [Fact]
    public void ForwardIsDeterministic()
    {
        Assert.Equal(Fingerprint(Forward([600f])), Fingerprint(Forward([600f])));
    }

    /// <summary>A second forward on the same transformer and the same encoder tensor reads the cached text context instead of
    /// building it, and must give the same numbers.</summary>
    [Fact]
    public void ASecondForwardOnTheSameTransformerReusesTheCachedContext()
    {
        using CpuBackend backend = new();
        using TinyCheckpoint checkpoint = new();
        using WanVideoTransformer transformer = new(TinyConfig);
        transformer.LoadWeights(checkpoint.Weights);
        using Tensor latent = IrisZTestData.Random(new Random(11), 1, Channels, LatentFrames, LatentHeight, LatentWidth);
        using Tensor text = IrisZTestData.Random(new Random(12), TextLength, TextDim);

        using Tensor first = transformer.Forward(backend, latent, text, [600f]);
        using Tensor second = transformer.Forward(backend, latent, text, [600f]);

        Assert.True(first.AsReadOnlySpan<float>().SequenceEqual(second.AsReadOnlySpan<float>()));
    }

    private static float[] Forward(float[] timesteps)
    {
        using CpuBackend backend = new();
        using TinyCheckpoint checkpoint = new();
        using WanVideoTransformer transformer = new(TinyConfig);
        transformer.LoadWeights(checkpoint.Weights);
        using Tensor latent = IrisZTestData.Random(new Random(11), 1, Channels, LatentFrames, LatentHeight, LatentWidth);
        using Tensor text = IrisZTestData.Random(new Random(12), TextLength, TextDim);

        using Tensor velocity = transformer.Forward(backend, latent, text, timesteps);

        Assert.Equal(new TensorShape([1L, Channels, LatentFrames, LatentHeight, LatentWidth]), velocity.Shape);
        foreach (float value in velocity.AsReadOnlySpan<float>())
            Assert.True(float.IsFinite(value), "the velocity holds a non-finite value");
        return velocity.AsReadOnlySpan<float>().ToArray();
    }

    private static double[] Fingerprint(float[] values)
    {
        double sum = 0, squares = 0, weighted = 0;
        for (int i = 0; i < values.Length; i++)
        {
            sum += values[i];
            squares += (double)values[i] * values[i];
            weighted += (double)values[i] * (i + 1);
        }
        return [sum, squares, weighted, values[0], values[values.Length / 2 + 3], values[^1]];
    }

    private static void AssertPinned(double[] actual, double[] expected)
    {
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(Math.Abs(actual[i] - expected[i]) <= RelativeTolerance * Math.Max(1.0, Math.Abs(expected[i])),
                $"fingerprint[{i}] {actual[i]:R} against the pinned {expected[i]:R}; actual: [{string.Join(", ", actual.Select(v => v.ToString("R")))}]");
        }
    }

    /// <summary>A diffusers-named Wan checkpoint with random weights, built in memory.</summary>
    private sealed class TinyCheckpoint : IDisposable
    {
        private readonly Random _rng = new(7);

        public TinyCheckpoint()
        {
            Weights["patch_embedding.weight"] = Random(Inner, Channels, 1, 2, 2);
            Weights["patch_embedding.bias"] = Random(Inner);
            Weights["proj_out.weight"] = Random(Channels * 4, Inner);
            Weights["proj_out.bias"] = Random(Channels * 4);
            Weights["scale_shift_table"] = Random(1, 2, Inner);
            Weights["condition_embedder.time_embedder.linear_1.weight"] = Random(Inner, FreqDim);
            Weights["condition_embedder.time_embedder.linear_1.bias"] = Random(Inner);
            Weights["condition_embedder.time_embedder.linear_2.weight"] = Random(Inner, Inner);
            Weights["condition_embedder.time_embedder.linear_2.bias"] = Random(Inner);
            Weights["condition_embedder.time_proj.weight"] = Random(6 * Inner, Inner);
            Weights["condition_embedder.time_proj.bias"] = Random(6 * Inner);
            Weights["condition_embedder.text_embedder.linear_1.weight"] = Random(Inner, TextDim);
            Weights["condition_embedder.text_embedder.linear_1.bias"] = Random(Inner);
            Weights["condition_embedder.text_embedder.linear_2.weight"] = Random(Inner, Inner);
            Weights["condition_embedder.text_embedder.linear_2.bias"] = Random(Inner);
            for (int i = 0; i < Layers; i++)
            {
                string block = $"blocks.{i}";
                Weights[$"{block}.scale_shift_table"] = Random(1, 6, Inner);
                foreach (string attention in new[] { "attn1", "attn2" })
                {
                    foreach (string projection in new[] { "to_q", "to_k", "to_v", "to_out.0" })
                    {
                        Weights[$"{block}.{attention}.{projection}.weight"] = Random(Inner, Inner);
                        Weights[$"{block}.{attention}.{projection}.bias"] = Random(Inner);
                    }
                    Weights[$"{block}.{attention}.norm_q.weight"] = Scale(Inner);
                    Weights[$"{block}.{attention}.norm_k.weight"] = Scale(Inner);
                }
                Weights[$"{block}.norm2.weight"] = Scale(Inner);
                Weights[$"{block}.norm2.bias"] = Random(Inner);
                Weights[$"{block}.ffn.net.0.proj.weight"] = Random(Ffn, Inner);
                Weights[$"{block}.ffn.net.0.proj.bias"] = Random(Ffn);
                Weights[$"{block}.ffn.net.2.weight"] = Random(Inner, Ffn);
                Weights[$"{block}.ffn.net.2.bias"] = Random(Inner);
            }
        }

        public Dictionary<string, Tensor> Weights { get; } = [];

        public void Dispose()
        {
            foreach (Tensor tensor in Weights.Values) tensor.Dispose();
        }

        private Tensor Random(params long[] dims) => IrisZTestData.Random(_rng, dims);

        // Norm weights near 1 but all different, so a dropped weight or a swapped norm_q / norm_k moves the output. The biases are
    // random and non-zero for the same reason: they load with TryGetValue, so a renamed bias key would otherwise go unnoticed.
    private Tensor Scale(params long[] dims)
    {
        Tensor tensor = Random(dims);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++) values[i] += 1f;
        return tensor;
    }
    }
}
