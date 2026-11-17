using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Pins the plain Wan-Video DiT forward on a tiny random checkpoint, on the CPU. The Wan test suite was removed
/// from the tree on 2026-11-05, and several models share <see cref="WanVideoTransformer"/>, so a change to its input
/// embedding, block loop or output head had nothing that failed on a silent change of the numbers. This does not show the
/// forward matches the Wan reference (the layer-diff against it needs the real weights); it shows a refactor kept the
/// numbers it had. The golden sums are values this code produced, not values from a reference.</summary>
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

    // Sum and sum of squares of the velocity for each forward below. Relative tolerance, not exact: the CPU kernels may
    // split a reduction differently on another machine.
    private const double SingleTimestepSum = 48.218642860651016;
    private const double SingleTimestepSquares = 1185.6868315964696;
    private const double PerFrameTimestepSum = 48.76617577672005;
    private const double PerFrameTimestepSquares = 1166.7405797909341;
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
        (double sum, double squares) = Forward([0.6f]);

        AssertPinned(sum, squares, SingleTimestepSum, SingleTimestepSquares);
    }

    /// <summary>One timestep per latent frame is the TI2V per-frame-modulation path, which runs different glue in every
    /// block and in the final layer.</summary>
    [Fact]
    public void PerFrameTimestepForwardKeepsItsNumbers()
    {
        (double sum, double squares) = Forward([0f, 0.6f, 0.6f]);

        AssertPinned(sum, squares, PerFrameTimestepSum, PerFrameTimestepSquares);
    }

    [Fact]
    public void ForwardIsDeterministic()
    {
        (double firstSum, double firstSquares) = Forward([0.6f]);
        (double secondSum, double secondSquares) = Forward([0.6f]);

        Assert.Equal(firstSum, secondSum);
        Assert.Equal(firstSquares, secondSquares);
    }

    private static (double Sum, double Squares) Forward(float[] timesteps)
    {
        using CpuBackend backend = new();
        using TinyCheckpoint checkpoint = new();
        using WanVideoTransformer transformer = new(TinyConfig);
        transformer.LoadWeights(checkpoint.Weights);
        using Tensor latent = IrisZTestData.Random(new Random(11), 1, Channels, LatentFrames, LatentHeight, LatentWidth);
        using Tensor text = IrisZTestData.Random(new Random(12), TextLength, TextDim);

        using Tensor velocity = transformer.Forward(backend, latent, text, timesteps);

        Assert.Equal(new TensorShape([1L, Channels, LatentFrames, LatentHeight, LatentWidth]), velocity.Shape);
        double sum = 0, squares = 0;
        foreach (float value in velocity.AsReadOnlySpan<float>())
        {
            Assert.True(float.IsFinite(value), "the velocity holds a non-finite value");
            sum += value;
            squares += (double)value * value;
        }
        return (sum, squares);
    }

    private static void AssertPinned(double sum, double squares, double expectedSum, double expectedSquares)
    {
        Assert.True(Math.Abs(sum - expectedSum) <= RelativeTolerance * Math.Max(1.0, Math.Abs(expectedSum)),
            $"sum {sum:R} against the pinned {expectedSum:R}");
        Assert.True(Math.Abs(squares - expectedSquares) <= RelativeTolerance * Math.Max(1.0, expectedSquares),
            $"sum of squares {squares:R} against the pinned {expectedSquares:R}");
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
                    Weights[$"{block}.{attention}.norm_q.weight"] = Ones(Inner);
                    Weights[$"{block}.{attention}.norm_k.weight"] = Ones(Inner);
                }
                Weights[$"{block}.norm2.weight"] = Ones(Inner);
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

        private static Tensor Ones(params long[] dims) => IrisZTestData.Filled(1f, dims);
    }
}
