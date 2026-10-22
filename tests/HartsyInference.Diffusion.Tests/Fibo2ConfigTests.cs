using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Fibo 2.0 gist schedule and checkpoint key mapping: the trap-prone integer rules, checked on hand-computed values.</summary>
public sealed class Fibo2ConfigTests
{
    [Theory]
    [InlineData(0, 256)]
    [InlineData(100, 256)]
    [InlineData(256, 256)]
    [InlineData(300, 320)]
    [InlineData(1000, 384)]
    [InlineData(1152, 384)]
    [InlineData(2000, 512)]
    [InlineData(2048, 512)]
    [InlineData(9999, 512)]
    public void GistCount_Checkpoint_MatchesComfySchedule(int textLength, int expected) =>
        Assert.Equal(expected, Fibo2Config.Checkpoint.GistCount(textLength));

    [Fact]
    public void GistCount_TinyFixtureSchedule_MatchesPythonReference()
    {
        // The schedule of TINY in tests/python-reference/fibo2_reference.py.
        Fibo2Config tiny = Fibo2Config.Checkpoint with
        {
            MaxGist = 8,
            MinGist = 4,
            GistStep = 2,
            GistMinTextLen = 16,
            GistMaxTextLen = 64,
        };
        Assert.Equal(6, tiny.GistCount(40));
        Assert.Equal(4, tiny.GistCount(16));
        Assert.Equal(8, tiny.GistCount(64));
    }

    [Fact]
    public void FromWeights_ReadsStructureFromTensorShapes()
    {
        Dictionary<string, Tensor> weights = TinyCheckpointShapes();
        try
        {
            Fibo2Config config = Fibo2Config.FromWeights(weights, TinyTemplate());
            Assert.Equal(8, config.Dim);
            Assert.Equal(2, config.NumHeads);
            Assert.Equal(4, config.HeadDim);
            Assert.Equal(3, config.NumLayers);
            Assert.Equal(2, config.NumRefinerLayers);
            Assert.Equal(1, config.PerceiverLayers);
            Assert.Equal(4, config.PerceiverQueries);
            Assert.Equal(12, config.FfnDim);
            Assert.Equal(6, config.CapFeatDim);
            Assert.Equal(new[] { 1 }, config.InjectionLayers);
        }
        finally
        {
            DisposeAll(weights);
        }
    }

    [Fact]
    public void FromWeights_NonContiguousBlockIndices_Throws()
    {
        Dictionary<string, Tensor> weights = TinyCheckpointShapes();
        try
        {
            // Layers 0 and 2 without 1: a gap must fail, not silently shorten the stack.
            weights["layers.1.attention.qkv.weight"].Dispose();
            weights.Remove("layers.1.attention.qkv.weight");
            Assert.Throws<InvalidDataException>(() => Fibo2Config.FromWeights(weights, TinyTemplate()));
        }
        finally
        {
            DisposeAll(weights);
        }
    }

    [Fact]
    public void Checkpoint_PassesValidation() => Fibo2Config.Checkpoint.Validate();

    private static Fibo2Config TinyTemplate() => Fibo2Config.Checkpoint with
    {
        InChannels = 32,
        PatchSize = 2,
        AxesDims = [2, 1, 1],
        MaxGist = 4,
        MinGist = 2,
        GistStep = 2,
        GistMinTextLen = 8,
        GistMaxTextLen = 16,
    };

    /// <summary>The key set and shapes a dim-8, 2-head, 3-layer checkpoint (injection at layer 1) defines.</summary>
    private static Dictionary<string, Tensor> TinyCheckpointShapes()
    {
        Dictionary<string, Tensor> weights = new(StringComparer.Ordinal);
        for (int i = 0; i < 3; i++)
        {
            weights[$"layers.{i}.attention.qkv.weight"] = Shaped(24, 8);
            weights[$"layers.{i}.attention.q_norm.weight"] = Shaped(4);
            weights[$"layers.{i}.feed_forward.w13.weight"] = Shaped(24, 8);
            weights[$"layers.{i}.feed_forward.w2.weight"] = Shaped(8, 12);
        }
        weights["layers.1.cross_attention.to_q.weight"] = Shaped(8, 8);
        weights["noise_refiner.0.attention.qkv.weight"] = Shaped(24, 8);
        weights["noise_refiner.1.attention.qkv.weight"] = Shaped(24, 8);
        weights["perceiver.layers.0.self_attention.to_kv.weight"] = Shaped(16, 8);
        weights["perceiver.layers.0.cross_attention.to_kv.weight"] = Shaped(16, 6);
        weights["perceiver.latent_queries"] = Shaped(4, 8);
        return weights;
    }

    private static Tensor Shaped(params long[] dims) => new(new TensorShape(dims), DType.F32);

    private static void DisposeAll(Dictionary<string, Tensor> weights)
    {
        foreach (Tensor tensor in weights.Values)
            tensor.Dispose();
    }
}
