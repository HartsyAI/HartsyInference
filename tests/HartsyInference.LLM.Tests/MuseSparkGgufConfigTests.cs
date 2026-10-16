using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Gguf;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>A Muse Spark GGUF's per-expert output norm is read from its first sparse layer, so leading dense layers, which carry none,
/// do not switch it off for the sparse ones.</summary>
public sealed class MuseSparkGgufConfigTests
{
    private static GgufMetadata Metadata(uint leadingDense)
    {
        GgufMetadata m = new();
        m.Add("general.architecture", "muse-spark");
        m.Add("muse-spark.block_count", 3u);
        m.Add("muse-spark.embedding_length", 8u);
        m.Add("muse-spark.attention.head_count", 2u);
        m.Add("muse-spark.feed_forward_length", 16u);
        m.Add("muse-spark.expert_count", 4u);
        m.Add("muse-spark.expert_used_count", 2u);
        m.Add("muse-spark.expert_feed_forward_length", 8u);
        m.Add("muse-spark.leading_dense_block_count", leadingDense);
        return m;
    }

    private static Dictionary<string, Tensor> Weights(uint firstSparse)
    {
        Dictionary<string, Tensor> w = new() { ["model.embed_tokens.weight"] = new Tensor(new TensorShape(32, 8), DType.F32) };
        for (uint layer = firstSparse; layer < 3; layer++)
        {
            w[$"model.layers.{layer}.mlp.post_expert_norm.weight"] = new Tensor(new TensorShape(8), DType.F32);
        }
        return w;
    }

    [Theory]
    [InlineData(0u)]
    [InlineData(1u)]
    public void PostExpertNorm_IsReadFromTheFirstSparseLayer(uint leadingDense)
    {
        TransformerConfig config = GgufConfigFactory.FromGguf(Metadata(leadingDense), Weights(leadingDense));

        Assert.True(config.Moe!.PostExpertNorm);
        Assert.Equal((int)leadingDense, config.Moe.FirstDenseLayers);
    }

    [Fact]
    public void NoPostExpertNormTensor_LeavesItOff()
    {
        TransformerConfig config = GgufConfigFactory.FromGguf(Metadata(0u), Weights(firstSparse: 3));

        Assert.False(config.Moe!.PostExpertNorm);
    }
}
