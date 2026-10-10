using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Gguf;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Published Kolibri-1 GGUFs carry only expert FFN lengths; the dense <c>feed_forward_length</c> key is absent.</summary>
public sealed class KolibriGgufConfigTests
{
    private static GgufMetadata Metadata(bool denseLength, bool expertLength)
    {
        GgufMetadata m = new();
        m.Add("general.architecture", "kolibri1");
        m.Add("kolibri1.block_count", 2u);
        m.Add("kolibri1.embedding_length", 8u);
        m.Add("kolibri1.attention.head_count", 2u);
        if (denseLength) m.Add("kolibri1.feed_forward_length", 24u);
        if (expertLength) m.Add("kolibri1.expert_feed_forward_length", 16u);
        return m;
    }

    private static Dictionary<string, Tensor> Weights() => new()
    {
        ["model.embed_tokens.weight"] = new Tensor(new TensorShape(32, 8), DType.F32),
    };

    [Fact]
    public void MissingDenseFfnLength_FallsBackToExpertFfnLength()
    {
        TransformerConfig config = GgufConfigFactory.FromGguf(Metadata(denseLength: false, expertLength: true), Weights());
        Assert.Equal(16, config.IntermediateSize);
    }

    [Fact]
    public void NoFfnLengthAtAll_Throws()
    {
        Assert.Throws<ArgumentException>(() => GgufConfigFactory.FromGguf(Metadata(false, false), Weights()));
    }
}
