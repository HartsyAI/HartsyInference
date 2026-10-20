using System.Collections.Immutable;
using HartsyInference.LLM.Transformer;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>The batched decode step projects K/V for every layer at one head dim and has no V-norm, latent attention,
/// cross-attention or per-layer embedding inputs. A model using any of those decodes one sequence at a time instead of
/// running the batched step with the feature silently dropped.</summary>
public sealed class BatchDecodeExactTests
{
    private static TransformerConfig Plain => new()
    {
        HiddenSize = 16, NumLayers = 4, NumHeads = 4, NumKvHeads = 2, HeadDim = 4, IntermediateSize = 32, VocabSize = 32,
    };

    [Fact]
    public void APlainDecoder_Batches() => Assert.True(Plain.BatchDecodeExact);

    [Fact]
    public void EachFeatureTheBatchedStepLacks_DecodesAlone()
    {
        Assert.False((Plain with { VNorm = true }).BatchDecodeExact);
        Assert.False((Plain with { KvSharedFromLayer = 2 }).BatchDecodeExact);
        Assert.False((Plain with { HeadDimSwa = 8 }).BatchDecodeExact);
        Assert.False((Plain with { PerLayerEmbeddingDim = 8 }).BatchDecodeExact);
        Assert.False((Plain with { CrossAttnLayers = ImmutableHashSet.Create(1) }).BatchDecodeExact);
        Assert.False((Plain with
        {
            Mla = new MlaConfig { KvLoraRank = 8, QkNopeHeadDim = 4, QkRopeHeadDim = 2, VHeadDim = 4 },
        }).BatchDecodeExact);
    }
}
