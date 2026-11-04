using Xunit;
using HartsyInference.ModelAssets.CheckpointConverters;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Key mapping for the Cosmos-Predict1 AR checkpoint. The real model.pt names its embedding
/// <c>tok_embeddings.weight</c>. A suffix rename without a '.' boundary used to turn that into
/// <c>tok_tok_embeddings.weight</c>, so the transformer could not find its embedding.</summary>
public class CosmosArCheckpointConverterTests
{
    [Theory]
    [InlineData("tok_embeddings.weight", "tok_embeddings.weight")]
    [InlineData("embeddings.weight", "tok_embeddings.weight")]
    [InlineData("model.embeddings.weight", "tok_embeddings.weight")]
    [InlineData("model.tok_embeddings.weight", "tok_embeddings.weight")]
    [InlineData("token_embedding.weight", "tok_embeddings.weight")]
    [InlineData("layers.0.attention.wq.weight", "layers.0.attention.wq.weight")]
    public void MapKey_NamesTheEmbeddingOnce(string source, string expected) =>
        Assert.Equal(expected, CosmosArCheckpointConverter.MapKey(source));
}
