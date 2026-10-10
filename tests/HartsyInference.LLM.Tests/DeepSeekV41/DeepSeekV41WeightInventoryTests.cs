using HartsyInference.Core.Exceptions;
using HartsyInference.LLM.DeepSeekV41;
using Xunit;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>Weight classification and the per-class byte sums the memory estimate is built from.</summary>
public sealed class DeepSeekV41WeightInventoryTests
{
    [Theory]
    [InlineData("layers.7.attn.wq_a.weight", DeepSeekV41WeightClass.Dense)]
    [InlineData("layers.7.ffn.gate.weight", DeepSeekV41WeightClass.Dense)]
    [InlineData("layers.7.ffn.experts.383.w3.scale", DeepSeekV41WeightClass.Expert)]
    [InlineData("layers.1.engram.embed.weight", DeepSeekV41WeightClass.Engram)]
    [InlineData("embed.weight", DeepSeekV41WeightClass.Embed)]
    [InlineData("head.weight", DeepSeekV41WeightClass.Head)]
    [InlineData("vision.blocks.3.attn.qkv.weight", DeepSeekV41WeightClass.Vision)]
    [InlineData("mtp.2.ffn.experts.5.w1.weight", DeepSeekV41WeightClass.Draft)]
    public void Classify_AssignsEachKeyItsMemoryClass(string key, DeepSeekV41WeightClass expected) =>
        Assert.Equal(expected, DeepSeekV41WeightClassifier.Classify(key));

    [Theory]
    [InlineData("model.layers.0.weight")]
    [InlineData("lm_head.weight")]
    public void Classify_ReturnsNullForForeignNames(string key) => Assert.Null(DeepSeekV41WeightClassifier.Classify(key));

    [Fact]
    public void Summarize_ThrowsOnAKeyItCannotClassify_InsteadOfDroppingItsBytes()
    {
        KeyValuePair<string, long>[] tensors = [new("embed.weight", 10), new("mystery.tensor", 5)];

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => DeepSeekV41WeightInventory.Summarize(tensors));

        Assert.Contains("mystery.tensor", error.Message);
    }

    [Fact]
    public void RealOfficialHeaders_SumToTheIndexTotalSize()
    {
        DeepSeekV41WeightInventory inventory = DeepSeekV41WeightInventory.Summarize(DeepSeekV41HeaderTemplates.ExpandOfficial());

        Assert.Equal(DeepSeekV41HeaderTemplates.OfficialTensorCount, inventory.TotalCount);
        Assert.Equal(DeepSeekV41HeaderTemplates.OfficialTotalSize, inventory.TotalBytes);
    }

}
