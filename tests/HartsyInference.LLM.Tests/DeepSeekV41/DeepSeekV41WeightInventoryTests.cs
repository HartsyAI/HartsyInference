using HartsyInference.Core.Exceptions;
using HartsyInference.LLM.DeepSeekV41;
using Xunit;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>Weight classification and the per-class byte sums the memory estimate is built from.</summary>
public sealed class DeepSeekV41WeightInventoryTests
{
    [Theory]
    [InlineData("layers.7.attn.wq_a.weight", DeepSeekV41WeightClass.Dense)]
    [InlineData("layers.7.ffn.shared_experts.w1.weight", DeepSeekV41WeightClass.Dense)]
    [InlineData("layers.7.ffn.gate.weight", DeepSeekV41WeightClass.Dense)]
    [InlineData("layers.1.engram.wkv.weight", DeepSeekV41WeightClass.Dense)]
    [InlineData("layers.7.ffn.experts.383.w3.scale", DeepSeekV41WeightClass.Expert)]
    [InlineData("layers.1.engram.embed.weight", DeepSeekV41WeightClass.Engram)]
    [InlineData("layers.14.engram.embed.scale", DeepSeekV41WeightClass.Engram)]
    [InlineData("embed.weight", DeepSeekV41WeightClass.Embed)]
    [InlineData("head.weight", DeepSeekV41WeightClass.Head)]
    [InlineData("norm.weight", DeepSeekV41WeightClass.Head)]
    [InlineData("vision.blocks.3.attn.qkv.weight", DeepSeekV41WeightClass.Vision)]
    [InlineData("aligner.w1.weight", DeepSeekV41WeightClass.Vision)]
    [InlineData("image_newline", DeepSeekV41WeightClass.Vision)]
    [InlineData("mtp.2.ffn.experts.5.w1.weight", DeepSeekV41WeightClass.Draft)]
    [InlineData("mtp.0.markov_head.proj.weight", DeepSeekV41WeightClass.Draft)]
    public void Classify_AssignsEachKeyItsMemoryClass(string key, DeepSeekV41WeightClass expected) =>
        Assert.Equal(expected, DeepSeekV41WeightClassifier.Classify(key));

    [Theory]
    [InlineData("model.layers.0.weight")]
    [InlineData("layers.x.attn.weight")]
    [InlineData("layers.3")]
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
    public void Summarize_ReportsEveryClassEvenWhenEmpty()
    {
        DeepSeekV41WeightInventory inventory = DeepSeekV41WeightInventory.Summarize([new("embed.weight", 10)]);

        Assert.Equal(Enum.GetValues<DeepSeekV41WeightClass>().Length, inventory.BytesByClass.Count);
        Assert.Equal(10, inventory.BytesByClass[DeepSeekV41WeightClass.Embed]);
        Assert.Equal(0, inventory.BytesByClass[DeepSeekV41WeightClass.Engram]);
        Assert.Equal(10, inventory.TotalBytes);
        Assert.Equal(1, inventory.TotalCount);
    }

    [Fact]
    public void RealOfficialHeaders_SumToTheIndexTotalSize()
    {
        DeepSeekV41WeightInventory inventory = DeepSeekV41WeightInventory.Summarize(DeepSeekV41HeaderTemplates.ExpandOfficial());

        Assert.Equal(DeepSeekV41HeaderTemplates.OfficialTensorCount, inventory.TotalCount);
        Assert.Equal(DeepSeekV41HeaderTemplates.OfficialTotalSize, inventory.TotalBytes);
    }

    [Fact]
    public void RealOfficialHeaders_SplitIntoTheKnownClassBytes()
    {
        DeepSeekV41WeightInventory inventory = DeepSeekV41WeightInventory.Summarize(DeepSeekV41HeaderTemplates.ExpandOfficial());

        Assert.Equal(7_199_083_968, inventory.BytesByClass[DeepSeekV41WeightClass.Dense]);
        Assert.Equal(288_777_830_400, inventory.BytesByClass[DeepSeekV41WeightClass.Expert]);
        Assert.Equal(202_758_032_400, inventory.BytesByClass[DeepSeekV41WeightClass.Engram]);
        Assert.Equal(1_323_827_200, inventory.BytesByClass[DeepSeekV41WeightClass.Embed]);
        Assert.Equal(1_323_837_440, inventory.BytesByClass[DeepSeekV41WeightClass.Head]);
        Assert.Equal(970_536_960, inventory.BytesByClass[DeepSeekV41WeightClass.Vision]);
        Assert.Equal(7_932_874_632, inventory.BytesByClass[DeepSeekV41WeightClass.Draft]);
        Assert.Equal(1_251, inventory.CountsByClass[DeepSeekV41WeightClass.Dense]);
        Assert.Equal(92_160, inventory.CountsByClass[DeepSeekV41WeightClass.Expert]);
        Assert.Equal(4, inventory.CountsByClass[DeepSeekV41WeightClass.Engram]);
        Assert.Equal(2_401, inventory.CountsByClass[DeepSeekV41WeightClass.Draft]);
    }

    [Fact]
    public void RealEngramTables_AreTwoHundredGigabytesAndSizedByTheConfig()
    {
        DeepSeekV41Config config = DeepSeekV41Config.Parse(DeepSeekV41Fixtures.Read("official_config.json"));
        long engramEmbedBytes = DeepSeekV41HeaderTemplates.ExpandOfficial()
            .Where(static pair => pair.Key.EndsWith(".engram.embed.weight", StringComparison.Ordinal)).Sum(static pair => pair.Value);

        Assert.Equal(config.EngramNumEmbeddings.Sum() * config.EngramHeadDim, engramEmbedBytes);
    }
}
