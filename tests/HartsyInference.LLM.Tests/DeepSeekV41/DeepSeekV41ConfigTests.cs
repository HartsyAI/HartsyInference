using System.Text.Json.Nodes;
using HartsyInference.Core.Exceptions;
using HartsyInference.LLM.DeepSeekV41;
using Xunit;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>Parsing and validation of <see cref="DeepSeekV41Config"/> against the real pinned config.json.</summary>
public sealed class DeepSeekV41ConfigTests
{
    [Fact]
    public void PinnedOfficialConfig_ParsesEveryField()
    {
        DeepSeekV41Config config = DeepSeekV41Config.Parse(DeepSeekV41Fixtures.Read("official_config.json"));

        Assert.Equal(129280, config.VocabSize);
        Assert.Equal(5120, config.HiddenSize);
        Assert.Equal(40, config.NumHiddenLayers);
        Assert.Equal(3, config.NumNextnPredictLayers);
        Assert.Equal(43, config.TotalLayerCount);
        Assert.Equal(43, config.CompressRatios.Count);
        Assert.Equal(384, config.NRoutedExperts);
        Assert.Equal(128, config.DsparkNRoutedExperts);
        Assert.Equal(6, config.NumExpertsPerTok);
        Assert.Equal(1_048_576, config.MaxPositionEmbeddings);
        Assert.Equal(160000.0, config.CompressRopeTheta);
        Assert.Equal([1, 14], config.EngramLayerIds);
        Assert.Equal([384006168L, 384016682L], config.EngramNumEmbeddings);
        Assert.Equal([37, 38, 39], config.DsparkTargetLayerIds);
        Assert.Equal(129264, config.ImageTokenId);
        Assert.NotNull(config.RopeScaling);
        Assert.Equal(16.0, config.RopeScaling!.Factor);
        Assert.Equal(65536, config.RopeScaling.OriginalMaxPositionEmbeddings);
        Assert.NotNull(config.Vision);
        Assert.Equal(32, config.Vision!.NumLayers);
        Assert.Equal(1024, config.Vision.HiddenSize);
    }

    [Fact]
    public void PinnedOfficialConfig_DerivesPerLayerPlans()
    {
        DeepSeekV41Config config = DeepSeekV41Config.Parse(DeepSeekV41Fixtures.Read("official_config.json"));

        Assert.Equal(43, config.LayerPlans.Count);
        Assert.Equal([1, 14], config.LayerPlans.Where(static plan => plan.HasEngram).Select(static plan => plan.Layer));
        Assert.Equal(0, config.LayerPlans[1].EngramSlot);
        Assert.Equal(1, config.LayerPlans[14].EngramSlot);
        Assert.Equal([2, 8, 14, 20], config.LayerPlans.Where(static plan => plan.IsKvSource).Select(static plan => plan.Layer));
        Assert.Equal([2, 8, 14, 20, 24, 28, 32, 36],
            config.LayerPlans.Where(static plan => plan.IsIndexSource).Select(static plan => plan.Layer));
        Assert.Equal([40, 41, 42], config.LayerPlans.Where(static plan => plan.IsDraft).Select(static plan => plan.Layer));
        Assert.All(config.LayerPlans.Where(static plan => plan.IsDraft), static plan => Assert.Equal(0, plan.CompressRatio));
        Assert.Equal(2, config.LayerPlans[2].CompressRatio);
        Assert.Equal(1, config.LayerPlans[20].CompressRatio);
        Assert.Equal(0, config.LayerPlans[0].CompressRatio);
    }

    [Fact]
    public void FlatMlxConfig_ParsesWithoutTextConfigOrRopeScaling()
    {
        DeepSeekV41Config config = DeepSeekV41Config.Parse(DeepSeekV41Fixtures.Read("mlx_config.json"));

        Assert.Equal(40, config.NumHiddenLayers);
        Assert.Equal(43, config.CompressRatios.Count);
        Assert.Equal(384, config.NRoutedExperts);
        Assert.Equal([1, 14], config.EngramLayerIds);
    }

    [Theory]
    [InlineData(42)]
    [InlineData(44)]
    [InlineData(40)]
    public void CompressRatiosLengthMustBeLayersPlusDraftLayers(int length)
    {
        string json = DeepSeekV41Fixtures.OfficialConfig(text => text["compress_ratios"] = new JsonArray(
            Enumerable.Repeat(0, length).Select(static ratio => (JsonNode?)ratio).ToArray()));

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => DeepSeekV41Config.Parse(json));

        Assert.Contains($"compress_ratios has {length} entries, expected", error.Message);
        Assert.Contains("= 43", error.Message);
    }

    [Fact]
    public void MismatchedEngramArrays_AreRejected()
    {
        string json = DeepSeekV41Fixtures.OfficialConfig(text => text["engram_num_embeddings"] = new JsonArray(384006168L));

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => DeepSeekV41Config.Parse(json));

        Assert.Contains("engram_layer_ids has 2 entries but engram_num_embeddings has 1", error.Message);
    }

    [Fact]
    public void LayerIdsOutsideTheBackbone_AreRejected()
    {
        string json = DeepSeekV41Fixtures.OfficialConfig(text => text["kv_source_layer_ids"] = new JsonArray(2, 40));

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => DeepSeekV41Config.Parse(json));

        Assert.Contains("kv_source_layer_ids entry 40 is outside the 40 backbone layers", error.Message);
    }

    [Fact]
    public void EveryInvariantFailureIsListedAtOnce()
    {
        string json = DeepSeekV41Fixtures.OfficialConfig(text =>
        {
            text["compress_ratios"] = new JsonArray(0, 0);
            text["num_experts_per_tok"] = 999;
        });

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => DeepSeekV41Config.Parse(json));

        Assert.Contains("compress_ratios has 2 entries", error.Message);
        Assert.Contains("num_experts_per_tok 999 exceeds n_routed_experts 384", error.Message);
    }

    [Fact]
    public void WrongModelType_IsRejected()
    {
        string json = DeepSeekV41Fixtures.Read("official_config.json").Replace("\"deepseek_v41\"", "\"llama\"", StringComparison.Ordinal);

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => DeepSeekV41Config.Parse(json));

        Assert.Contains("model_type is 'llama'", error.Message);
    }

    [Fact]
    public void MissingRequiredField_NamesTheField()
    {
        string json = DeepSeekV41Fixtures.OfficialConfig(text => text.Remove("hidden_size"));

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => DeepSeekV41Config.Parse(json));

        Assert.Contains("hidden_size", error.Message);
    }

    [Fact]
    public void MalformedJson_ThrowsAClearError()
    {
        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => DeepSeekV41Config.Parse("{ not json"));

        Assert.Contains("not valid JSON", error.Message);
    }
}
