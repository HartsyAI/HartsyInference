using HartsyInference.LLM.Glm5Next;
using Xunit;

namespace HartsyInference.LLM.Tests.Glm5Next;

/// <summary>The glm5_next config reader against the published Sage-S2.1-Flash <c>config.json</c> (BF16 pin eacc61b), and its refusals.</summary>
public sealed class Glm5NextHfConfigTests
{
    [Fact]
    public void PublishedConfig_ParsesToTheCheckpointsShape()
    {
        Glm5NextConfig c = Glm5NextHfConfig.Parse(Glm5NextIndexFixture.ConfigJson());
        Assert.Equal((4096, 154880, 45, 1, 64), (c.Hidden, c.Vocab, c.Layers, c.MtpLayers, c.Heads));
        Assert.Equal(11, c.LayerKinds.Count(k => k == Glm5NextAttentionKind.Dsa));
        for (int l = 0; l < c.Layers; l++)
            Assert.Equal(l % 4 == 3 ? Glm5NextAttentionKind.Dsa : Glm5NextAttentionKind.Kda, c.LayerKinds[l]);
        Assert.True(c.SparseMlp.Take(3).All(s => !s));
        Assert.True(c.SparseMlp.Skip(3).All(s => s));
        Assert.Equal((128, 4, -5f), (c.KdaHeadDim, c.ConvKernel, c.GateLowerBound));
        Assert.Equal((1536, 512, 256, 256), (c.QLoraRank, c.KvLoraRank, c.QkDim, c.VDim));
        Assert.Equal((32, 128, 2048, 4, true), (c.IndexHeads, c.IndexHeadDim, c.IndexTopK, c.IndexPool, c.IndexAlwaysSelectTail));
        Assert.Equal(512, c.IndexSelectedPools);
        Assert.Equal((288, 8, 2048, 1, 2.5f, true), (c.Experts, c.ExpertsPerToken, c.MoeIntermediate, c.SharedExperts, c.RoutedScalingFactor, c.NormTopKProb));
        Assert.Equal((4, 1e-6f, 20, 1e-5f, 10f), (c.HcMult, c.HcEps, c.HcSinkhornIters, c.RmsNormEps, c.SwigluLimit));
        Assert.Equal([154820, 154827, 154829], c.EosTokenIds);
        Assert.True(c.IsMtpLayer(45) && !c.IsMtpLayer(44) && c.IsDsa(45) && c.IsSparse(45));
        Assert.Equal((154854, 154855), (c.ImageTokenId, c.VideoTokenId));
        Glm5NextVisionConfig v = c.Vision!;
        Assert.Equal((24, 1024, 16, 4096, 14, 2, 2, 4096, 10240), (v.Depth, v.Hidden, v.Heads, v.Intermediate, v.PatchSize, v.TemporalPatchSize,
            v.SpatialMergeSize, v.OutHidden, v.MergerIntermediate));
        Assert.Equal(64, v.HeadDim);
    }

    [Fact]
    public void LaterTransformersSpelling_OfTheDsaLayerType_ReadsTheSame()
    {
        string json = Glm5NextIndexFixture.ConfigJson().Replace("deepseek_sparse_attention", "indexed_attention");
        Glm5NextConfig a = Glm5NextHfConfig.Parse(Glm5NextIndexFixture.ConfigJson()), b = Glm5NextHfConfig.Parse(json);
        Assert.Equal(a.LayerKinds, b.LayerKinds);
    }

    [Theory]
    [InlineData("\"qk_rope_head_dim\": 0", "\"qk_rope_head_dim\": 64", "NoPE")]
    [InlineData("\"index_kpool\": 4", "\"index_kpool\": 3", "multiple of index_kpool")]
    [InlineData("\"n_group\": 1", "\"n_group\": 8", "n_group")]
    [InlineData("\"index_kpool_compress\": true", "\"index_kpool_compress\": false", "index_kpool_compress")]
    [InlineData("\"model_type\": \"glm5_next_text\"", "\"model_type\": \"glm4\"", "glm5_next_text")]
    [InlineData("\"hidden_act\": \"silu\",\n    \"hidden_size\": 4096", "\"hidden_act\": \"gelu\",\n    \"hidden_size\": 4096", "silu")]
    public void ConfigThatTheEngineDoesNotRun_IsRefusedByName(string find, string replace, string expected)
    {
        string json = Glm5NextIndexFixture.ConfigJson();
        Assert.Contains(find, json);
        InvalidDataException e = Assert.Throws<InvalidDataException>(() => Glm5NextHfConfig.Parse(json.Replace(find, replace)));
        Assert.Contains(expected, e.Message);
    }

    [Fact]
    public void SharedIndexerLayer_IsRefused()
    {
        string json = Glm5NextIndexFixture.ConfigJson();
        int at = json.IndexOf("\"indexer_types\": [", StringComparison.Ordinal);
        int full = json.IndexOf("\"full\"", at, StringComparison.Ordinal);
        string edited = json[..full] + "\"shared\"" + json[(full + "\"full\"".Length)..];
        InvalidDataException e = Assert.Throws<InvalidDataException>(() => Glm5NextHfConfig.Parse(edited));
        Assert.Contains("indexer_types", e.Message);
    }

    [Fact]
    public void ScheduleThatDisagreesWithLayerTypes_IsRefused()
    {
        string json = Glm5NextIndexFixture.ConfigJson().Replace("\"full_attn_layers\": [\n        3,", "\"full_attn_layers\": [\n        2,");
        InvalidDataException e = Assert.Throws<InvalidDataException>(() => Glm5NextHfConfig.Parse(json));
        Assert.Contains("disagrees with layer_types", e.Message);
    }
}
