using System.Text.Json;
using HartsyInference.LLM.Maxwell;
using HartsyInference.LLM.Transformer;
using Xunit;

namespace HartsyInference.LLM.Tests.Maxwell;

/// <summary>The key map against the release's layout: its counts come from the real <c>model.safetensors.index.json</c> (33,471 main keys and 799
/// MTP keys at commit 5dd7c6f), and an optional check reads that index itself.</summary>
public sealed class MaxwellWeightMapTests
{
    private static readonly MaxwellConfig Config = MaxwellConfig.Parse(MaxwellConfigTests.Json());

    [Fact]
    public void ExpectedKeys_MatchTheReleaseCounts()
    {
        Assert.Equal(33_471, MaxwellWeightMap.ExpectedKeys(Config).Count);
        Assert.Equal(799, MaxwellWeightMap.ExpectedMtpKeys(Config).Count);
        Assert.Equal(33_471, MaxwellWeightMap.ExpectedKeys(Config).Distinct().Count());
    }

    [Fact]
    public void Check_SkipsMtpUnlessAskedAndFoldsScaleCompanions()
    {
        List<string> keys = [.. MaxwellWeightMap.ExpectedKeys(Config), .. MaxwellWeightMap.ExpectedMtpKeys(Config)];
        keys.Add("model.layers.3.block_sparse_moe.experts.0.w1.weight_scale_inv");
        keys.Add("model.layers.0.self_attn.q_proj.weight_scale_inv");

        CheckpointKeyCoverage skipped = MaxwellWeightMap.Check(Config, keys);
        CheckpointKeyCoverage loaded = MaxwellWeightMap.Check(Config, keys, mtp: true);

        skipped.RequireComplete();
        loaded.RequireComplete();
        Assert.Equal((33_471, 2, 799), (skipped.Mapped, skipped.ScaleCompanions, skipped.SkippedMtp));
        Assert.Equal((33_471 + 799, 2, 0), (loaded.Mapped, loaded.ScaleCompanions, loaded.SkippedMtp));
    }

    [Fact]
    public void Check_NamesAnUnmappedKeyAndAMissingParameter()
    {
        List<string> keys = [.. MaxwellWeightMap.ExpectedKeys(Config)];
        keys.Remove("model.layers.5.self_attn.indexer.linear_k_proj.weight");
        keys.Add("model.layers.5.self_attn.rotary_emb.inv_freq");

        CheckpointKeyCoverage coverage = MaxwellWeightMap.Check(Config, keys);

        Assert.Equal(["model.layers.5.self_attn.rotary_emb.inv_freq"], coverage.Unmapped);
        Assert.Equal(["model.layers.5.self_attn.indexer.linear_k_proj.weight"], coverage.Missing);
        Exception error = Assert.Throws<InvalidDataException>(coverage.RequireComplete);
        Assert.Contains("Maxwell", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void MtpDecoder_HasDenseAttentionAndNoIndexer()
    {
        IReadOnlyList<string> mtp = MaxwellWeightMap.ExpectedMtpKeys(Config);

        Assert.DoesNotContain(mtp, static k => k.Contains("indexer", StringComparison.Ordinal));
        Assert.Contains("mtp_layers.0.decoder_layer.block_sparse_moe.expert_bias", mtp);
        Assert.All(mtp, static k => Assert.True(MaxwellWeightMap.IsMtpKey(k)));
    }

    [RealWeightsFact("MAXWELL_INDEX_JSON")]
    [Trait("Category", "Integration")]
    public void RealIndex_MapsKeyForKey()
    {
        using JsonDocument index = JsonDocument.Parse(File.ReadAllText(Environment.GetEnvironmentVariable("MAXWELL_INDEX_JSON")!));
        IEnumerable<string> keys = index.RootElement.GetProperty("weight_map").EnumerateObject().Select(static p => p.Name);

        CheckpointKeyCoverage coverage = MaxwellWeightMap.Check(Config, keys, mtp: true);

        coverage.RequireComplete();
        Assert.Equal(33_471 + 799, coverage.Mapped);
    }
}
