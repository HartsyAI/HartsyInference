using HartsyInference.LLM.Glm5Next;
using Xunit;

namespace HartsyInference.LLM.Tests.Glm5Next;

/// <summary>The engine's key names against the real BF16 and FP8 index key sets (header-only fixtures, no weights).</summary>
public sealed class Glm5NextWeightCoverageTests
{
    private static readonly Glm5NextConfig Config = Glm5NextHfConfig.Parse(Glm5NextIndexFixture.ConfigJson());

    [Fact]
    public void EveryKeyOfTheBf16Index_IsNamed_AndNothingElse()
    {
        HashSet<string> index = Glm5NextIndexFixture.Load("sage-s21-flash-bf16-index-keys.txt");
        HashSet<string> named = Glm5NextWeightNames.AllKeys(Config, includeVision: true).ToHashSet();
        Assert.Equal(38770, index.Count);
        Assert.Empty(index.Except(named).Take(5));
        Assert.Empty(named.Except(index).Take(5));
    }

    [Fact]
    public void TheFp8Index_IsTheBf16Keys_PlusScaleCompanionsOfExactlyTheFp8Weights()
    {
        HashSet<string> fp8 = Glm5NextIndexFixture.Load("sage-s21-flash-fp8-index-keys.txt");
        HashSet<string> expected = Glm5NextWeightNames.AllKeys(Config, includeVision: true).ToHashSet();
        foreach (string k in expected.ToArray())
            if (Glm5NextWeightNames.IsFp8Weight(Config, k)) expected.Add(Glm5NextWeightNames.ScaleOf(k));
        Assert.Equal(76108, fp8.Count);
        Assert.Empty(fp8.Except(expected).Take(5));
        Assert.Empty(expected.Except(fp8).Take(5));
    }

    [Fact]
    public void MtpKeys_AreThoseOfLayer45_AndNoDecoderLayerOwnsSharedHead()
    {
        HashSet<string> index = Glm5NextIndexFixture.Load("sage-s21-flash-bf16-index-keys.txt");
        string[] mtp = index.Where(k => Glm5NextWeightNames.IsMtpKey(Config, k)).ToArray();
        Assert.Equal(Glm5NextWeightNames.LayerKeys(Config, 45).Count(), mtp.Length);
        Assert.Contains(Glm5NextWeightNames.Layer(45) + "shared_head.norm.weight", mtp);
        Assert.DoesNotContain(mtp, k => k.Contains(".hc_"));
        Assert.DoesNotContain(index, k => k.Contains("shared_head") && !Glm5NextWeightNames.IsMtpKey(Config, k));
    }

    [Fact]
    public void TinyReferenceCheckpoint_KeysAreExactlyTheNamedKeys()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Glm5Next", "tiny");
        Glm5NextConfig tiny = Glm5NextHfConfig.Parse(File.ReadAllText(Path.Combine(dir, "config.json")));
        using System.Text.Json.JsonDocument index = System.Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "model.safetensors.index.json")));
        HashSet<string> keys = index.RootElement.GetProperty("weight_map").EnumerateObject().Select(p => p.Name).ToHashSet();
        HashSet<string> named = Glm5NextWeightNames.AllKeys(tiny, includeVision: false).ToHashSet();
        Assert.Empty(keys.Except(named));
        Assert.Empty(named.Except(keys));
        Assert.Contains(tiny.LayerKinds, k => k == Glm5NextAttentionKind.Kda);
        Assert.Contains(tiny.LayerKinds, k => k == Glm5NextAttentionKind.Dsa);
    }

    [Fact]
    public void KeysOfLayersTheConfigDoesNotName_AreNotFp8OrMtp()
    {
        // A checkpoint key can name any layer; the predicates answer false for a number the config does not have instead of throwing.
        Assert.False(Glm5NextWeightNames.IsFp8Weight(Config, "model.language_model.layers.99.mlp.gate_proj.weight"));
        Assert.False(Glm5NextWeightNames.IsMtpKey(Config, "model.language_model.layers.99.enorm.weight"));
        Assert.False(Glm5NextWeightNames.IsFp8Weight(Config, "model.language_model.layers.x.mlp.gate_proj.weight"));
        Assert.False(Glm5NextWeightNames.IsMtpKey(Config, "model.language_model.embed_tokens.weight"));
    }

    [Fact]
    public void ExpertKeys_GateComesBeforeUp()
    {
        Assert.Equal(["gate_proj", "up_proj", "down_proj"], Glm5NextWeightNames.ExpertProjections);
        Assert.Equal("model.language_model.layers.5.mlp.experts.7.gate_proj.weight", Glm5NextWeightNames.Expert(5, 7, "gate_proj"));
    }
}
