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
    public void ExpertKeys_GateComesBeforeUp()
    {
        Assert.Equal(["gate_proj", "up_proj", "down_proj"], Glm5NextWeightNames.ExpertProjections);
        Assert.Equal("model.language_model.layers.5.mlp.experts.7.gate_proj.weight", Glm5NextWeightNames.Expert(5, 7, "gate_proj"));
    }
}
