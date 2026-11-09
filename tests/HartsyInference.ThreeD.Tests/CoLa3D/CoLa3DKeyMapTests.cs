using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.CoLa3D;
using Xunit;

namespace HartsyInference.ThreeD.Tests.CoLa3D;

public sealed class CoLa3DKeyMapTests
{
    [Fact]
    public void FreeQ20Fixture_EveryKeyMapsAndMatchesConfigShape()
    {
        AssertFixtureMaps("cola3d_free_q20_keys.txt", CoLa3DTestSupport.FreeQ20(), expectedTensors: 251);
    }

    [Fact]
    public void MaskQ20Fixture_EveryKeyMapsAndMatchesConfigShape()
    {
        AssertFixtureMaps("cola3d_mask_q20_keys.txt", CoLa3DTestSupport.MaskQ20(), expectedTensors: 273);
    }

    [Fact]
    public void PointQ20Fixture_EveryKeyMapsAndMatchesConfigShape()
    {
        AssertFixtureMaps("cola3d_point_q20_keys.txt", CoLa3DTestSupport.PointQ20(), expectedTensors: 256);
    }

    [Fact]
    public void FreeQ20Fixture_MapsOntoEveryExpectedParameterExactlyOnce()
    {
        CoLa3DConfig config = CoLa3DTestSupport.FreeQ20();
        Dictionary<string, long[]> fixture = CoLa3DTestSupport.ReadFixture("cola3d_free_q20_keys.txt");
        List<string> mapped = fixture.Keys.Select(k => CoLa3DKeyMap.TryMap(k)!).ToList();

        Assert.Equal(mapped.Count, mapped.Distinct(StringComparer.Ordinal).Count());
        Assert.Equal(
            CoLa3DKeyMap.ExpectedShapes(config).Keys.Order(StringComparer.Ordinal),
            mapped.Order(StringComparer.Ordinal));
    }

    [Theory]
    [InlineData("pat.blocks.0.q_to_qkv.running_mean")]
    [InlineData("pat.blocks.0.unknown_module.weight")]
    [InlineData("pat.unknown.weight")]
    [InlineData("completely.unrelated")]
    public void UnknownKeys_AreNotMapped(string key)
    {
        Assert.Null(CoLa3DKeyMap.TryMap(key));
    }

    [Fact]
    public void BlockKeys_CarryTheLayerIndexIntoTheParameterName()
    {
        Assert.Equal("backbone.blocks.4.qSelfQkv.weight", CoLa3DKeyMap.TryMap("pat.blocks.4.q_to_qkv.weight"));
        Assert.Equal("backbone.blocks.5.pWriteQkNorm.gamma", CoLa3DKeyMap.TryMap("pat.blocks.5.q2p_q_rms_norm.gamma"));
        Assert.Equal("backbone.blocks.1.pFfnUp.bias", CoLa3DKeyMap.TryMap("pat.blocks.1.p_mlp.mlp.0.bias"));
        Assert.Equal("maskPrompt.conv.6.weight", CoLa3DKeyMap.TryMap("mask_prompt_encoder.encoder.6.weight"));
        Assert.Equal("query.learnable", CoLa3DKeyMap.TryMap("pat.query_embed"));
    }

    [Fact]
    public void Remap_RejectsAKeyTheConfigDoesNotUse()
    {
        using Tensor stray = new(new TensorShape(4), DType.F32);
        Dictionary<string, Tensor> torch = new(StringComparer.Ordinal) { ["mask_prompt_encoder.encoder.0.weight"] = stray };

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => CoLa3DKeyMap.Remap(CoLa3DTestSupport.FreeQ20(), torch));
        Assert.Contains("not used by this configuration", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Remap_RejectsAWrongShape()
    {
        CoLa3DConfig config = CoLa3DTestSupport.Tiny(CoLa3DConfig.QuerySourceKind.Fixed);
        Dictionary<string, Tensor> weights = CoLa3DTestSupport.RandomWeights(config, seed: 1);
        try
        {
            Tensor bad = new(new TensorShape(3, 3), DType.F32);
            Dictionary<string, Tensor> torch = new(StringComparer.Ordinal) { ["pat.latent_proj.weight"] = bad };

            HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => CoLa3DKeyMap.Remap(config, torch));
            Assert.Contains("expected", error.Message, StringComparison.Ordinal);
            bad.Dispose();
        }
        finally
        {
            CoLa3DTestSupport.Dispose(weights.Values);
        }
    }

    [Fact]
    public void ValidateNamed_ReportsMissingParameters()
    {
        CoLa3DConfig config = CoLa3DTestSupport.Tiny(CoLa3DConfig.QuerySourceKind.Fixed);
        Dictionary<string, Tensor> weights = CoLa3DTestSupport.RandomWeights(config, seed: 2);
        try
        {
            weights.Remove("backbone.fusion.weight");
            HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => CoLa3DKeyMap.ValidateNamed(config, weights));
            Assert.Contains("missing", error.Message, StringComparison.Ordinal);
        }
        finally
        {
            CoLa3DTestSupport.Dispose(weights.Values);
        }
    }

    private static void AssertFixtureMaps(string fixture, CoLa3DConfig config, int expectedTensors)
    {
        Dictionary<string, long[]> tensors = CoLa3DTestSupport.ReadFixture(fixture);
        IReadOnlyDictionary<string, long[]> expected = CoLa3DKeyMap.ExpectedShapes(config);
        Assert.Equal(expectedTensors, tensors.Count);

        List<string> unmapped = tensors.Keys.Where(k => CoLa3DKeyMap.TryMap(k) is null).Order(StringComparer.Ordinal).ToList();
        Assert.True(unmapped.Count == 0, $"Unmapped keys in {fixture}: {string.Join(", ", unmapped.Take(10))}");

        foreach (KeyValuePair<string, long[]> entry in tensors)
        {
            string name = CoLa3DKeyMap.TryMap(entry.Key)!;
            Assert.True(expected.ContainsKey(name), $"{entry.Key} maps to {name}, which the config does not use.");
            Assert.Equal(expected[name], entry.Value);
        }

        Assert.Equal(expected.Count, tensors.Count);
    }
}
