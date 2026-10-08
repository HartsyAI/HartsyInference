using System.Text.Json.Nodes;
using HartsyInference.Core.Exceptions;
using HartsyInference.LLM.DeepSeekV41;
using Xunit;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>The vision tower's section of the V4.1 config: both layouts, the rotary base and its default, and the derived widths.</summary>
public sealed class DeepSeekV41VisionConfigTests
{
    private static DeepSeekV41VisionConfig OfficialVision(Action<JsonObject>? editVision = null)
    {
        JsonObject root = JsonNode.Parse(DeepSeekV41Fixtures.Read("official_config.json"))!.AsObject();
        editVision?.Invoke(root["vision_config"]!.AsObject());
        return DeepSeekV41Config.Parse(root.ToJsonString()).Vision!;
    }

    [Fact]
    public void PinnedOfficialConfig_GivesTheRealTowerAndItsDerivedWidths()
    {
        DeepSeekV41VisionConfig vision = OfficialVision();

        Assert.Equal(new DeepSeekV41VisionConfig(32, 1024, 16, 2816, 14, 3, 10000.0), vision);
        Assert.Equal(64, vision.HeadDim);
        Assert.Equal(588, vision.PatchInputDim);
        Assert.Equal(9216, vision.AlignerInputDim);
        vision.Validate();
    }

    [Fact]
    public void ARopeThetaInTheOfficialLayoutIsRead_AndAnAbsentOneIsTheUpstreamDefault()
    {
        Assert.Equal(500.0, OfficialVision(v => v["rope_theta"] = 500).RopeTheta);
        Assert.Equal(DeepSeekV41VisionConfig.DefaultRopeTheta, OfficialVision(v => v.Remove("rope_theta")).RopeTheta);
        Assert.Equal(10000.0, DeepSeekV41VisionConfig.DefaultRopeTheta);
    }

    [Fact]
    public void FlatMlxConfig_ReadsItsRopeTheta()
    {
        DeepSeekV41VisionConfig vision = DeepSeekV41Config.Parse(DeepSeekV41Fixtures.Read("mlx_config.json")).Vision!;
        Assert.Equal(10000.0, vision.RopeTheta);

        JsonObject edited = JsonNode.Parse(DeepSeekV41Fixtures.Read("mlx_config.json"))!.AsObject();
        edited["vision_rope_theta"] = 2500.0;
        Assert.Equal(2500.0, DeepSeekV41Config.Parse(edited.ToJsonString()).Vision!.RopeTheta);
        edited.Remove("vision_rope_theta");
        Assert.Equal(DeepSeekV41VisionConfig.DefaultRopeTheta, DeepSeekV41Config.Parse(edited.ToJsonString()).Vision!.RopeTheta);
    }

    [Fact]
    public void ATextOnlyConfigHasNoVision()
    {
        JsonObject root = JsonNode.Parse(DeepSeekV41Fixtures.Read("official_config.json"))!.AsObject();
        root.Remove("vision_config");

        Assert.Null(DeepSeekV41Config.Parse(root.ToJsonString()).Vision);
    }

    [Theory]
    [InlineData(0, 1024, 16, 2816, 14, 3, 10000.0)]
    [InlineData(32, 1024, 16, 2816, 14, 0, 10000.0)]
    [InlineData(32, 1000, 16, 2816, 14, 3, 10000.0)]
    [InlineData(32, 1008, 16, 2816, 14, 3, 10000.0)]
    [InlineData(32, 1024, 16, 2816, 14, 3, 0.0)]
    [InlineData(32, 1024, 16, 2816, 14, 3, double.PositiveInfinity)]
    public void Validate_RefusesATowerThatCannotRun(int layers, int hidden, int heads, int inter, int patch, int ratio, double theta) =>
        Assert.Throws<HartsyInferenceException>(() => new DeepSeekV41VisionConfig(layers, hidden, heads, inter, patch, ratio, theta).Validate());
}
