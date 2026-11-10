using HartsyInference.ModelAssets.Registry;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

public sealed class MiniMaxH3ImageDetectionTests
{
    private static readonly string[] SharedKeys = ["video_patch_proj.weight", "condition_proj.weight", "blocks.0.attn.qkv_proj.weight"];
    private static readonly string[] AudioKeys = ["audio_patch_proj.weight", "final_layer.audio_out.weight"];

    [Fact]
    public void VideoWithAudioStaysJointH3()
    {
        Assert.Equal(ModelArchitecture.MiniMaxH3, ModelArchitectureDetector.Detect(Keys(string.Empty, SharedKeys, AudioKeys)));
    }

    [Fact]
    public void SameKeysWithoutAudioResolveToImage()
    {
        Assert.Equal(ModelArchitecture.MiniMaxH3Image, ModelArchitectureDetector.Detect(Keys(string.Empty, SharedKeys)));
    }

    [Theory]
    [InlineData("diffusion_model.")]
    [InlineData("model.diffusion_model.")]
    public void WrappedPrefixesResolveToImage(string prefix)
    {
        Assert.Equal(ModelArchitecture.MiniMaxH3Image, ModelArchitectureDetector.Detect(Keys(prefix, SharedKeys)));
        Assert.Equal(ModelArchitecture.MiniMaxH3, ModelArchitectureDetector.Detect(Keys(prefix, SharedKeys, AudioKeys)));
    }

    [Fact]
    public void ConditionProjectionWithoutVideoProjectionMatchesNeither()
    {
        ModelArchitecture arch = ModelArchitectureDetector.Detect(["condition_proj.weight", "blocks.0.attn.qkv_proj.weight"]);

        Assert.NotEqual(ModelArchitecture.MiniMaxH3Image, arch);
        Assert.NotEqual(ModelArchitecture.MiniMaxH3, arch);
    }

    private static IEnumerable<string> Keys(string prefix, params string[][] groups) =>
        groups.SelectMany(group => group).Select(key => prefix + key);
}
