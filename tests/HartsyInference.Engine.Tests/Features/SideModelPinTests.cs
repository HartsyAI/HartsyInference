using HartsyInference.Engine;
using HartsyInference.Engine.Features;

namespace HartsyInference.Engine.Tests.Features;

/// <summary>A side model found on disk must be the pinned build, and the int8 Fibo 2.0 encoder must not share the path of a
/// same-named file from another build.</summary>
public sealed class SideModelPinTests
{
    private const string Pinned = "32de113cbf8282b02cbff0794b637b9d6676da09314a1db5a30bcea9bf4ee01c";
    private const string Other = "5ebf6c30aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    [Fact]
    public void PinnedMatch_IsAccepted_CaseInsensitively()
    {
        ModelFileLocator.RequirePinnedBuild("x.safetensors", SideModels.Fibo2Qwen3VL4BInt8, "text encoder", Pinned.ToUpperInvariant());
    }

    [Fact]
    public void WrongHash_IsRefused_AndNamesBothHashes()
    {
        InvalidDataException ex = Assert.Throws<InvalidDataException>(() =>
            ModelFileLocator.RequirePinnedBuild("models/x.safetensors", SideModels.Fibo2Qwen3VL4BInt8, "text encoder", Other));

        Assert.Contains(Other, ex.Message);
        Assert.Contains(Pinned, ex.Message);
        Assert.Contains("Refusing to load it", ex.Message);
    }

    [Fact]
    public void AssetWithoutAPin_IsAcceptedAnyHash()
    {
        ModelAsset unpinned = SideModels.Fibo2Qwen3VL4BInt8 with { Sha256 = null };

        ModelFileLocator.RequirePinnedBuild("x.safetensors", unpinned, "text encoder", Other);
    }

    [Fact]
    public void Int8Encoder_LandsBesideTheSameNamedFileNotOnIt()
    {
        // The earlier target name was fibo2/qwen3vl_4b_int8_convrot.safetensors; a file there from another build satisfied it.
        Assert.NotEqual("fibo2/qwen3vl_4b_int8_convrot.safetensors", SideModels.Fibo2Qwen3VL4BInt8.TargetName);
        Assert.Equal(Pinned, SideModels.Fibo2Qwen3VL4BInt8.Sha256);
    }
}
