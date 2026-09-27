using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Video;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Covers how a distilled checkpoint under a dev family id reaches the distilled sampling contract, now through
/// <see cref="LtxVideo2Variants"/>. Every failure here is silent: a routing miss runs the dev contract on a distilled
/// checkpoint (or vice versa) and still produces plausible video, just the wrong one.</summary>
public sealed class LtxVideo25DistilledRoutingTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("ltx-routing-").FullName;

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    [Theory]
    [InlineData("ltx-2.5")]
    [InlineData("ltx-2")]
    [InlineData("ltx-2.3")]
    [InlineData("ltx-video-2")]
    [InlineData("lightricks-ltx-video-2")]
    public void DistilledFilenameRoutesEveryDevFamilyId(string familyId)
    {
        string path = Touch("ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors");
        Assert.Equal(LtxVideo2Variants.DistilledFamilyId, ModelCapabilities.VideoFamilyIdFor(Spec(familyId, path)));
    }

    [Fact]
    public void NonDistilledFilenameDoesNotRoute()
    {
        string path = Touch("ltx-2.5-22b-dev-transformer-int8_lean_convrot.safetensors");
        Assert.Equal("ltx-2.5", ModelCapabilities.VideoFamilyIdFor(Spec("ltx-2.5", path)));
    }

    /// <summary>A hint is the definitive signal the file name only guesses at: it routes a renamed distilled build.</summary>
    [Fact]
    public void AnExplicitVariantRoutesWithoutAFilenameToken()
    {
        string path = Touch("renamed.safetensors");
        Assert.Equal(LtxVideo2Variants.DistilledFamilyId, ModelCapabilities.VideoFamilyIdFor(Spec("ltx-2.5:distilled", path)));
        Assert.Equal(LtxVideo2Variants.DistilledFamilyId,
            ModelCapabilities.VideoFamilyIdFor(Spec("ltx-2.5", path) with { Variant = "distilled" }));
    }

    [Fact]
    public void RoutingScansDirectoryContents()
    {
        // Distilled runs stage a directory (transformer-only distilled file + sibling VAEs); the dir name says
        // nothing, so the scan must look at the contained safetensors names.
        Touch("foo-distilled-transformer.safetensors");
        Touch("video-vae.safetensors");
        Assert.Equal(LtxVideo2Variants.DistilledFamilyId, ModelCapabilities.VideoFamilyIdFor(Spec("ltx-2.5", _dir)));
    }

    [Fact]
    public void DevDirectoryDoesNotRoute()
    {
        Touch("ltx-2.5-22b-dev-transformer.safetensors");
        Assert.Equal("ltx-2.5", ModelCapabilities.VideoFamilyIdFor(Spec("ltx-2.5", _dir)));
    }

    [Fact]
    public void RoutingLeavesForeignFamiliesAlone()
    {
        string path = Touch("some-distilled-model.safetensors");
        Assert.Equal("wan", ModelCapabilities.VideoFamilyIdFor(Spec("wan", path)));
        Assert.Equal("hunyuan-video", ModelCapabilities.VideoFamilyIdFor(Spec("hunyuan-video", path)));
        // The distilled id itself passes through untouched (already routed).
        Assert.Equal(LtxVideo2Variants.DistilledFamilyId,
            ModelCapabilities.VideoFamilyIdFor(Spec(LtxVideo2Variants.DistilledFamilyId, path)));
    }

    [Fact]
    public void DistilledContractOnAPre25CheckpointSkipsTwoStage()
    {
        // 2.0/2.3 distilled builds exist; the shared 8-step schedule applies but the x2 upsampler is a 2.5 model.
        Diffusion.Models.Denoisers.LtxVideo2Config detected23 =
            Diffusion.Models.Denoisers.LtxVideo2Config.V23;
        Diffusion.Models.Denoisers.LtxVideo2Config gated = LtxVideo2Recipe.ApplyDistilledContract(detected23);
        Assert.NotNull(gated.FixedSigmas);
        Assert.Equal(1.0f, gated.GuidanceScale);
        Assert.False(gated.TwoStage);

        Diffusion.Models.Denoisers.LtxVideo2Config detected25 =
            Diffusion.Models.Denoisers.LtxVideo2Config.V25;
        Assert.True(LtxVideo2Recipe.ApplyDistilledContract(detected25).TwoStage);
    }

    [Fact]
    public void RoutedSpecResolvesTheDistilledDefaults()
    {
        // Routing only matters if the capability query lands on the distilled registration's defaults — a rename of
        // either registration breaks the chain silently.
        string path = Touch("ltx-2.5-22b-distilled-transformer.safetensors");
        VideoDefaults defaults = ModelCapabilities.VideoDefaultsFor(Spec("ltx-2.5", path));
        Assert.Equal(8, defaults.Steps);
        Assert.Equal(1.0f, defaults.CfgScale);
    }

    private static ModelSpec Spec(string requested, string path) =>
        new ModelSpec { Requested = requested, Modality = Modality.Video, LocalPath = path };

    private string Touch(string fileName)
    {
        string path = Path.Combine(_dir, fileName);
        File.WriteAllText(path, "");
        return path;
    }
}
