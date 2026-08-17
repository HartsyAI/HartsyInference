using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Video;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Covers the filename remap that routes distilled checkpoints under a dev family id to the distilled
/// sampling contract. Every failure here is silent: a routing miss runs the dev contract on a distilled
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
    public void DistilledFilenameRemapsEveryDevFamilyId(string familyId)
    {
        string path = "/models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors";
        Assert.Equal("ltx-2.5-distilled", LtxVideo2DistilledRouting.RemapFamilyId(familyId, path));
    }

    [Fact]
    public void NonDistilledFilenameDoesNotRemap()
    {
        Assert.Equal("ltx-2.5", LtxVideo2DistilledRouting.RemapFamilyId(
            "ltx-2.5", "/models/ltx-2.5-22b-dev-transformer-int8_lean_convrot.safetensors"));
        Assert.Equal("ltx-2.5", LtxVideo2DistilledRouting.RemapFamilyId("ltx-2.5", null));
        Assert.Equal("ltx-2.5", LtxVideo2DistilledRouting.RemapFamilyId("ltx-2.5", ""));
    }

    [Fact]
    public void RemapScansDirectoryContents()
    {
        // Distilled runs stage a directory (transformer-only distilled file + sibling VAEs); the dir name says
        // nothing, so the scan must look at the contained safetensors names.
        File.WriteAllText(Path.Combine(_dir, "foo-distilled-transformer.safetensors"), "");
        File.WriteAllText(Path.Combine(_dir, "video-vae.safetensors"), "");
        Assert.Equal("ltx-2.5-distilled", LtxVideo2DistilledRouting.RemapFamilyId("ltx-2.5", _dir));
    }

    [Fact]
    public void DevDirectoryDoesNotRemap()
    {
        File.WriteAllText(Path.Combine(_dir, "ltx-2.5-22b-dev-transformer.safetensors"), "");
        Assert.Equal("ltx-2.5", LtxVideo2DistilledRouting.RemapFamilyId("ltx-2.5", _dir));
    }

    [Fact]
    public void RemapLeavesForeignFamiliesAlone()
    {
        string path = "/models/some-distilled-model.safetensors";
        Assert.Equal("wan", LtxVideo2DistilledRouting.RemapFamilyId("wan", path));
        Assert.Equal("hunyuan-video", LtxVideo2DistilledRouting.RemapFamilyId("hunyuan-video", path));
        // The distilled id itself passes through untouched (already routed).
        Assert.Equal("ltx-2.5-distilled", LtxVideo2DistilledRouting.RemapFamilyId("ltx-2.5-distilled", path));
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
    public void RegistryServesTheDistilledContractForARemappedId()
    {
        // The remap only matters if the registry resolves the remapped id to the distilled defaults — a rename
        // of either registration breaks the chain silently.
        string remapped = LtxVideo2DistilledRouting.RemapFamilyId(
            "ltx-2.5", "/models/ltx-2.5-22b-distilled-transformer.safetensors");
        VideoDefaults? defaults = VideoRecipeRegistry.Resolve(remapped)?.Defaults;
        Assert.NotNull(defaults);
        Assert.Equal(8, defaults!.Steps);
        Assert.Equal(1.0f, defaults.CfgScale);
    }
}
