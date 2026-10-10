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
    public void DistilledContractOnAPre25CheckpointSkipsTwoStage()
    {
        // 2.0/2.3 distilled builds exist; the shared 8-step schedule applies but the x2 upsampler is a 2.5 model.
        Diffusion.Models.Denoisers.LtxVideo2Config detected23 =
            Diffusion.Models.Denoisers.LtxVideo2Config.V23;
        Diffusion.Models.Denoisers.LtxVideo2Config gated23 = LtxVideo2Recipe.ApplyDistilledContract(detected23);
        Assert.NotNull(gated23.FixedSigmas);
        Assert.Equal(1.0f, gated23.GuidanceScale);
        Assert.False(gated23.TwoStage);
        // Asked for through numerics.ltx2TwoStage, two-stage is still refused on 2.3.
        Assert.NotNull(LtxVideo2Recipe.TwoStageRefusal(gated23, distilled: true));

        // Two-stage is opt-in, so the contract leaves it off on 2.5 too; asked for, it runs there.
        Diffusion.Models.Denoisers.LtxVideo2Config gated25 =
            LtxVideo2Recipe.ApplyDistilledContract(Diffusion.Models.Denoisers.LtxVideo2Config.V25);
        Assert.False(gated25.TwoStage);
        Assert.Null(LtxVideo2Recipe.TwoStageRefusal(gated25, distilled: true));

        // A 2.5 checkpoint outside the distilled family has no documented two-stage schedule.
        Assert.NotNull(LtxVideo2Recipe.TwoStageRefusal(Diffusion.Models.Denoisers.LtxVideo2Config.V25, distilled: false));
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
