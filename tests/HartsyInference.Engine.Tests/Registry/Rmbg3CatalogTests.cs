using System.Text.RegularExpressions;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Variants;

namespace HartsyInference.Engine.Tests.Registry;

public sealed class Rmbg3CatalogTests
{
    private static readonly Regex Sha256 = new("^[0-9a-f]{64}$", RegexOptions.Compiled);

    private static ResolvedModelVariant VariantFor(string hint) =>
        ModelVariantResolver.Resolve(Fibo2Variants.Catalog, new ModelVariantEvidence(null, [hint]));

    [Theory]
    [InlineData("rmbg3", "rmbg3_bf16.safetensors", "Fibo2Qwen3VL4B")]
    [InlineData("rmbg3-int8", "rmbg3_int8_convrot.safetensors", "Fibo2Qwen3VL4BInt8")]
    public void Entry_IsACliDrivableVisionEntryOnTheBriaRepackWithTheFiboSideModels(string id, string file, string encoder)
    {
        CatalogEntry? entry = ModelCatalog.Find(id);

        Assert.NotNull(entry);
        Assert.Equal(Modality.Vision, entry.Modality);
        Assert.True(entry.CliDrivable);
        ModelAsset transformer = entry.Assets[0];
        Assert.Equal("briaai/RMBG-3.0-Comfy", transformer.Repo);
        Assert.Equal("diffusion_models/" + file, transformer.RepoPath);
        ModelAsset expectedEncoder = encoder == "Fibo2Qwen3VL4B" ? SideModels.Fibo2Qwen3VL4B : SideModels.Fibo2Qwen3VL4BInt8;
        Assert.Contains(expectedEncoder, entry.Assets);
        Assert.Contains(SideModels.Fibo2Vae, entry.Assets);
        foreach (ModelAsset asset in entry.Assets)
            Assert.Matches(Sha256, asset.Sha256!);
        Assert.Equal(entry.Assets.Count, entry.Assets.Select(a => Path.Combine(a.TargetSubdir, a.FileName))
            .Distinct(StringComparer.OrdinalIgnoreCase).Count());
        Assert.True(new Fibo2Recipe().Matches(id));
    }

    [Theory]
    [InlineData("rmbg3")]
    [InlineData("rmbg3-int8")]
    public void CatalogId_ResolvesTheRmbg3VariantWithFourStepGuidanceFreeDefaults(string id)
    {
        ResolvedModelVariant variant = VariantFor(id);

        Assert.True(variant.Is(Fibo2Variants.Rmbg3));
        ImageDefaults defaults = new Fibo2Recipe().DefaultsFor(variant);
        Assert.Equal(4, defaults.Steps);
        Assert.Equal(1.0f, defaults.CfgScale);
        Assert.Null(defaults.SigmaShift);
    }

    [Fact]
    public void OtherFiboBuilds_KeepTheirVariantAndDefaults()
    {
        Assert.True(VariantFor("edit").Is(Fibo2Variants.EditTurbo));
        Assert.True(VariantFor("fibo2-dev").Is(Fibo2Variants.Dev));
        Assert.Same(Fibo2Recipe.EditTurboDefaults, new Fibo2Recipe().DefaultsFor(VariantFor("edit")));
    }

    [Fact]
    public void DynamicShift_MatchesDiffusersCalculateShiftAtItsAnchorsAndAnOffRatioSize()
    {
        Assert.Equal(1.15, Fibo2Recipe.DynamicShift(1024, 1024), 9);
        Assert.Equal(0.5, Fibo2Recipe.DynamicShift(256, 256), 9);
        Assert.Equal(0.5 + (2816 - 256) * 0.65 / 3840, Fibo2Recipe.DynamicShift(1024, 704), 9);
    }

    [Fact]
    public void Handles_OnlyTheRmbg3Entries_NotRmbg14()
    {
        ModelSpec Spec(string id) => new() { Requested = id, Modality = Modality.Vision, Catalog = ModelCatalog.Find(id) };

        Assert.True(Rmbg3Matte.Handles(Spec("rmbg3")));
        Assert.True(Rmbg3Matte.Handles(Spec("rmbg3-int8")));
        Assert.False(Rmbg3Matte.Handles(Spec("rmbg")));
        Assert.False(Rmbg3Matte.Handles(Spec("birefnet-matting")));
    }

    [Theory]
    [InlineData("rmbg3")]
    [InlineData("rmbg3-int8")]
    public void CatalogFeatures_FlagTheRmbg3EntriesAsMatteWithAGeneratorReleaseFirst(string id)
    {
        CatalogFeatures features = ModelCatalog.Find(id)!.Features;

        Assert.True(features.HasFlag(CatalogFeatures.BackgroundRemoval));
        Assert.True(features.HasFlag(CatalogFeatures.AlphaMatte));
        Assert.True(features.HasFlag(CatalogFeatures.ReleasesGeneratorFirst));

        ImageRequest request = new() { Prompt = "", RemoveBackground = true, RemoveBackgroundModel = id };
        Assert.Equal(id, BackgroundRemovalStage.ModelIdFor(request));
        Assert.True(BackgroundRemovalStage.NeedsGeneratorReleased(request));
    }

    [Fact]
    public void CatalogFeatures_RmbgFourteenHasNoMatteOrReleaseFlag()
    {
        CatalogFeatures features = ModelCatalog.Find("rmbg")?.Features ?? CatalogFeatures.None;

        Assert.False(features.HasFlag(CatalogFeatures.AlphaMatte));
        Assert.False(features.HasFlag(CatalogFeatures.ReleasesGeneratorFirst));
        Assert.False(BackgroundRemovalStage.NeedsGeneratorReleased(new ImageRequest { Prompt = "", RemoveBackground = true }));
    }

    [Fact]
    public void AlphaFromPicture_UsesPilIntegerLumaAndKeepsTheSourceSize()
    {
        byte[] rgb = [255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0];

        byte[] same = Rmbg3Matte.AlphaFromPicture(rgb, 2, 2, 2, 2);
        Assert.Equal(new byte[] { 255, 0, 76, 150 }, same);

        byte[] grey = new byte[8 * 8 * 3];
        Array.Fill(grey, (byte)128);
        byte[] resized = Rmbg3Matte.AlphaFromPicture(grey, 8, 8, 6, 4);
        Assert.Equal(24, resized.Length);
        Assert.All(resized, value => Assert.Equal(128, value));
    }

    [Fact]
    public void RequireInstruction_AcceptsBlankTheFixedTextAndItsJsonFormOnly()
    {
        Rmbg3Matte.RequireInstruction("");
        Rmbg3Matte.RequireInstruction(Rmbg3Matte.Instruction);
        Rmbg3Matte.RequireInstruction(System.Text.Json.JsonSerializer.Serialize(new { edit_instruction = Rmbg3Matte.Instruction }));

        Assert.Throws<NotSupportedException>(() => Rmbg3Matte.RequireInstruction("remove the background"));
        Assert.Throws<NotSupportedException>(() => Rmbg3Matte.RequireInstruction("{\"edit_instruction\":\"make it red\"}"));
    }
}
