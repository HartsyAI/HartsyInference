using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Variants;
using HartsyInference.ModelAssets.Checkpoints;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Anima, HiDream-I1 and Boogu-Image tell their distilled builds apart by file name, and each build resolves
/// to its own sampling defaults. A wrong match is silent: a distilled file runs base numbers or the reverse.</summary>
public sealed class ImageFamilyVariantDefaultsTests
{
    [Theory]
    [InlineData("anima-base-v1.0.safetensors", "base")]
    [InlineData("anima-aesthetic-v1.1.safetensors", "base")]
    [InlineData("anima-turbo-v1.0.safetensors", "turbo")]
    [InlineData("anima-turbo-v1.1.safetensors", "turbo")]
    public void Anima_VariantResolvesFromFileName(string fileName, string expected) =>
        Assert.Equal(expected, Classify(AnimaVariants.Catalog, fileName).Id);

    [Fact]
    public void Anima_BaseUsesCfgAndTurboIsGuidanceFree()
    {
        AnimaRecipe recipe = new();
        ImageDefaults baseDefaults = recipe.DefaultsFor(Classify(AnimaVariants.Catalog, "anima-base-v1.0.safetensors"));
        ImageDefaults turbo = recipe.DefaultsFor(Classify(AnimaVariants.Catalog, "anima-turbo-v1.1.safetensors"));
        Assert.Equal((30, 4.0f), (baseDefaults.Steps, baseDefaults.CfgScale));
        Assert.Equal((10, 1.0f), (turbo.Steps, turbo.CfgScale));
    }

    [Theory]
    [InlineData("hidream_i1_dev_fp8.safetensors", "dev")]
    [InlineData("hidream_i1_dev_bf16.safetensors", "dev")]
    [InlineData("hidream_i1_full_fp16.safetensors", "full")]
    [InlineData("hidream_i1_fast_fp8.safetensors", "fast")]
    [InlineData("hidream.safetensors", "dev")]
    [InlineData("hidream_i1_dev_full_precision.safetensors", "dev")]
    [InlineData("hidream_i1_fast_full_precision.safetensors", "fast")]
    public void HiDream_VariantResolvesFromFileName(string fileName, string expected) =>
        Assert.Equal(expected, Classify(HiDreamVariants.Catalog, fileName).Id);

    [Fact]
    public void HiDream_OnlyFullSamplesWithCfg()
    {
        HiDreamRecipe recipe = new();
        ImageDefaults dev = recipe.DefaultsFor(Classify(HiDreamVariants.Catalog, "hidream_i1_dev_fp8.safetensors"));
        ImageDefaults full = recipe.DefaultsFor(Classify(HiDreamVariants.Catalog, "hidream_i1_full_fp16.safetensors"));
        ImageDefaults fast = recipe.DefaultsFor(Classify(HiDreamVariants.Catalog, "hidream_i1_fast_fp8.safetensors"));
        Assert.Equal((28, 1.0f), (dev.Steps, dev.CfgScale));
        Assert.Equal((50, 5.0f), (full.Steps, full.CfgScale));
        Assert.Equal((16, 1.0f), (fast.Steps, fast.CfgScale));
    }

    [Theory]
    [InlineData("boogu_image_base_bf16.safetensors", "base")]
    [InlineData("boogu_image_edit_fp8_scaled.safetensors", "edit")]
    [InlineData("boogu_image_turbo_hotfix_bf16.safetensors", "turbo")]
    [InlineData("boogu_image_turbo_fp8_scaled.safetensors", "turbo")]
    [InlineData("boogu_image_edit_turbo_bf16.safetensors", "edit-turbo")]
    [InlineData("boogu_image_edit_turbo_hotfix_1k_20260708_int8_convrot.safetensors", "edit-turbo")]
    public void Boogu_VariantResolvesFromFileName(string fileName, string expected) =>
        Assert.Equal(expected, Classify(BooguImageVariants.Catalog, fileName).Id);

    [Theory]
    [InlineData("boogu_image_turbo_bf16.safetensors", 4, 1.0f)]
    [InlineData("boogu_image_edit_turbo_bf16.safetensors", 4, 1.0f)]
    [InlineData("boogu_image_edit_fp8_scaled.safetensors", 25, 3.5f)]
    [InlineData("boogu_image_base_bf16.safetensors", 25, 3.5f)]
    public void Boogu_TurboBuildsRunFewGuidanceFreeSteps(string fileName, int steps, float cfg)
    {
        ImageDefaults defaults = new BooguImageRecipe().DefaultsFor(Classify(BooguImageVariants.Catalog, fileName));
        Assert.Equal((steps, cfg), (defaults.Steps, defaults.CfgScale));
    }

    /// <summary>Only the text-to-image Turbo names LCM: the reference-edit path refuses any sampler selection, so a
    /// default sampler on Edit Turbo would fail every edit request.</summary>
    [Theory]
    [InlineData("boogu_image_turbo_bf16.safetensors", "lcm")]
    [InlineData("boogu_image_edit_turbo_bf16.safetensors", null)]
    [InlineData("boogu_image_edit_fp8_scaled.safetensors", null)]
    public void Boogu_OnlyTextToImageTurboDefaultsToLcm(string fileName, string? sampler)
    {
        ImageDefaults defaults = new BooguImageRecipe().DefaultsFor(Classify(BooguImageVariants.Catalog, fileName));
        Assert.Equal(sampler, defaults.Sampler);
    }

    private static ResolvedModelVariant Classify(ModelVariantCatalog catalog, string fileName) =>
        ModelVariantResolver.Classify(catalog, CheckpointProbe.Empty with { FileNames = [fileName] }, []);
}
