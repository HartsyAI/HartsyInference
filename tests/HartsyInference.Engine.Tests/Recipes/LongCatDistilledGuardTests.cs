using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Registry;
using HartsyInference.ModelAssets.Lora;
using Xunit;

namespace HartsyInference.Engine.Tests;

/// <summary>The distilled base build's guards: its steps and guidance are locked, its cfg_step LoRA must exist and must apply in full, and it shares
/// the base row's DiT asset list.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatDistilledGuardTests
{
    [Fact]
    public void TheDistilledDefaultsLockStepsAndGuidance()
    {
        VideoDefaults defaults = new LongCatVideoDistilledRecipe().Defaults;
        Assert.Equal(VideoLockedFields.Steps | VideoLockedFields.CfgScale, defaults.LockedFields);
    }

    [Fact]
    public void ALoraThatDoesNotApplyInFullIsRefused()
    {
        LongCatRecipeSupport.RequireFullyApplied(new HartsyInference.ModelAssets.Lora.LoraStack.ApplyReport(1152, 1152, 0, 0, 0, 0), "cfg_step_lora");
        Assert.Throws<NotSupportedException>(() => LongCatRecipeSupport.RequireFullyApplied(new HartsyInference.ModelAssets.Lora.LoraStack.ApplyReport(0, 0, 0, 0, 0, 0), "x"));
        Assert.Throws<NotSupportedException>(() => LongCatRecipeSupport.RequireFullyApplied(new HartsyInference.ModelAssets.Lora.LoraStack.ApplyReport(1150, 1150, 0, 2, 0, 0), "x"));
        Assert.Throws<NotSupportedException>(() => LongCatRecipeSupport.RequireFullyApplied(new HartsyInference.ModelAssets.Lora.LoraStack.ApplyReport(1150, 1150, 0, 0, 2, 0), "x"));
    }

    [Fact]
    public void TheMissingCfgStepLoraIsNamedBeforeAnyWeightIsRead()
    {
        NotSupportedException refused = Assert.Throws<NotSupportedException>(() => LongCatRecipeSupport.RequireCfgStepLora("no-such-folder/dit"));
        Assert.Contains("cfg_step_lora.safetensors", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheBaseAndDistilledRowsShareTheDitAssetList()
    {
        string[] Dit(string id) => ModelCatalog.Find(id)!.Assets
            .Where(asset => asset.RepoPath.StartsWith("dit/", StringComparison.Ordinal))
            .Select(asset => asset.RepoPath).ToArray();
        Assert.NotEmpty(Dit("longcat-video"));
        Assert.Equal(Dit("longcat-video"), Dit("longcat-video-distilled"));
        Assert.Contains(ModelCatalog.Find("longcat-video-distilled")!.Assets, asset => asset.RepoPath == "lora/cfg_step_lora.safetensors");
    }
}
