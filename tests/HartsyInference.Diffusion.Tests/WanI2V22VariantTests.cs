using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Variants;
using HartsyInference.ModelAssets.Checkpoints;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Wan 2.2 I2V-14B is told from the other 14B checkpoints by structure, as SwarmUI's <c>wan-2_2-image2video-14b</c>
/// class is, and takes core's CFG 3.5 instead of Wan 2.1's 6.</summary>
public sealed class WanI2V22VariantTests
{
    [Theory]
    [InlineData(5120, 36, new string[0], "i2v-22-14b")]
    [InlineData(5120, 36, new[] { "blocks.0.cross_attn.k_img.weight", "img_emb.proj.0.bias" }, "base")]
    [InlineData(5120, 36, new[] { "img_emb.emb_pos" }, "base")]
    [InlineData(5120, 16, new string[0], "base")]
    [InlineData(1536, 36, new string[0], "base")]
    [InlineData(5120, 36, new[] { "vace_blocks.0.after_proj.bias" }, "vace")]
    public void Classify_NeedsThe14BHiddenSizeA36ChannelPatchEmbeddingAndNoImageBranch(long hidden, long inChannels, string[] extraKeys, string expected)
    {
        Dictionary<string, TensorShape> shapes = new Dictionary<string, TensorShape>(StringComparer.Ordinal)
        {
            ["patch_embedding.weight"] = new TensorShape([hidden, inChannels, 1L, 2L, 2L]),
        };
        HashSet<string> keys = new HashSet<string>(extraKeys.Append("patch_embedding.weight"), StringComparer.Ordinal);
        CheckpointProbe probe = CheckpointProbe.Empty with { Keys = keys, Shapes = shapes };

        Assert.Equal(expected, ModelVariantResolver.Classify(WanVideoVariants.Catalog, probe, []).Id);
    }

    /// <summary>The resolver runs structure rules before metadata ones, so Animate-2's metadata must keep a repack without
    /// the image branch away from the I2V-14B rule.</summary>
    [Fact]
    public void Classify_Animate2MetadataOnTheSameShape_IsNotTakenForI2V22()
    {
        CheckpointProbe probe = CheckpointProbe.Empty with
        {
            Keys = new HashSet<string>(["patch_embedding.weight"], StringComparer.Ordinal),
            Shapes = new Dictionary<string, TensorShape>(StringComparer.Ordinal) { ["patch_embedding.weight"] = new TensorShape([5120L, 36L, 1L, 2L, 2L]) },
            Metadata = new Dictionary<string, string>(StringComparer.Ordinal) { ["config"] = "{\"transformer\":{\"model_type\":\"animate2\"}}" },
        };

        Assert.Equal("animate2", ModelVariantResolver.Classify(WanVideoVariants.Catalog, probe, []).Id);
    }

    [Fact]
    public void DefaultsFor_I2V22_TakesCoreCfg_AndStaysAPlainBackbone()
    {
        WanVideoRecipe recipe = new WanVideoRecipe(WanVideoRecipe.Wan21_14BCompatClassId);
        ResolvedModelVariant i2v = new ResolvedModelVariant(WanVideoVariants.Catalog.FamilyId, WanVideoVariants.I2V22_14B, ModelVariantSource.Structure, "test");
        VideoDefaults defaults = recipe.DefaultsFor(i2v);

        Assert.Equal((3.5f, 81, 24, 3), (defaults.CfgScale, defaults.Frames, defaults.Fps, defaults.Revision));
        Assert.True(WanVideoRecipe.IsPlainBackbone(i2v));
    }
}
