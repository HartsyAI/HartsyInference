using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Variants;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Regression coverage for the Tier 0.2 <c>wan-22-5b</c>/<c>wan-21-1_3b</c> <see cref="VideoFeatures.EndFrame"/>
/// over-claim fix, <b>updated 2026-08-11 for Tier 3.3's real wiring</b>: <c>wan-22-5b</c> (TI2V-5B) now DOES claim
/// <see cref="VideoFeatures.EndFrame"/> — <see cref="WanVideoRecipePipeline"/>'s non-concat path VAE-encodes
/// <c>VideoRequest.VideoEndFrame</c> into a <c>lastFrameLatent</c> exactly like <c>InitImage</c>'s
/// <c>firstFrameLatent</c>, real-weight verified against the local TI2V-5B checkpoint. <c>wan-21-1_3b</c> shares
/// the identical code path but stays narrowed — no local 1.3B checkpoint exists to verify against, and this
/// backlog's rule is real-checkpoint verification, not "should work by symmetry." Weight-free — <c>Supports</c> is
/// resolved purely from the family id, no checkpoint touched, which is exactly the kind of silent flag/behavior
/// drift a unit test should catch.</summary>
public sealed class WanVideoRecipeSupportsTests
{
    [Fact]
    public void Supports_UnverifiedNonConcatFamily_DoesNotClaimEndFrame()
    {
        VideoFeatures supports = new WanVideoRecipe(WanVideoRecipe.Wan21_1_3BCompatClassId).Supports;

        Assert.Equal(VideoFeatures.None, supports & VideoFeatures.EndFrame);
        Assert.Equal(VideoFeatures.InitImage, supports & VideoFeatures.InitImage);
    }

    [Fact]
    public void Supports_VerifiedTi2V5B_ClaimsEndFrame()
    {
        VideoFeatures supports = new WanVideoRecipe(WanVideoRecipe.Wan22_5BCompatClassId).Supports;

        Assert.Equal(VideoFeatures.EndFrame, supports & VideoFeatures.EndFrame);
        Assert.Equal(VideoFeatures.InitImage, supports & VideoFeatures.InitImage);
    }

    /// <summary>The 14B class covers T2V and concat-I2V files and neither has been checked with an end frame; the generic
    /// slug carries no size. Both keep the init image and stop claiming the end frame at family level.</summary>
    [Theory]
    [InlineData(WanVideoRecipe.Wan21_14BCompatClassId)]
    [InlineData("wan")]
    public void Supports_UnverifiedOrSizelessFamilies_DoNotClaimEndFrame(string familyId)
    {
        VideoFeatures supports = new WanVideoRecipe(familyId).Supports;

        Assert.Equal(VideoFeatures.None, supports & VideoFeatures.EndFrame);
        Assert.Equal(VideoFeatures.InitImage, supports & VideoFeatures.InitImage);
    }

    /// <summary>The verified end-frame run goes through the generic slug, so a TI2V-5B file under it must get the end
    /// frame back from its header; a 16-channel (1.3B / 14B) file must not.</summary>
    [Theory]
    [InlineData(48, true)]
    [InlineData(16, false)]
    public void SupportsFor_GenericSlug_ClaimsEndFrameOnlyForTi2V5BHeader(int latentChannels, bool expectEndFrame)
    {
        string path = WriteBackbone(latentChannels);
        try
        {
            Assert.Equal(expectEndFrame, Resolve(path).Is(WanVideoVariants.Ti2V5B));
            VideoFeatures supports = new WanVideoRecipe("wan").SupportsFor(Resolve(path));
            Assert.Equal(expectEndFrame ? VideoFeatures.EndFrame : VideoFeatures.None, supports & VideoFeatures.EndFrame);
            Assert.Equal(VideoFeatures.InitImage, supports & VideoFeatures.InitImage);
        }
        finally
        {
            File.Delete(path);
        }
    }

    /// <summary>A 5B-shaped file under the 14B class is a misclassified checkpoint; the class answer wins, as it does
    /// for the config the recipe builds.</summary>
    [Fact]
    public void SupportsFor_14BClass_IgnoresA5BHeader()
    {
        string path = WriteBackbone(48);
        try
        {
            VideoFeatures supports = new WanVideoRecipe(WanVideoRecipe.Wan21_14BCompatClassId).SupportsFor(Resolve(path));
            Assert.Equal(VideoFeatures.None, supports & VideoFeatures.EndFrame);
        }
        finally
        {
            File.Delete(path);
        }
    }

    /// <summary>A damaged file answers "not a 5B" instead of throwing out of a capability query.</summary>
    [Fact]
    public void Ti2V5B_GarbageFile_IsNotClaimed()
    {
        string path = Path.Combine(Path.GetTempPath(), $"wan-garbage-{Guid.NewGuid():N}.safetensors");
        File.WriteAllBytes(path, [0x10, 0, 0, 0, 0, 0, 0, 0, (byte)'{', (byte)'"']);
        try
        {
            Assert.False(Resolve(path).Is(WanVideoVariants.Ti2V5B));
            Assert.Equal(VideoFeatures.None, new WanVideoRecipe("wan").SupportsFor(Resolve(path)) & VideoFeatures.EndFrame);
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static ResolvedModelVariant Resolve(string path) =>
        ModelVariantResolver.Resolve(WanVideoVariants.Catalog, new ModelVariantEvidence(path, []));

    /// <summary>A header-only stand-in for a plain Wan backbone: the patch embedding at the given latent width plus one
    /// block key, which is all the variant sniff and the size check read.</summary>
    private static string WriteBackbone(int latentChannels)
    {
        string path = Path.Combine(Path.GetTempPath(), $"wan-backbone-{latentChannels}-{Guid.NewGuid():N}.safetensors");
        using Tensor patch = new Tensor(new TensorShape([4L, latentChannels, 1L, 2L, 2L]), DType.F32);
        using Tensor block = new Tensor(new TensorShape(4, 4), DType.F32);
        SafeTensorsWriter.Save(path, new Dictionary<string, Tensor>
        {
            ["patch_embedding.weight"] = patch,
            ["blocks.0.self_attn.q.weight"] = block,
        });
        return path;
    }

    [Fact]
    public void SupportsFor_NullCheckpoint_FallsBackToVerifiedSupports_Ti2V5B()
    {
        WanVideoRecipe recipe = new WanVideoRecipe(WanVideoRecipe.Wan22_5BCompatClassId);

        Assert.Equal(recipe.Supports, recipe.SupportsFor(null));
        Assert.Equal(VideoFeatures.EndFrame, recipe.SupportsFor(null) & VideoFeatures.EndFrame);
    }
}
