using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Variants;
using HartsyInference.ModelAssets.Checkpoints;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>FLUX.2 Klein 4B, Klein 9B and Dev are told apart by the hidden size of the input projection. The catalog
/// reads the same tensor names the removed detection read, and the file name is the logged last resort for the
/// prefixed checkpoints those names never reach.</summary>
public sealed class Flux2VariantTests
{
    [Theory]
    [InlineData("img_in.weight", 3072, "klein-4b")]
    [InlineData("x_embedder.weight", 4096, "klein-9b")]
    [InlineData("patch_embed.proj.weight", 6144, "dev")]
    public void HiddenSize_SelectsTheBuildStructurally(string tensor, long hidden, string expectedId)
    {
        ResolvedModelVariant resolved = Classify(Probe(tensor, hidden, 128));
        Assert.Equal(expectedId, resolved.Id);
        Assert.Equal(ModelVariantSource.Structure, resolved.Source);
    }

    [Fact]
    public void FirstPresentProjectionTensorDecides_AndARankZeroOneFallsThrough()
    {
        CheckpointProbe probe = Probe("img_in.weight", 6144, 128) with
        {
            Keys = new HashSet<string>(["img_in.weight", "x_embedder.weight"], StringComparer.Ordinal),
            Shapes = new Dictionary<string, TensorShape>(StringComparer.Ordinal)
            {
                ["img_in.weight"] = new TensorShape(new long[] { 6144, 128 }),
                ["x_embedder.weight"] = new TensorShape(new long[] { 3072, 128 }),
            },
        };
        Assert.Equal("dev", Classify(probe).Id);

        CheckpointProbe scalar = Probe("img_in.weight", 3072, 128) with
        {
            Shapes = new Dictionary<string, TensorShape>(StringComparer.Ordinal) { ["img_in.weight"] = new TensorShape(Array.Empty<long>()) },
        };
        Assert.Equal(ModelVariantSource.Default, Classify(scalar).Source);
    }

    [Theory]
    [InlineData("flux2_dev_fp8", "dev")]
    [InlineData("flux-2-dev", "dev")]
    [InlineData("flux2-dev-sft", "dev")]
    [InlineData("flux-2-klein-4b", "klein-4b")]
    [InlineData("flux2_klein_4b_fp8", "klein-4b")]
    [InlineData("klein4b", "klein-4b")]
    [InlineData("flux2_klein_9b_fp8", "klein-9b")]
    public void FileName_IsTheLoggedLastResort_ForAPrefixedCheckpoint(string fileName, string expectedId)
    {
        CheckpointProbe prefixed = CheckpointProbe.Empty with
        {
            Keys = new HashSet<string>(["model.diffusion_model.img_in.weight"], StringComparer.Ordinal),
            FileNames = [fileName],
        };
        ResolvedModelVariant resolved = Classify(prefixed);
        Assert.Equal(expectedId, resolved.Id);
        Assert.Equal(ModelVariantSource.Filename, resolved.Source);
    }

    [Fact]
    public void NoEvidence_FallsBackToKlein4B()
    {
        ResolvedModelVariant resolved = Classify(CheckpointProbe.Empty with { FileNames = ["checkpoint"] });
        Assert.Equal(ModelVariantSource.Default, resolved.Source);
        Assert.Equal("klein-4b", resolved.Id);
    }

    [Theory]
    [InlineData("flux.2-klein-9b", "klein-9b")]
    [InlineData("flux.2-dev", "dev")]
    public void CallerHint_SelectsTheBuildWhenNoWeightsSayOtherwise(string hint, string expectedId)
    {
        ResolvedModelVariant resolved = ModelVariantResolver.Classify(Flux2Variants.Catalog, CheckpointProbe.Empty, [hint]);
        Assert.Equal(expectedId, resolved.Id);
        Assert.Equal(ModelVariantSource.CallerHint, resolved.Source);
    }

    [Fact]
    public void DefaultsFor_ReturnsTheResolvedVariantsDefaults()
    {
        Flux2Recipe recipe = new Flux2Recipe();
        ImageDefaults klein4B = recipe.DefaultsFor(Classify(Probe("img_in.weight", 3072, 128)));
        ImageDefaults klein9B = recipe.DefaultsFor(Classify(Probe("img_in.weight", 4096, 128)));
        ImageDefaults dev = recipe.DefaultsFor(Classify(Probe("img_in.weight", 6144, 128)));
        Assert.Equal((8, 1.0f, 1024, 1024), (klein4B.Steps, klein4B.CfgScale, klein4B.Width, klein4B.Height));
        Assert.Equal((8, 1.0f), (klein9B.Steps, klein9B.CfgScale));
        Assert.Equal((50, 3.5f, 1024, 1024), (dev.Steps, dev.CfgScale, dev.Width, dev.Height));
    }

    [Fact]
    public void ConfigFor_CarriesTheGuidanceEmbeddingOnlyForDev()
    {
        Assert.False(Flux2Variants.ConfigFor(Classify(Probe("img_in.weight", 3072, 128))).GuidanceEmbed);
        Assert.False(Flux2Variants.ConfigFor(Classify(Probe("img_in.weight", 4096, 128))).GuidanceEmbed);
        Assert.True(Flux2Variants.ConfigFor(Classify(Probe("img_in.weight", 6144, 128))).GuidanceEmbed);
    }

    private static CheckpointProbe Probe(string tensor, long hidden, long inChannels) => CheckpointProbe.Empty with
    {
        Keys = new HashSet<string>([tensor], StringComparer.Ordinal),
        Shapes = new Dictionary<string, TensorShape>(StringComparer.Ordinal) { [tensor] = new TensorShape(new[] { hidden, inChannels }) },
    };

    private static ResolvedModelVariant Classify(CheckpointProbe probe) =>
        ModelVariantResolver.Classify(Flux2Variants.Catalog, probe, []);
}
