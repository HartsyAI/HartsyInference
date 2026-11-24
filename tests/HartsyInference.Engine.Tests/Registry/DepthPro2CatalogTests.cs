using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Variants;
using HartsyInference.Engine.Vision;
using HartsyInference.ModelAssets.Checkpoints;

namespace HartsyInference.Engine.Tests.Registry;

/// <summary>The Depth Pro 2 rows download only their own size plus the focal estimator, and the variant family tells the sizes apart by the
/// checkpoint's backbone width.</summary>
public sealed class DepthPro2CatalogTests
{
    [Theory]
    [InlineData("depthpro2-s", "apple/DepthPro2-S", "Vision/DepthPro2-S")]
    [InlineData("depthpro2-b", "apple/DepthPro2-B", "Vision/DepthPro2-B")]
    [InlineData("depthpro2-l", "apple/DepthPro2-L", "Vision/DepthPro2-L")]
    [InlineData("depthpro2-h", "apple/DepthPro2-H", "Vision/DepthPro2-H")]
    public void EachRowListsItsOwnSizeAndTheFocalEstimatorOnly(string id, string repo, string folder)
    {
        CatalogEntry entry = ModelCatalog.Find(id)!;

        Assert.Equal(Modality.Vision, entry.Modality);
        Assert.Equal(2, entry.Assets.Count);
        ModelAsset depth = entry.Assets.Single(a => a.Role == "transformer");
        Assert.Equal((repo, "model.safetensors", folder), (depth.Repo, depth.RepoPath, depth.TargetSubdir));
        Assert.Equal(64, depth.Sha256!.Length);
        ModelAsset focal = entry.Assets.Single(a => a.Role == DepthPro2Catalog.FocalRole);
        Assert.Equal(("apple/DepthPro2-focal", "Vision/DepthPro2-focal"), (focal.Repo, focal.TargetSubdir));
        Assert.Equal("8f567d1c97ff916137288668f6bd306ce2094da2e2b852b0e4539dbb139c8c86", focal.Sha256);
        Assert.NotNull(DepthPro2Variants.Catalog.Find(id));
    }

    [Theory]
    [InlineData(384, "depthpro2-s")]
    [InlineData(768, "depthpro2-b")]
    [InlineData(1024, "depthpro2-l")]
    [InlineData(1280, "depthpro2-h")]
    public void TheBackboneWidthPicksTheSizeOverAContradictingHint(int width, string expected)
    {
        ResolvedModelVariant resolved = ModelVariantResolver.Classify(DepthPro2Variants.Catalog, Probe(width, withHead: true), ["depthpro2-b"]);

        Assert.Equal(ModelVariantSource.Structure, resolved.Source);
        Assert.Equal(expected, resolved.Id);
        Assert.Equal(width, DepthPro2Variants.PresetFor(resolved).Backbone.HiddenSize);
    }

    [Fact]
    public void TheFocalEstimatorOrAnUnknownWidthIsNotADepthCheckpoint()
    {
        Assert.False(DepthPro2Variants.IsDepthPro2(Probe(384, withHead: false)));
        Assert.False(DepthPro2Variants.IsDepthPro2(Probe(512, withHead: true)));
        Assert.True(DepthPro2Variants.IsDepthPro2(Probe(384, withHead: true)));
    }

    private static CheckpointProbe Probe(int width, bool withHead)
    {
        Dictionary<string, TensorShape> shapes = new(StringComparer.Ordinal) { ["encoder.blocks.0.attn.qkv.weight"] = new TensorShape(3 * width, width) };
        if (withHead) shapes["head.4.weight"] = new TensorShape(1, 32, 1, 1);
        return new CheckpointProbe(new HashSet<string>(shapes.Keys, StringComparer.Ordinal), shapes, new Dictionary<string, string>(), []);
    }
}
