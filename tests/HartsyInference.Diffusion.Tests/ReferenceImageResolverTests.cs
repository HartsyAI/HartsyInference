using HartsyInference.Engine.Features;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The shared reference-image resolver: picture order, the overflow rules, and each copy's size under the
/// MOSS-Image (scale down only, Ostris budgets) and Qwen-Image-Edit (scale to the budget) policies.</summary>
public sealed class ReferenceImageResolverTests
{
    private static readonly ReferenceImagePolicy Moss = MossImageRecipePipeline.ReferencePolicy(null);

    [Fact]
    public void Collect_PutsTheInitReferenceFirst_AndAppliesTheOverflowRule()
    {
        ImageData init = Image(64, 64), a = Image(32, 32), b = Image(48, 48);
        Assert.Equal([init, a, b], ReferenceImageResolver.Collect(init, [a, b], Moss, "test"));
        Assert.Throws<ArgumentException>(() => ReferenceImageResolver.Collect(init, [a, b, Image(16, 16)], Moss, "test"));
        ReferenceImagePolicy drop = Moss with { Overflow = ReferenceOverflow.Drop };
        Assert.Equal([init, a, b], ReferenceImageResolver.Collect(init, [a, b, Image(16, 16)], drop, "test"));
        Assert.Empty(ReferenceImageResolver.Collect(null, null, Moss, "test"));
    }

    [Theory]
    [InlineData(1920, 1080, 1360, 768, 512, 288)]
    [InlineData(1024, 1024, 1024, 1024, 384, 384)]
    [InlineData(100, 100, 96, 96, 100, 100)]
    [InlineData(1000, 700, 992, 704, 459, 321)]
    public void Moss_ScalesEachCopyDownToItsBudget_AndSnapsTheVaeCopy(int width, int height, int latentW, int latentH, int visionW, int visionH)
    {
        Assert.Equal((latentW, latentH), ReferenceImageResolver.LatentSize(width, height, Moss));
        Assert.Equal((visionW, visionH), ReferenceImageResolver.VisionSize(width, height, Moss));
    }

    [Fact]
    public void Resolve_BuildsBothCopies_ThroughTheVisionSizer()
    {
        using ResolvedReferences? references = ReferenceImageResolver.Resolve([Image(1920, 1080)], Moss,
            static (w, h) => (w / 32 * 32, h / 32 * 32));
        Assert.NotNull(references);
        Assert.Equal(new long[] { 1, 3, 768, 1360 }, Dims(references!.Latent[0]));
        Assert.Equal(new long[] { 1, 3, 288, 512 }, Dims(references.Vision[0]));
        using ResolvedReferences? keepSize = ReferenceImageResolver.Resolve([Image(1000, 700)], Moss with { LatentTargetArea = 0 });
        Assert.Equal(new long[] { 1, 3, 704, 992 }, Dims(keepSize!.Latent[0]));
        Assert.Null(ReferenceImageResolver.Resolve([], Moss));
    }

    [Fact]
    public void QwenPolicy_ScalesUpToItsBudgets()
    {
        ReferenceImagePolicy qwen = QwenImageEditConditioning.PolicyFor(QwenImageEditTemplate.EditPlus);
        (int w, int h) = ReferenceImageResolver.LatentSize(320, 240, qwen);
        Assert.InRange((long)w * h, 1024L * 1024 * 9 / 10, 1024L * 1024 * 11 / 10);
        (int vw, int vh) = ReferenceImageResolver.VisionSize(64, 64, qwen);
        Assert.Equal((384, 384), (vw, vh));
    }

    private static ImageData Image(int width, int height) => new() { Rgb = new byte[width * height * 3], Width = width, Height = height };

    private static long[] Dims(HartsyInference.Core.Tensors.Tensor t) => [t.Shape[0], t.Shape[1], t.Shape[2], t.Shape[3]];
}
