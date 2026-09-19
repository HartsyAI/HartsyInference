using System.Linq;
using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Planning;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Diffusion.Models.TextEncoders;

namespace HartsyInference.Diffusion.Tests;

/// <summary>What happens to H3's rank-5 vision patch embedding when it arrives inside a GGUF. ggml caps a tensor at
/// <c>GGML_MAX_DIMS = 4</c>, so no GGUF can hold a rank-5 weight and every published repack stores
/// <c>[hidden, inChannels, t, h, w]</c> with its leading pair folded — <c>qwen3vl_32b_minimax_h3-Q4_K_M.gguf</c> ships
/// <c>visual.patch_embed.proj.weight</c> as <c>[3456, 2, 16, 16]</c>, which is 1152·3 in front. Row-major the two
/// layouts are the same bytes in the same order, so the fold is a relabeling; the danger is that every dimension after
/// the first shifts left, which reads the temporal patch size where the input channel count should be.</summary>
public sealed class MiniMaxH3GgufConvFoldTests
{
    private const long Hidden = 1152, InChannels = 3, Temporal = 2, Patch = 16;

    private static SafeTensorDescriptor Descriptor(string name, DType dtype, params long[] dimensions) => new()
    {
        Name = name,
        DType = dtype,
        Shape = new TensorShape(dimensions),
        DataOffset = 0,
        ByteLength = dtype.ComputeByteCount(dimensions.Aggregate(1L, (product, value) => product * value)),
    };

    /// <summary>Only the patch embedding varies between these cases, so everything else the text-encoder validator
    /// checks is supplied at its expected shape and left alone.</summary>
    private static Dictionary<string, SafeTensorDescriptor> TextEncoder(params long[] patchEmbedShape)
    {
        Dictionary<string, SafeTensorDescriptor> descriptors = new(StringComparer.Ordinal)
        {
            ["visual.patch_embed.proj.weight"] =
                Descriptor("visual.patch_embed.proj.weight", DType.BF16, patchEmbedShape),
            ["visual.merger.norm.weight"] = Descriptor("visual.merger.norm.weight", DType.F32, 1152),
            ["visual.merger.linear_fc1.weight"] = Descriptor("visual.merger.linear_fc1.weight", DType.BF16, 4608, 4608),
            ["visual.merger.linear_fc2.weight"] = Descriptor("visual.merger.linear_fc2.weight", DType.BF16, 5120, 4608),
        };
        return descriptors;
    }

    private static string[] PatchEmbedIssues(params long[] patchEmbedShape)
    {
        List<VideoPlanIssue> issues = [];
        VideoProfileResolver.ValidateComponentStructure("textEncoder", TextEncoder(patchEmbedShape), issues);
        return issues.Where(issue => issue.Message.Contains("patch_embed", StringComparison.Ordinal))
            .Select(issue => issue.Message).ToArray();
    }

    /// <summary>The safetensors form, which has always worked and must keep working.</summary>
    [Fact]
    public void PlannerAcceptsTheRankFiveWeight() =>
        Assert.Empty(PatchEmbedIssues(Hidden, InChannels, Temporal, Patch, Patch));

    /// <summary>The regression this exists for: the published GGUF text encoder was refused at preflight as a shape
    /// error, which read as a corrupt file rather than a container limit.</summary>
    [Fact]
    public void PlannerAcceptsTheGgufFoldOfIt() =>
        Assert.Empty(PatchEmbedIssues(Hidden * InChannels, Temporal, Patch, Patch));

    /// <summary>The fold is accepted as that exact shape and not as "rank 4 with the right number of elements". A
    /// weight that redistributes the same elements is a different tensor and stays refused.</summary>
    [Theory]
    [InlineData(new long[] { 1152, 6, 16, 16 })]
    [InlineData(new long[] { 1728, 4, 16, 16 })]
    [InlineData(new long[] { 3456, 2, 8, 32 })]
    public void PlannerStillRefusesAnyOtherRankFourWeight(long[] shape)
    {
        string[] issues = PatchEmbedIssues(shape);
        Assert.Single(issues);
        // Both accepted forms are named, so the message says what to look for rather than only what was wrong.
        Assert.Contains("1152,3,2,16,16", issues[0], StringComparison.Ordinal);
        Assert.Contains("3456,2,16,16", issues[0], StringComparison.Ordinal);
    }

    /// <summary>The consumer half. <c>InChannels</c> used to be read as <c>Shape[1]</c>, which is the input channel
    /// count on the rank-5 weight and the TEMPORAL PATCH SIZE on the folded one — 2 where 3 belongs. Nothing would
    /// have thrown: the tower would have built a [1152, 1024] projection out of a [1152, 1536] weight and copied the
    /// front of it, serving a silently truncated patch embedding.</summary>
    [Theory]
    [InlineData(new long[] { Hidden, InChannels, Temporal, Patch, Patch })]
    [InlineData(new long[] { Hidden * InChannels, Temporal, Patch, Patch })]
    public void InputChannelsAreReadFromTheWeightsSizeNotItsRank(long[] shape)
    {
        using Tensor patchProj = new Tensor(new TensorShape(shape), DType.F32);
        Assert.Equal(InChannels, MiniMaxH3TextEncoder.PatchEmbedInChannels(patchProj, (int)Hidden));
    }

    /// <summary>A weight that does not divide by the bias-derived hidden size is a mismatched pair of files, and says
    /// so rather than rounding down to a channel count that happens to fit.</summary>
    [Fact]
    public void AWeightThatDoesNotDivideByTheHiddenSizeIsRefused()
    {
        using Tensor patchProj = new Tensor(new TensorShape(Hidden * InChannels + 1, Temporal, Patch, Patch), DType.F32);
        InvalidOperationException ex = Assert.Throws<InvalidOperationException>(
            () => MiniMaxH3TextEncoder.PatchEmbedInChannels(patchProj, (int)Hidden));
        Assert.Contains("does not divide", ex.Message, StringComparison.Ordinal);
    }
}
