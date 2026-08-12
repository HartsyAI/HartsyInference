using HartsyInference.Engine.Features;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Pure-logic tests for <see cref="ReferenceCropResolver"/>'s <c>&lt;refcrop:&gt;</c> grammar — the
/// malformed/edge-case paths that never touch CLIPSeg (a malformed tag is dropped before
/// <c>VisionModelPaths.FindClipSegDirectory</c> is ever called, since <c>Apply</c> only reaches it when at least
/// one tag actually parsed). The successful-crop path (real index, real CLIPSeg match) is real-weight tested
/// separately — this file only locks in the parser's tolerance and grammar, mirroring
/// <see cref="SegmentRefinementStripTests"/>'s own scope split for the sibling <c>&lt;segment:&gt;</c> feature.</summary>
public sealed class ReferenceCropResolverTests
{
    private static VideoRequest MakeRequest(string prompt) => new VideoRequest { Prompt = prompt };

    [Fact]
    public void NoTags_HasCropTagsFalse_ApplyReturnsSameInstance()
    {
        VideoRequest request = MakeRequest("a lighthouse at sunset");
        Assert.False(ReferenceCropResolver.HasCropTags(request.Prompt));
        VideoRequest result = ReferenceCropResolver.Apply(request, backend: null!, segmenter: null!, cancel: default);
        Assert.Same(request, result);
    }

    [Fact]
    public void MissingComma_DroppedWithTagStripped()
    {
        VideoRequest request = MakeRequest("base text <refcrop:1> tail");
        Assert.True(ReferenceCropResolver.HasCropTags(request.Prompt));
        VideoRequest result = ReferenceCropResolver.Apply(request, backend: null!, segmenter: null!, cancel: default);
        Assert.Equal("base text  tail", result.Prompt);
    }

    [Fact]
    public void ZeroIndex_Dropped_OneBasedNotZeroBased()
    {
        VideoRequest request = MakeRequest("base <refcrop:0,the cat> tail");
        VideoRequest result = ReferenceCropResolver.Apply(request, backend: null!, segmenter: null!, cancel: default);
        Assert.Equal("base  tail", result.Prompt);
    }

    [Fact]
    public void NonNumericIndex_Dropped()
    {
        VideoRequest request = MakeRequest("base <refcrop:x,the cat> tail");
        VideoRequest result = ReferenceCropResolver.Apply(request, backend: null!, segmenter: null!, cancel: default);
        Assert.Equal("base  tail", result.Prompt);
    }

    [Fact]
    public void EmptyQuery_Dropped()
    {
        VideoRequest request = MakeRequest("base <refcrop:1,> tail");
        VideoRequest result = ReferenceCropResolver.Apply(request, backend: null!, segmenter: null!, cancel: default);
        Assert.Equal("base  tail", result.Prompt);
    }

    [Fact]
    public void UnterminatedTag_LeftAsLiteralText()
    {
        // No closing '>' — same tolerance PromptRegionParser has for an unterminated "<...".
        const string prompt = "base <refcrop:1,the cat";
        VideoRequest request = MakeRequest(prompt);
        VideoRequest result = ReferenceCropResolver.Apply(request, backend: null!, segmenter: null!, cancel: default);
        Assert.Equal(prompt, result.Prompt);
    }

    [Fact]
    public void ReferenceImagesUntouched_WhenAllTagsMalformed()
    {
        ImageData img = new ImageData { Rgb = new byte[3], Width = 1, Height = 1 };
        VideoRequest request = MakeRequest("base <refcrop:bad> tail") with { ReferenceImages = [img] };
        VideoRequest result = ReferenceCropResolver.Apply(request, backend: null!, segmenter: null!, cancel: default);
        Assert.Same(request.ReferenceImages, result.ReferenceImages);
    }
}
