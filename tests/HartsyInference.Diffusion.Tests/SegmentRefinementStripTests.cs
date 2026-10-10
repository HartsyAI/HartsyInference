using HartsyInference.Engine.Features;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Pure-logic tests for <see cref="SegmentRefinement.StripSegmentText"/> — the base-prompt tag-leak fix
/// found while live-verifying Tier 3.2 (a same-seed A/B against the real running SwarmUI service measured ~40-47
/// mean abs diff on a patch of pure untouched background, proving the literal <c>&lt;segment:X&gt;</c> tag text
/// was reaching the base pass's text encoder). No GPU needed — this only exercises the split/accumulate loop
/// against <see cref="PromptRegionParser"/>'s own tag grammar.</summary>
public sealed class SegmentRefinementStripTests
{
    [Fact]
    public void SegmentAtEnd_StrippedWithTrailingSpaceTrimmed()
    {
        const string prompt = "a red apple next to a green pear <segment:the red apple,0.95,0.5>a bright blue apple";
        Assert.Equal("a red apple next to a green pear", SegmentRefinement.StripSegmentText(prompt));
    }

    [Fact]
    public void SegmentFollowedByRegionTag_RegionSurvivesVerbatim()
    {
        const string prompt = "base text <segment:the apple,0.5>blue apple<region:0,0,0.5,1,1>left half text<region:end> tail";
        // The segment's own sub-prompt ("blue apple") is dropped; the region tag + its content + the trailing
        // "<region:end> tail" are preserved byte-for-byte so RegionalPromptResolver re-parses them unchanged.
        Assert.Equal("base text <region:0,0,0.5,1,1>left half text<region:end> tail", SegmentRefinement.StripSegmentText(prompt));
    }

    [Fact]
    public void CidSuffix_MatchedOnPrefixNotLiteralTagText()
    {
        const string prompt = "base <segment:the red apple,0.95,0.5//cid=11>a bright blue apple";
        Assert.Equal("base", SegmentRefinement.StripSegmentText(prompt));
    }

    [Fact]
    public void SegmentEndQuirk_MatchesLiteralEndTextAndIsStripped()
    {
        // <segment:end> is NOT a closer (unlike <region:end>) — it parses as an ordinary segment whose matcher
        // text is the literal string "end". The stripper must drop it the same as any other segment tag.
        const string prompt = "base text <segment:the apple,0.5>blue apple<segment:end>";
        Assert.Equal("base text", SegmentRefinement.StripSegmentText(prompt));
    }

    [Fact]
    public void ClearTagFollowedByRegionTag_RegionSurvivesVerbatim()
    {
        const string prompt = "base <clear:the background,0.5>ignored<region:0,0,0.5,1,1>kept text";
        Assert.Equal("base <region:0,0,0.5,1,1>kept text", SegmentRefinement.StripSegmentText(prompt));
    }

    [Fact]
    public void HasSegmentParts_StillTrueOnOriginalPromptAfterStripping()
    {
        // The strip helper must never be applied to the prompt SegmentRefinement.Apply itself re-parses — this
        // just locks in that HasSegmentParts keeps seeing the ORIGINAL prompt's tags (the caller passes `resolved`,
        // not the stripped copy, into Apply).
        const string prompt = "base <segment:the apple,0.5>blue apple";
        Assert.True(SegmentRefinement.HasSegmentParts(prompt));
        Assert.False(SegmentRefinement.HasSegmentParts(SegmentRefinement.StripSegmentText(prompt)));
    }
}
