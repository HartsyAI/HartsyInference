using Xunit;
using HartsyInference.Engine.Features;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Who owns the emphasis when a prompt carries both region tags and a weight. A recipe that declares a
/// weighting mode encodes its base prompt from the whole string, region tags included, so a region's own
/// <c>(word:N)</c> would otherwise surface as a base-prompt weight and scale rows belonging to the tag text.</summary>
public sealed class RegionalPromptWeightSplitTests
{
    /// <summary>The case that must keep working: the emphasis is inside a region, so the base carries none and
    /// nothing is refused. Getting this wrong turns a legitimate weighted region into an error.</summary>
    [Fact]
    public void AWeightInsideARegionIsNotABaseWeight()
    {
        Assert.False(RegionalPromptWeightSplit.BaseTextCarriesWeight(
            "a snowy field <region:0,0,0.5,1,1>a red (fox:0.5)"));
    }

    /// <summary>The case that must be refused: both halves weighted would land two sets of weights on the same
    /// conditioning rows, because the base encode covers the tag text too.</summary>
    [Fact]
    public void AWeightOutsideEveryRegionIsABaseWeight()
    {
        Assert.True(RegionalPromptWeightSplit.BaseTextCarriesWeight(
            "a red (fox:1.5) <region:0,0,0.5,1,1>a snowdrift"));
    }

    /// <summary>SwarmUI's tag spelling counts the same as the Comfy parens a CLI user types — the tag is what
    /// actually arrives from SwarmUI, so checking only the parens would miss every real request.</summary>
    [Fact]
    public void TheSwarmTagSpellingCountsToo()
    {
        Assert.True(RegionalPromptWeightSplit.BaseTextCarriesWeight(
            "a red <weight[1.5]:fox> <region:0,0,0.5,1,1>a snowdrift"));
    }

    /// <summary>A weight of exactly 1 does nothing, so it is not worth refusing a request over.</summary>
    [Fact]
    public void AUnitWeightIsNotAWeight()
    {
        Assert.False(RegionalPromptWeightSplit.BaseTextCarriesWeight(
            "a red (fox:1.0) <region:0,0,0.5,1,1>a snowdrift"));
    }

    /// <summary>With regions present the base text loses the grammar in BOTH spellings. Flattening alone handles
    /// only the tag; the literal parens would still split the builder's spans, which is the bug this exists for.
    /// </summary>
    [Theory]
    [InlineData("a snowy field <region:0,0,0.5,1,1>a red (fox:0.5)", "a snowy field <region:0,0,0.5,1,1>a red fox")]
    [InlineData("a snowy field <region:0,0,0.5,1,1>a red <weight[0.5]:fox>", "a snowy field <region:0,0,0.5,1,1>a red fox")]
    public void RegionsStripTheBaseWeightGrammarInBothSpellings(string prompt, string expected)
    {
        Assert.Equal(expected, RegionalPromptWeightSplit.BaseText(prompt, hasRegionParts: true));
    }

    /// <summary>Without regions the grammar is KEPT, because that is what the weighted builder consumes. This is
    /// the pairing that matters: the same helper must not quietly disarm weighting on an ordinary prompt.</summary>
    [Fact]
    public void WithoutRegionsTheWeightGrammarSurvives()
    {
        Assert.Equal("a red (fox:0.5) in snow",
            RegionalPromptWeightSplit.BaseText("a red <weight[0.5]:fox> in snow", hasRegionParts: false));
    }

    /// <summary>An unweighted prompt is unchanged either way, so a recipe routing through this never pays for a
    /// rewrite it did not need.</summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void AnUnweightedPromptIsUntouched(bool hasRegionParts)
    {
        Assert.Equal("a red fox in snow", RegionalPromptWeightSplit.BaseText("a red fox in snow", hasRegionParts));
    }
}
