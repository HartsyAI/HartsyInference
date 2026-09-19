using Xunit;
using HartsyInference.Diffusion.Prompting;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The joint-attention half of Krea 2's weighting: which positions carry a weight, and which of the two
/// things SwarmUI's patch does each one gets. The split by direction is the part worth pinning — a down-weight and
/// an up-weight are not the same operation scaled differently, so a test that only checks "something changed"
/// would pass with the two swapped.</summary>
public sealed class TextTokenWeightsTests
{
    /// <summary>An unweighted prompt must not produce a patch at all. Everything downstream keys off null here:
    /// the step cache, the captured graph and DiT sharding all stay available only while there is nothing to apply.
    /// </summary>
    [Fact]
    public void AllOnesProducesNothingToApply()
    {
        Assert.Null(TextTokenWeights.TryBuild([1f, 1f, 1f, 1f], condLength: 4));
    }

    /// <summary>The template prefix the encoder drops must take its weights with it. SwarmUI right-aligns
    /// (<c>offset = cond_len − len(batch)</c>), so with 6 token weights against 4 surviving rows the first two fall
    /// off the front and the rest land on their own tokens — not shifted two places left.</summary>
    [Fact]
    public void WeightsAreRightAlignedAgainstTheRowsThatSurviveTheTemplateTrim()
    {
        TextTokenWeights? weights = TextTokenWeights.TryBuild([0.5f, 0.6f, 0.7f, 1f, 1f, 1.5f], condLength: 4);

        Assert.NotNull(weights);
        // offset = 4 − 6 = −2: entries 0 and 1 land at −2/−1 and are dropped; entry 2 lands at row 0.
        Assert.Equal([(0, 0.7f), (3, 1.5f)], weights!.Entries);
    }

    /// <summary>Below 1 scales the value rows; above 1 does not. Up-weighting through V would raise a token's
    /// contribution wherever attention already looked at it, which is a different effect from making attention look
    /// at it — SwarmUI uses the logit bias for that, and mixing the two double-applies the emphasis.</summary>
    [Fact]
    public void OnlyDownWeightsReachTheValueRows()
    {
        TextTokenWeights weights = TextTokenWeights.TryBuild([0.5f, 1f, 1.5f], condLength: 3)!;

        float[]? scale = weights.BuildValueRowScale(jointSeq: 5);
        Assert.NotNull(scale);
        Assert.Equal([0.5f, 1f, 1f, 1f, 1f], scale);
    }

    /// <summary>Above 1 becomes an additive key bias of <c>(w−1)·2</c>; below 1 does not. Zero everywhere else,
    /// because an additive mask's identity is 0 and not 1.</summary>
    [Fact]
    public void OnlyUpWeightsReachTheKeyLogitBias()
    {
        TextTokenWeights weights = TextTokenWeights.TryBuild([0.5f, 1f, 1.5f], condLength: 3)!;

        float[]? bias = weights.BuildKeyLogitBias(jointSeq: 5);
        Assert.NotNull(bias);
        Assert.Equal([0f, 0f, 1f, 0f, 0f], bias);
    }

    /// <summary>A prompt weighted in one direction only builds one of the two, so the other stays off its path
    /// entirely — no all-ones multiply over a 14 MB buffer, and no all-zero mask forcing SDPA's masked kernel.</summary>
    [Theory]
    [InlineData(0.5f, true, false)]
    [InlineData(1.5f, false, true)]
    public void EachDirectionBuildsOnlyItsOwnHalf(float weight, bool expectScale, bool expectBias)
    {
        TextTokenWeights weights = TextTokenWeights.TryBuild([weight], condLength: 1)!;

        Assert.Equal(expectScale, weights.BuildValueRowScale(jointSeq: 3) is not null);
        Assert.Equal(expectBias, weights.BuildKeyLogitBias(jointSeq: 3) is not null);
    }

    /// <summary>Image tokens follow the text in Krea 2's joint concat, so a text position is already a joint
    /// position and the arrays simply run past it unweighted. That is the layout SwarmUI's <c>seq == img_slice[1]</c>
    /// guard establishes before it applies anything.</summary>
    [Fact]
    public void ImagePositionsPastTheTextAreLeftAlone()
    {
        TextTokenWeights weights = TextTokenWeights.TryBuild([0.25f, 2f], condLength: 2)!;

        float[] scale = weights.BuildValueRowScale(jointSeq: 6)!;
        float[] bias = weights.BuildKeyLogitBias(jointSeq: 6)!;
        Assert.Equal([0.25f, 1f, 1f, 1f, 1f, 1f], scale);
        Assert.Equal([0f, 2f, 0f, 0f, 0f, 0f], bias);
    }
}
