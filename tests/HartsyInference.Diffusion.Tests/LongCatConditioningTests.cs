using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatConditioningTests
{
    [Fact]
    public void ANoReferenceSplitNeedsNoIndex()
    {
        Assert.Equal(-1, LongCatConditioning.None.RefImgIndex);
        Assert.Equal(-1, new LongCatConditioning().RefImgIndex);
        new LongCatConditioning(NumCondLatents: 2).Validate(5);
        default(LongCatConditioning).Validate(5);
    }

    [Fact]
    public void AReferenceFrameWithoutAnIndexIsRefused()
    {
        LongCatConditioning unset = LongCatConditioning.None with { NumCondLatents = 3, NumRefLatents = 1 };

        Assert.Throws<ArgumentOutOfRangeException>(() => unset.Validate(9));
    }

    [Fact]
    public void TheAvatarFactoryPlacesTheReferenceAtTheShippedIndex()
    {
        LongCatConditioning avatar = LongCatConditioning.Avatar(numCond: 3);

        Assert.Equal(new LongCatConditioning(3, 1, 10, 0), avatar);
        avatar.Validate(9);
    }

    [Fact]
    public void AnExplicitReferenceIndexIsKept()
    {
        LongCatConditioning explicitIndex = new(3, 1, 4, 1);

        Assert.Equal(4, explicitIndex.RefImgIndex);
        explicitIndex.Validate(9);
    }
}
