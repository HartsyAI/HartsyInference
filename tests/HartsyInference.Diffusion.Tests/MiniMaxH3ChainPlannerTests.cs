using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Engine.Recipes.Video;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Geometry gates for long-form MiniMax-H3 chaining. A protected head that ended inside a latent token
/// would hold part of it fixed while denoising the rest, which the sampler's row masks cannot express.</summary>
public class MiniMaxH3ChainPlannerTests
{
    [Fact]
    public void TargetInsideOneGenerationDoesNotChain()
    {
        IReadOnlyList<MiniMaxH3ChainPlanner.Segment> plan = MiniMaxH3ChainPlanner.Plan(141);
        MiniMaxH3ChainPlanner.Segment only = Assert.Single(plan);
        Assert.Equal(141, only.FrameCount);
        Assert.Equal(0, only.ContextFrames);
        Assert.Equal(141, only.NewFrames);
    }

    [Fact]
    public void TargetPastOneGenerationChainsAndReachesIt()
    {
        const int Target = 900;
        IReadOnlyList<MiniMaxH3ChainPlanner.Segment> plan = MiniMaxH3ChainPlanner.Plan(Target);
        Assert.True(plan.Count > 1, "A target past the single-generation envelope must chain.");
        Assert.True(MiniMaxH3ChainPlanner.TotalFrames(plan) >= Target,
            $"Chain produced {MiniMaxH3ChainPlanner.TotalFrames(plan)} frames for a {Target}-frame target.");
    }

    [Theory]
    [InlineData(400)]
    [InlineData(600)]
    [InlineData(900)]
    [InlineData(1500)]
    [InlineData(3000)]
    public void EverySegmentStaysOnTheGridAndMakesProgress(int target)
    {
        IReadOnlyList<MiniMaxH3ChainPlanner.Segment> plan = MiniMaxH3ChainPlanner.Plan(target);
        Assert.All(plan, segment =>
        {
            Assert.Equal(5, segment.FrameCount % 17);
            Assert.True(segment.FrameCount <= MiniMaxH3Geometry.TrainedFrameEnvelope);
            Assert.True(segment.NewFrames > 0, "Every segment must contribute new frames or the chain cannot end.");
        });
        Assert.Equal(0, plan[0].ContextFrames);
        Assert.All(plan.Skip(1), segment =>
            Assert.Equal(MiniMaxH3ChainPlanner.DefaultContextFrames, segment.ContextFrames));
        Assert.True(MiniMaxH3ChainPlanner.TotalFrames(plan) >= target);
    }

    /// <summary>The protected head must cover exactly the context frames, with no partial token.</summary>
    [Theory]
    [InlineData(39)]
    [InlineData(90)]
    [InlineData(141)]
    public void ProtectedHeadCoversWholeLatentTokens(int contextFrames)
    {
        IReadOnlyList<MiniMaxH3ChainPlanner.Segment> plan =
            MiniMaxH3ChainPlanner.Plan(900, contextFrames);
        MiniMaxH3ChainPlanner.Segment second = plan[1];
        Assert.Equal(MiniMaxH3Geometry.VideoLatentFrames(contextFrames), second.ContextLatentFrames);
        Assert.Equal(MiniMaxH3Geometry.AudioLatentFrames(contextFrames), second.ContextAudioLatentFrames);
        // 17k+5 frames occupy 5k+2 tokens, and those tokens span exactly 17k+5 pixel frames.
        Assert.Equal(contextFrames, PixelFramesCovered(second.ContextLatentFrames));
    }

    [Fact]
    public void MaskValuesProtectTheHeadAndGenerateTheRest()
    {
        IReadOnlyList<MiniMaxH3ChainPlanner.Segment> plan = MiniMaxH3ChainPlanner.Plan(900);
        MiniMaxH3ChainPlanner.Segment second = plan[1];

        float[] video = MiniMaxH3ChainPlanner.VideoMaskFrameValues(second);
        Assert.Equal(MiniMaxH3Geometry.VideoLatentFrames(second.FrameCount), video.Length);
        Assert.All(video.Take(second.ContextLatentFrames), value => Assert.Equal(0f, value));
        Assert.All(video.Skip(second.ContextLatentFrames), value => Assert.Equal(1f, value));

        float[] audio = MiniMaxH3ChainPlanner.AudioMaskValues(second);
        Assert.Equal(MiniMaxH3Geometry.AudioLatentFrames(second.FrameCount), audio.Length);
        Assert.All(audio.Take(second.ContextAudioLatentFrames), value => Assert.Equal(0f, value));
        Assert.All(audio.Skip(second.ContextAudioLatentFrames), value => Assert.Equal(1f, value));
    }

    /// <summary>The first segment is an ordinary generation, so its mask is all-generate.</summary>
    [Fact]
    public void FirstSegmentMaskPreservesNothing()
    {
        MiniMaxH3ChainPlanner.Segment first = MiniMaxH3ChainPlanner.Plan(900)[0];
        Assert.All(MiniMaxH3ChainPlanner.VideoMaskFrameValues(first), value => Assert.Equal(1f, value));
        Assert.All(MiniMaxH3ChainPlanner.AudioMaskValues(first), value => Assert.Equal(1f, value));
    }

    [Theory]
    [InlineData(40)]   // off-grid
    [InlineData(4)]    // below the grid floor
    [InlineData(362)]  // leaves no new frames in a capped segment
    public void OffGridOrOversizedContextIsRefused(int contextFrames)
        => Assert.Throws<ArgumentOutOfRangeException>(() => MiniMaxH3ChainPlanner.Plan(900, contextFrames));

    [Fact]
    public void TooShortATargetIsRefused()
        => Assert.Throws<ArgumentOutOfRangeException>(() => MiniMaxH3ChainPlanner.Plan(4));

    /// <summary>Pixel frames spanned by <paramref name="latentTokens"/> tokens of the <c>{1,4,4,4,4}</c> cycle.</summary>
    private static int PixelFramesCovered(int latentTokens)
    {
        int[] framePerToken = [1, 4, 4, 4, 4];
        int total = 0;
        for (int k = 0; k < latentTokens; k++)
        {
            total += framePerToken[k % 5];
        }
        return total;
    }
}
