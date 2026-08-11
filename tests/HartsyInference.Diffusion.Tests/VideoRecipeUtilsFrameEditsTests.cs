using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Pure-logic regression test for <c>VideoRecipeUtils.ToVideoFrames</c>'s trim/boomerang math — the
/// engine-side half of a real bug found while scoping Tier 3.5 (streaming video decode): every
/// <c>*RecipePipeline.Generate</c> already routes through <c>ToResult</c>/<c>ToVideoFrames</c>, which applies
/// <c>VideoRequest.VideoBoomerang</c> here, BEFORE the frames ever leave the engine. The extension's
/// <c>HartsyInferenceBackend.GenerateVideo</c> (SwarmUI-HartsyInference-Backend repo) then applied boomerang a
/// SECOND time on the already-boomeranged result — a request with Boomerang checked produced a ping-pong of a
/// ping-pong, not a single loop, on every video family (all ten <c>*RecipePipeline</c>s route through this same
/// helper, confirmed by grep). Fixed by deleting the extension's redundant re-application; this test locks in
/// that the engine's single application is itself correct, so removing the extension's copy couldn't silently
/// leave boomerang unapplied.</summary>
public sealed class VideoRecipeUtilsFrameEditsTests
{
    private static byte[][] MakeFrames(int count) =>
        [.. Enumerable.Range(0, count).Select(i => new byte[] { (byte)i })];

    [Fact]
    public void ToVideoFrames_BoomerangOnly_ProducesSinglePingPongNotDouble()
    {
        byte[][] frames = MakeFrames(5); // [0,1,2,3,4]
        VideoRequest request = new VideoRequest { Prompt = "x", VideoBoomerang = true };

        IReadOnlyList<VideoFrame> result = VideoRecipeUtils.ToVideoFrames(frames, width: 4, height: 4, request);

        // 2N-2, not 2N (a double-apply would either throw on the already-doubled input length or produce a
        // second full reversal on top of the first, landing on 2*(2N-2)-2 frames instead).
        Assert.Equal(8, result.Count);
        byte[] indices = [.. result.Select(f => f.Rgb[0])];
        Assert.Equal<byte>([0, 1, 2, 3, 4, 3, 2, 1], indices);
    }

    [Fact]
    public void ToVideoFrames_TrimThenBoomerang_TrimsFirstThenLoopsTheTrimmedSequence()
    {
        byte[][] frames = MakeFrames(7); // [0..6]
        VideoRequest request = new VideoRequest
        {
            Prompt = "x",
            TrimVideoStartFrames = 1,
            TrimVideoEndFrames = 1,
            VideoBoomerang = true,
        };

        IReadOnlyList<VideoFrame> result = VideoRecipeUtils.ToVideoFrames(frames, width: 4, height: 4, request);

        // Trim to [1,2,3,4,5] (5 frames) first, then boomerang: [1,2,3,4,5,4,3,2] (2*5-2=8).
        byte[] indices = [.. result.Select(f => f.Rgb[0])];
        Assert.Equal<byte>([1, 2, 3, 4, 5, 4, 3, 2], indices);
    }

    [Fact]
    public void ToVideoFrames_NoEdits_PassesFramesThroughUnchanged()
    {
        byte[][] frames = MakeFrames(4);
        VideoRequest request = new VideoRequest { Prompt = "x" };

        IReadOnlyList<VideoFrame> result = VideoRecipeUtils.ToVideoFrames(frames, width: 4, height: 4, request);

        byte[] indices = [.. result.Select(f => f.Rgb[0])];
        Assert.Equal<byte>([0, 1, 2, 3], indices);
    }
}
