using HartsyInference.ModelAssets.Duo;
using Xunit;

namespace HartsyInference.ModelAssets.Tests.Duo;

/// <summary>The <c>duo_5_17</c> codec geometry and decode schedule, checked against the counts upstream's decoder is driven with. These
/// are bookkeeping tests; the VAE math itself is not run here (parity against upstream is not run).</summary>
public sealed class DuoVideoDecodePlanTests
{
    [Fact]
    public void Duo5x17HasSeventeenFramesFiveLatentsAndTwentyNineAudioLatents()
    {
        DuoCodecSpec spec = DuoCodecSpec.Get("duo_5_17");

        Assert.Equal(17, spec.Frames);
        Assert.Equal(5, spec.Latents);
        // ceil(17 * 32000 / (24 * 800)) = ceil(28.33) = 29 audio latent steps per channel.
        Assert.Equal(29, spec.AudioLatents);
        Assert.True(spec.Delayed);
    }

    [Fact]
    public void AnUnknownModeIsRefusedNotGuessed()
    {
        Assert.Throws<ArgumentException>(() => DuoCodecSpec.Get("duo_5_18"));
    }

    [Fact]
    public void TheFirstChunkPublishesNothing()
    {
        Assert.Null(DuoVideoDecodePlan.OnChunk(0));
        Assert.Null(DuoVideoDecodePlan.AtEnd(0));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(6)]
    public void EveryBlockIsPublishedOnceInOrderWithSeventeenFrames(int chunks)
    {
        List<DuoDecodeStep> steps = [];
        for (int c = 1; c < chunks; c++)
        {
            steps.Add(DuoVideoDecodePlan.OnChunk(c)!.Value);
        }
        steps.Add(DuoVideoDecodePlan.AtEnd(chunks)!.Value);

        Assert.Equal(Enumerable.Range(0, chunks), steps.Select(s => s.Block));
        Assert.All(steps, s => Assert.Equal(17, s.CropEnd - s.CropStart));
    }

    [Fact]
    public void WindowsAreSevenLatentsForTheFirstBlockAndTwelveAfterwards()
    {
        DuoDecodeStep first = DuoVideoDecodePlan.OnChunk(1)!.Value;
        Assert.Equal(7, first.WindowLatents);
        Assert.Equal(22, first.RawFrames);
        Assert.Equal((0, 17), (first.CropStart, first.CropEnd));

        for (int c = 2; c < 6; c++)
        {
            DuoDecodeStep step = DuoVideoDecodePlan.OnChunk(c)!.Value;
            Assert.Equal(12, step.WindowLatents);
            Assert.Equal(39, step.RawFrames);
            Assert.Equal((17, 34), (step.CropStart, step.CropEnd));
        }
    }

    [Fact]
    public void EofWithOneChunkDecodesSevenLatentsAndWithMoreDecodesTwelve()
    {
        DuoDecodeStep one = DuoVideoDecodePlan.AtEnd(1)!.Value;
        Assert.Equal((0, 0, 7, 22), (one.Block, one.FirstChunk, one.WindowLatents, one.RawFrames));
        Assert.Equal((0, 17), (one.CropStart, one.CropEnd));

        DuoDecodeStep many = DuoVideoDecodePlan.AtEnd(4)!.Value;
        Assert.Equal((3, 2, 3, 12, 39), (many.Block, many.FirstChunk, many.LastChunk, many.WindowLatents, many.RawFrames));
        Assert.Equal((17, 34), (many.CropStart, many.CropEnd));
    }

    [Fact]
    public void EachWindowReadsOnlyChunksAlreadyReceivedPlusTheTwoLookaheadLatents()
    {
        for (int c = 1; c < 8; c++)
        {
            DuoDecodeStep step = DuoVideoDecodePlan.OnChunk(c)!.Value;
            Assert.True(step.LastChunk == c, "The window ends on the chunk that just arrived.");
            // Whole chunks, then two latents of the newest chunk: never more than the 2-latent lookahead beyond what was received.
            int wholeChunks = step.LastChunk - step.FirstChunk;
            Assert.Equal(wholeChunks * 5 + 2, step.WindowLatents);
        }
    }
}
