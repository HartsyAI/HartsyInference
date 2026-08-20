using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Video.Pipelines;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Pins the arithmetic of Wan-Animate's chunked extension (ComfyUI <c>WanAnimateToVideo.continue_motion</c>):
/// the prefix → latent-frame → trim derivation, which concat-mask cells the prefix marks known, the driving-video
/// offset chain across successive chunks, and the seeked/pinned driving-frame rules. All of it is silent when wrong —
/// an off-by-one here still renders a video, just one that jumps or repeats at every chunk seam.</summary>
public sealed class WanAnimateContinueMotionTests
{
    private const int Step = 4;   // Wan VAE temporal compression

    [Theory]
    [InlineData(0, 0, 0)]      // first chunk: no prefix, nothing to trim
    [InlineData(1, 1, 1)]      // on-grid
    [InlineData(5, 2, 5)]      // the reference default
    [InlineData(9, 3, 9)]
    [InlineData(13, 4, 13)]
    [InlineData(2, 1, 1)]      // off-grid: trim UNDER-drops, leaking re-rendered prefix frames
    [InlineData(3, 1, 1)]
    [InlineData(4, 1, 1)]
    [InlineData(8, 2, 5)]
    [InlineData(12, 3, 9)]
    public void PrefixDerivesItsLatentLengthAndTrim(int motionFrames, int expectedRefLatent, int expectedTrim)
    {
        int refLatent = WanAnimateChunkMath.RefMotionLatentLength(motionFrames);
        Assert.Equal(expectedRefLatent, refLatent);
        Assert.Equal(expectedTrim, WanAnimateChunkMath.TrimImageFrames(refLatent));
    }

    [Fact]
    public void TrimEqualsThePrefixExactlyOnTheGrid()
    {
        for (int n = 1; n <= 40; n++)
        {
            int trim = WanAnimateChunkMath.TrimImageFrames(WanAnimateChunkMath.RefMotionLatentLength(n));
            if (n % Step == 1)
            {
                Assert.Equal(n, trim);
            }
            else
            {
                Assert.True(trim < n, $"n={n} → trim {trim} should under-drop off-grid");
            }
        }
    }

    [Theory]
    [InlineData(0, 0)]
    [InlineData(1, 1)]
    [InlineData(4, 1)]
    [InlineData(5, 5)]
    [InlineData(8, 5)]
    [InlineData(9, 9)]
    [InlineData(20, 17)]
    public void SnapMotionFramesFloorsOntoTheFourNPlusOneGrid(int requested, int expected)
    {
        int snapped = WanAnimateChunkMath.SnapMotionFrames(requested);
        Assert.Equal(expected, snapped);
        Assert.True(snapped <= requested);
        Assert.True(snapped == 0 || snapped % Step == 1);
    }

    [Theory]
    [InlineData(5, 200, 81, 5)]     // steady state: the request wins
    [InlineData(9, 200, 81, 9)]
    [InlineData(81, 200, 81, 77)]   // clamped to chunkLen - 1, then snapped (80 → 77)
    [InlineData(5, 3, 81, 1)]       // only 3 frames generated so far → snaps to a 1-frame prefix
    [InlineData(9, 6, 81, 5)]
    public void MotionPrefixIsClampedByOutputAndChunkLength(int requested, int available, int chunkFrames, int expected)
    {
        Assert.Equal(expected, WanAnimateChunkMath.MotionPrefixFrames(requested, available, chunkFrames));
    }

    [Fact]
    public void MaskPrefixEndIsRefTimesFourWithoutACharacterMask()
    {
        Assert.Equal(0, WanAnimateChunkMath.MaskPrefixEnd(0, hasCharacterMask: false));
        Assert.Equal(4, WanAnimateChunkMath.MaskPrefixEnd(1, hasCharacterMask: false));
        Assert.Equal(8, WanAnimateChunkMath.MaskPrefixEnd(2, hasCharacterMask: false));
        Assert.Equal(12, WanAnimateChunkMath.MaskPrefixEnd(3, hasCharacterMask: false));
    }

    /// <summary>Upstream zeroes <c>[0, ref·4)</c> and THEN overwrites <c>[ref_images_num, …)</c> with the character
    /// mask, so flat indices 5/6/7 of a 5-frame prefix carry mask values rather than "known". Replicated verbatim —
    /// this test exists so a future "cleanup" of that overlap is a deliberate, visible change.</summary>
    [Fact]
    public void ACharacterMaskShortensTheKnownRunToTheTrimCount()
    {
        Assert.Equal(5, WanAnimateChunkMath.MaskPrefixEnd(2, hasCharacterMask: true));
        Assert.Equal(8, WanAnimateChunkMath.MaskPrefixEnd(2, hasCharacterMask: false));
        Assert.Equal(1, WanAnimateChunkMath.MaskPrefixEnd(1, hasCharacterMask: true));
    }

    [Fact]
    public void APrefixMarksExactlyTheLeadingLatentFramesKnown()
    {
        const int TrimLatent = 1, MaskChannels = 4, TotalLatent = 22;
        foreach (int motionFrames in new[] { 1, 5, 9, 13 })
        {
            int refLatent = WanAnimateChunkMath.RefMotionLatentLength(motionFrames);
            int prefixEnd = WanAnimateChunkMath.MaskPrefixEnd(refLatent, hasCharacterMask: false);
            for (int t = 0; t < TotalLatent; t++)
            {
                for (int m = 0; m < MaskChannels; m++)
                {
                    bool expected = t < TrimLatent || t - TrimLatent < refLatent;
                    Assert.Equal(expected, WanAnimateChunkMath.IsKnownMaskCell(t, m, TrimLatent, prefixEnd));
                }
            }
        }
    }

    [Fact]
    public void NoPrefixLeavesOnlyTheReferenceFrameKnown()
    {
        int prefixEnd = WanAnimateChunkMath.MaskPrefixEnd(0, hasCharacterMask: false);
        Assert.True(WanAnimateChunkMath.IsKnownMaskCell(0, 0, trimLatent: 1, prefixEnd));
        for (int t = 1; t < 8; t++)
        {
            for (int m = 0; m < 4; m++)
            {
                Assert.False(WanAnimateChunkMath.IsKnownMaskCell(t, m, trimLatent: 1, prefixEnd));
            }
        }
    }

    /// <summary>The one cell-level consequence of the charmask overlap: latent frame <c>ref-1</c>, mask channels 1..3
    /// of a 5-frame prefix stop being "known" once a character mask is supplied.</summary>
    [Fact]
    public void TheCharacterMaskOverlapUnmarksTheTailOfTheLastPrefixLatentFrame()
    {
        int prefixEnd = WanAnimateChunkMath.MaskPrefixEnd(2, hasCharacterMask: true);
        Assert.True(WanAnimateChunkMath.IsKnownMaskCell(1, 0, trimLatent: 1, prefixEnd));    // j = 0
        Assert.True(WanAnimateChunkMath.IsKnownMaskCell(2, 0, trimLatent: 1, prefixEnd));    // j = 4
        Assert.False(WanAnimateChunkMath.IsKnownMaskCell(2, 1, trimLatent: 1, prefixEnd));   // j = 5
        Assert.False(WanAnimateChunkMath.IsKnownMaskCell(2, 3, trimLatent: 1, prefixEnd));   // j = 7
    }

    /// <summary>The rewind happens BEFORE each chunk's driving slice, so slice offsets advance by exactly the number
    /// of NEW frames a chunk contributes and driving-frame index stays locked to output-frame index.</summary>
    [Fact]
    public void OffsetChainRewindsBeforeEveryChunkSlice()
    {
        const int ChunkLen = 81, Prefix = 5;
        int carried = 0;
        List<int> sliceOffsets = [];
        for (int chunk = 0; chunk < 4; chunk++)
        {
            int prefix = chunk == 0 ? 0 : Prefix;
            int sliceOffset = WanAnimateChunkMath.SliceOffset(carried, prefix);
            sliceOffsets.Add(sliceOffset);
            carried = WanAnimateChunkMath.NextCarriedOffset(sliceOffset, ChunkLen);
        }
        Assert.Equal([0, 76, 152, 228], sliceOffsets);
        for (int chunk = 1; chunk < sliceOffsets.Count; chunk++)
        {
            Assert.Equal(ChunkLen - Prefix, sliceOffsets[chunk] - sliceOffsets[chunk - 1]);
        }
    }

    [Fact]
    public void OffsetNeverGoesNegativeWhenThePrefixExceedsTheCarriedOffset()
    {
        Assert.Equal(0, WanAnimateChunkMath.SliceOffset(carriedOffset: 3, motionFrames: 9));
        Assert.Equal(0, WanAnimateChunkMath.SliceOffset(carriedOffset: 0, motionFrames: 5));
    }

    [Theory]
    [InlineData(81, 81, 5, 1)]
    [InlineData(81, 81, 0, 1)]
    [InlineData(157, 81, 5, 2)]     // 81 + 76 = 157 exactly
    [InlineData(158, 81, 5, 3)]
    [InlineData(300, 81, 5, 4)]     // 81 + 3*76 = 309 ≥ 300
    public void ChunkCountCoversTheRequestedTotal(int total, int chunkFrames, int motionFrames, int expected)
    {
        int chunks = WanAnimateChunkMath.ChunkCount(total, chunkFrames, motionFrames);
        Assert.Equal(expected, chunks);
        int trim = WanAnimateChunkMath.TrimImageFrames(WanAnimateChunkMath.RefMotionLatentLength(motionFrames));
        Assert.True(chunkFrames + (chunks - 1) * (chunkFrames - trim) >= total);
    }

    [Fact]
    public void ChunkCountRejectsAPrefixThatLeavesNoNewFrames()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => WanAnimateChunkMath.ChunkCount(200, chunkFrames: 5, motionFrames: 5));
    }

    [Fact]
    public void DropLeadingFramesSeeksInPlace()
    {
        List<byte[]> frames = [.. Enumerable.Range(0, 10).Select(i => new byte[] { (byte)i })];
        WanAnimateDrivingResolver.DropLeadingFrames(frames, 4);
        Assert.Equal(6, frames.Count);
        Assert.Equal([4, 5, 6, 7, 8, 9], frames.Select(f => (int)f[0]));
    }

    [Fact]
    public void DropLeadingFramesEmptiesAnExhaustedClip()
    {
        List<byte[]> frames = [.. Enumerable.Range(0, 3).Select(i => new byte[] { (byte)i })];
        Assert.Empty(WanAnimateDrivingResolver.DropLeadingFrames(frames, 3));
        List<byte[]> more = [.. Enumerable.Range(0, 3).Select(i => new byte[] { (byte)i })];
        Assert.Empty(WanAnimateDrivingResolver.DropLeadingFrames(more, 99));
    }

    [Fact]
    public void DropLeadingFramesIsANoOpAtOffsetZero()
    {
        List<byte[]> frames = [.. Enumerable.Range(0, 3).Select(i => new byte[] { (byte)i })];
        Assert.Equal(3, WanAnimateDrivingResolver.DropLeadingFrames(frames, 0).Count);
        Assert.Equal(3, WanAnimateDrivingResolver.DropLeadingFrames(frames, -7).Count);
    }

    /// <summary>A continuation chunk PINS its frame count. Letting the shrink rule run there would change the latent
    /// geometry mid-sequence the moment the seeked driving video ran short — which happens on the last chunk of
    /// nearly every run.</summary>
    [Theory]
    [InlineData(81, 24, true, 81)]
    [InlineData(81, 1, true, 81)]
    [InlineData(81, 24, false, 21)]
    [InlineData(81, 200, false, 81)]
    [InlineData(81, 50, false, 49)]
    public void ContinuationChunksPinTheFrameCountAndChunkZeroDoesNot(int requested, int available, bool pinned, int expected)
    {
        Assert.Equal(expected, WanAnimateDrivingResolver.ResolveChunkFrames(requested, available, Step, pinned));
    }

    /// <summary>End-to-end of the short-source path: seek past most of a 100-frame driving video, pin the count, and
    /// the tail repeat-pads its last frame — upstream's pose semantics, and the reason the geometry stays constant.</summary>
    [Fact]
    public void ASeekedShortDrivingClipRepeatPadsItsLastFrameUpToThePinnedCount()
    {
        const int Decoded = 100, Offset = 76, Requested = 81;
        List<byte[]> frames = [.. Enumerable.Range(0, Decoded).Select(i => new byte[] { (byte)(i % 256) })];
        WanAnimateDrivingResolver.DropLeadingFrames(frames, Offset);
        Assert.Equal(Decoded - Offset, frames.Count);
        int count = WanAnimateDrivingResolver.ResolveChunkFrames(Requested, frames.Count, Step, pinFrameCount: true);
        Assert.Equal(Requested, count);
        List<byte[]> fitted = WanAnimateDrivingResolver.FitFrames(frames, count);
        Assert.Equal(Requested, fitted.Count);
        Assert.Equal(Offset, fitted[0][0]);
        Assert.Equal(Decoded - 1, fitted[Decoded - Offset - 1][0]);
        for (int i = Decoded - Offset; i < Requested; i++)
        {
            Assert.Same(fitted[Decoded - Offset - 1], fitted[i]);
        }
    }

    /// <summary>Three chunks of driving frames end-to-end: every chunk consumes <c>chunkLen</c> frames starting at its
    /// rewound offset, so chunk k+1's window re-covers exactly the prefix frames chunk k ended on.</summary>
    [Fact]
    public void SuccessiveChunkWindowsOverlapByExactlyThePrefix()
    {
        const int ChunkLen = 21, Prefix = 5, Source = 200;
        List<byte[]> source = [.. Enumerable.Range(0, Source).Select(i => new byte[] { (byte)i })];
        int carried = 0;
        List<int> firstFrame = [], lastFrame = [];
        for (int chunk = 0; chunk < 3; chunk++)
        {
            int prefix = chunk == 0 ? 0 : Prefix;
            int sliceOffset = WanAnimateChunkMath.SliceOffset(carried, prefix);
            List<byte[]> window = [.. source];
            WanAnimateDrivingResolver.DropLeadingFrames(window, sliceOffset);
            int count = WanAnimateDrivingResolver.ResolveChunkFrames(ChunkLen, window.Count, Step, pinFrameCount: chunk > 0);
            WanAnimateDrivingResolver.FitFrames(window, count);
            Assert.Equal(ChunkLen, window.Count);
            firstFrame.Add(window[0][0]);
            lastFrame.Add(window[^1][0]);
            carried = WanAnimateChunkMath.NextCarriedOffset(sliceOffset, ChunkLen);
        }
        Assert.Equal([0, 16, 32], firstFrame);
        Assert.Equal([20, 36, 52], lastFrame);
        // Chunk 1 restarts 5 frames before chunk 0 ended (16..20 == the prefix it carries).
        Assert.Equal(Prefix - 1, lastFrame[0] - firstFrame[1]);
    }
}
