using HartsyInference.Audio.Models.LongCat;
using Xunit;

namespace HartsyInference.Audio.Tests.LongCat;

/// <summary>The host math between Whisper/wav2vec2 hidden states and LongCat's <c>AudioProjModel</c> input. Expectations
/// are tagged values and closed forms worked out from the reference code, not numbers read back from the code under test.</summary>
public sealed class LongCatAudioFeaturesTests
{
    // Every value encodes where it came from, so a misplaced slice shows as the wrong tag.
    private static float Tag(int frame, int window, int block) => frame * 100f + window * 10f + block;

    [Fact]
    public void PoolWhisperStates_MeansEightStatesPerBlock_AndKeepsTheLastStateAlone()
    {
        const int Frames = 2, Channels = 3;
        float[] states = new float[Frames * 33 * Channels];
        for (int t = 0; t < Frames; t++)
            for (int s = 0; s < 33; s++)
                for (int c = 0; c < Channels; c++) states[(t * 33 + s) * Channels + c] = s + 1000f * t + 0.5f * c;
        float[] pooled = new float[Frames * 5 * Channels];

        LongCatAudioFeatures.PoolWhisperStates(states, Frames, Channels, pooled);

        float[] meanOfStates = [3.5f, 11.5f, 19.5f, 27.5f, 32f];
        for (int t = 0; t < Frames; t++)
            for (int b = 0; b < 5; b++)
                for (int c = 0; c < Channels; c++)
                    Assert.Equal(meanOfStates[b] + 1000f * t + 0.5f * c, pooled[(t * 5 + b) * Channels + c], 4);
    }

    [Fact]
    public void PoolWhisperStates_RejectsAWrongStateCount()
    {
        Assert.Throws<ArgumentException>(() => LongCatAudioFeatures.PoolWhisperStates(new float[32 * 4], 1, 4, new float[5 * 4]));
    }

    [Fact]
    public void ResampleFrames_KeepsBlocksApart_AndBothEnds()
    {
        // 2 blocks x 1 channel, 3 frames: block 0 = 0,10,20 and block 1 = 100,90,80.
        float[] features = [0f, 100f, 10f, 90f, 20f, 80f];

        float[] resampled = LongCatAudioFeatures.ResampleFrames(features, 3, 2, 1, 5);

        Assert.Equal([0f, 100f, 5f, 95f, 10f, 90f, 15f, 85f, 20f, 80f], resampled);
    }

    [Fact]
    public void VideoFrames_IsAnIntegerOfSecondsTimesFps()
    {
        Assert.Equal(25 * 6, LongCatAudioFeatures.VideoFrames(16_000 * 6, 16_000, 25));
        Assert.Equal(12, LongCatAudioFeatures.VideoFrames(16_000 * 3 / 4, 16_000, 16));
    }

    [Fact]
    public void BuildWindows_CentresEachFrameAndClampsAtBothEnds()
    {
        LongCatAudioLayout layout = new() { Blocks = 1, Channels = 1, VideoFps = 25, AudioStride = 1 };
        float[] features = [0f, 1f, 2f, 3f, 4f, 5f];

        float[] windows = LongCatAudioFeatures.BuildWindows(features, 6, layout, startFrame: 0, count: 6);

        float[][] expected =
        [
            [0f, 0f, 0f, 1f, 2f], [0f, 0f, 1f, 2f, 3f], [0f, 1f, 2f, 3f, 4f],
            [1f, 2f, 3f, 4f, 5f], [2f, 3f, 4f, 5f, 5f], [3f, 4f, 5f, 5f, 5f],
        ];
        for (int i = 0; i < 6; i++) Assert.Equal(expected[i], windows[(i * 5)..(i * 5 + 5)]);
    }

    [Fact]
    public void BuildWindows_StridesOverTheFeatureRate_AndStartsAtTheSegment()
    {
        // Avatar 1.0: features run at twice the video rate, so frame i is centred on feature 2i.
        LongCatAudioLayout layout = new() { Blocks = 1, Channels = 1, VideoFps = 16, AudioStride = 2 };
        float[] features = Enumerable.Range(0, 20).Select(i => (float)i).ToArray();

        float[] windows = LongCatAudioFeatures.BuildWindows(features, 20, layout, startFrame: 6, count: 3);

        Assert.Equal([4f, 5f, 6f, 7f, 8f, 6f, 7f, 8f, 9f, 10f, 8f, 9f, 10f, 11f, 12f], windows);
    }

    [Fact]
    public void BuildWindows_KeepsBlockAndChannelOrderInsideAWindow()
    {
        LongCatAudioLayout layout = new() { Blocks = 2, Channels = 3, VideoFps = 25, AudioStride = 1 };
        float[] features = new float[4 * 2 * 3];
        for (int f = 0; f < 4; f++)
            for (int b = 0; b < 2; b++)
                for (int c = 0; c < 3; c++) features[(f * 2 + b) * 3 + c] = f * 100 + b * 10 + c;

        float[] windows = LongCatAudioFeatures.BuildWindows(features, 4, layout, startFrame: 1, count: 1);

        // Centre 1, window 5: feature frames -1,0,1,2,3 clamp to 0,0,1,2,3, each with its [block, channel] slab.
        int[] frames = [0, 0, 1, 2, 3];
        for (int w = 0; w < 5; w++)
            for (int b = 0; b < 2; b++)
                for (int c = 0; c < 3; c++)
                    Assert.Equal(frames[w] * 100 + b * 10 + c, windows[((w * 2) + b) * 3 + c]);
    }

    [Theory]
    [InlineData(0, 0)]
    [InlineData(1, 80)]
    [InlineData(3, 240)]
    public void SegmentStartFrame_AdvancesByTheNewFramesOfEachSegment(int segment, int expected)
        => Assert.Equal(expected, LongCatAudioFeatures.SegmentStartFrame(segment, LongCatAudioLayout.Whisper));

    [Fact]
    public void SegmentStartFrame_ScalesWithTheStride()
        => Assert.Equal(160, LongCatAudioFeatures.SegmentStartFrame(1, LongCatAudioLayout.Wav2Vec2));

    [Fact]
    public void GroupForAudioProj_FollowsTheReferenceSliceOrder()
    {
        LongCatAudioLayout layout = new() { Blocks = 1, Channels = 1, VideoFps = 25, AudioStride = 1 };
        const int Frames = 9;
        float[] windows = new float[Frames * 5];
        for (int f = 0; f < Frames; f++)
            for (int w = 0; w < 5; w++) windows[f * 5 + w] = Tag(f, w, 0);

        (float[] first, float[] later) = LongCatAudioFeatures.GroupForAudioProj(windows, Frames, layout);

        Assert.Equal([Tag(0, 0, 0), Tag(0, 1, 0), Tag(0, 2, 0), Tag(0, 3, 0), Tag(0, 4, 0)], first);
        Assert.Equal(2 * 8, later.Length);
        // Group 0 spans video frames 1..4: frame 1 windows 0-2, frames 2 and 3 window 2, frame 4 windows 2-4.
        Assert.Equal(
            [Tag(1, 0, 0), Tag(1, 1, 0), Tag(1, 2, 0), Tag(2, 2, 0), Tag(3, 2, 0), Tag(4, 2, 0), Tag(4, 3, 0), Tag(4, 4, 0)],
            later[..8]);
        Assert.Equal(
            [Tag(5, 0, 0), Tag(5, 1, 0), Tag(5, 2, 0), Tag(6, 2, 0), Tag(7, 2, 0), Tag(8, 2, 0), Tag(8, 3, 0), Tag(8, 4, 0)],
            later[8..]);
    }

    [Fact]
    public void GroupForAudioProj_KeepsBlockAndChannelOrderWithinASlice()
    {
        LongCatAudioLayout layout = new() { Blocks = 2, Channels = 2, VideoFps = 25, AudioStride = 1 };
        float[] windows = new float[5 * 5 * 2 * 2];
        for (int f = 0; f < 5; f++)
            for (int w = 0; w < 5; w++)
                for (int b = 0; b < 2; b++)
                    for (int c = 0; c < 2; c++) windows[(((f * 5) + w) * 2 + b) * 2 + c] = Tag(f, w, b) + c / 10f;

        (_, float[] later) = LongCatAudioFeatures.GroupForAudioProj(windows, 5, layout);

        Assert.Equal(8 * 2 * 2, later.Length);
        // Slice 3 of the group is frame 2, window 2: block 0 channels 0,1 then block 1 channels 0,1.
        Assert.Equal([Tag(2, 2, 0), Tag(2, 2, 0) + 0.1f, Tag(2, 2, 1), Tag(2, 2, 1) + 0.1f], later[(3 * 4)..(4 * 4)]);
    }

    [Fact]
    public void GroupForAudioProj_TheAvatarSegmentFeedsTheShippedProjectionWidths()
    {
        foreach ((LongCatAudioLayout layout, int proj1, int proj1Vf) in new[]
                 {
                     (LongCatAudioLayout.Whisper, 32_000, 51_200),
                     (LongCatAudioLayout.Wav2Vec2, 46_080, 73_728),
                 })
        {
            Assert.Equal(proj1, layout.WindowValues);
            Assert.Equal(proj1Vf, layout.GroupValues);
            float[] windows = new float[93 * layout.WindowValues];

            (float[] first, float[] later) = LongCatAudioFeatures.GroupForAudioProj(windows, 93, layout);

            Assert.Equal(proj1, first.Length);
            Assert.Equal(23 * proj1Vf, later.Length);
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(92)]
    [InlineData(94)]
    public void GroupForAudioProj_RejectsAFrameCountThatIsNotOnePlusMultiplesOfTheScale(int frames)
    {
        Assert.Throws<ArgumentException>(() => LongCatAudioFeatures.GroupForAudioProj(
            new float[frames * LongCatAudioLayout.Whisper.WindowValues], frames, LongCatAudioLayout.Whisper));
    }

    [Fact]
    public void SkipEmbeddingState_DropsOnlyTheFirstOfThirteenStates()
    {
        float[] states = new float[2 * 13 * 3];
        for (int t = 0; t < 2; t++)
            for (int s = 0; s < 13; s++)
                for (int c = 0; c < 3; c++) states[(t * 13 + s) * 3 + c] = t * 1000 + s * 10 + c;

        float[] layers = LongCatAudioFeatures.SkipEmbeddingState(states, 2, 13, 3);

        Assert.Equal(2 * 12 * 3, layers.Length);
        for (int t = 0; t < 2; t++)
            for (int s = 0; s < 12; s++)
                for (int c = 0; c < 3; c++) Assert.Equal(t * 1000 + (s + 1) * 10 + c, layers[(t * 12 + s) * 3 + c]);
    }

    [Fact]
    public void ABlockOfWhisperFeatures_RoundTripsFromEncoderRateToWindows()
    {
        // 6 s of audio at 50 Hz encoder frames, one block of ramp-valued channels: after the 25 fps resample the window of
        // video frame f reads the ramp's value at feature positions f-2..f+2 (clamped).
        const int VideoFrames = 150;
        const int EncoderFrames = 2 * VideoFrames;
        float[] pooled = new float[EncoderFrames * 5];
        for (int t = 0; t < EncoderFrames; t++)
            for (int b = 0; b < 5; b++) pooled[t * 5 + b] = t;
        LongCatAudioLayout layout = new() { Blocks = 5, Channels = 1, VideoFps = 25, AudioStride = 1 };

        float[] features = LongCatAudioFeatures.ResampleFrames(pooled, EncoderFrames, 5, 1, VideoFrames);
        float[] windows = LongCatAudioFeatures.BuildWindows(features, VideoFrames, layout, startFrame: 80, count: 93);

        double step = (EncoderFrames - 1.0) / (VideoFrames - 1);
        for (int i = 0; i < 93; i++)
            for (int w = 0; w < 5; w++)
            {
                int source = Math.Clamp(80 + i + w - 2, 0, VideoFrames - 1);
                Assert.Equal(source * step, windows[(i * 5 + w) * 5], 2);
            }
    }
}
