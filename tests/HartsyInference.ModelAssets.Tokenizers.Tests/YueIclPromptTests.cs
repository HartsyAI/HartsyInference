using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.ModelAssets.Tokenizers.Tests;

/// <summary>Asset-free tests for YuE's reference-audio (ICL) prompt arithmetic — infer.py's
/// <c>audio_prompt_codec</c>: offset raw codebook-0 indices by the CodecManipulator("xcodec") global offset,
/// interleave the two tracks for a dual-track reference, then slice the requested second-window (at 50 tokens/s
/// single-track, 100 tokens/s dual-track).
///
/// <para>Second boundaries are chosen to be exactly representable in binary floating point (multiples of 0.25 and
/// 0.5) so the assertions gate the slicing rule rather than a double-rounding accident.</para></summary>
public sealed class YueIclPromptTests
{
    private const int Offset = 45_334;

    private static int[] Ramp(int n, int start = 0)
    {
        int[] a = new int[n];
        for (int i = 0; i < n; i++) a[i] = start + i;
        return a;
    }

    [Fact]
    public void SingleTrack_OffsetsEveryFrame_WhenTheWindowCoversTheClip()
    {
        int[] vocal = Ramp(100);   // 100 frames = 2.0 s at 50 fps
        int[] got = YueTokenizer.BuildAudioPromptCodec(vocal, [], 0.0, 30.0);

        Assert.Equal(100, got.Length);
        for (int i = 0; i < 100; i++) Assert.Equal(Offset + i, got[i]);
    }

    [Fact]
    public void SingleTrack_SlicesAtFiftyTokensPerSecond()
    {
        int[] vocal = Ramp(100);
        // [0.5 s, 1.0 s) -> frames [25, 50)
        int[] got = YueTokenizer.BuildAudioPromptCodec(vocal, [], 0.5, 1.0);

        Assert.Equal(25, got.Length);
        Assert.Equal(Offset + 25, got[0]);
        Assert.Equal(Offset + 49, got[^1]);
        for (int i = 0; i < got.Length; i++) Assert.Equal(Offset + 25 + i, got[i]);
    }

    [Fact]
    public void SingleTrack_ExactSequenceForASmallClip()
    {
        int[] vocal = [7, 1023, 0, 512, 9];
        int[] got = YueTokenizer.BuildAudioPromptCodec(vocal, [], 0.0, 30.0);
        Assert.Equal([45_341, 46_357, 45_334, 45_846, 45_343], got);
    }

    [Fact]
    public void DualTrack_InterleavesVocalThenInstrumental()
    {
        int[] vocal = [10, 11, 12, 13];
        int[] instrumental = [20, 21, 22, 23];
        int[] got = YueTokenizer.BuildAudioPromptCodec(vocal, instrumental, 0.0, 30.0);

        // v0, i0, v1, i1, ... (infer.py: rearrange([vocals, instrumental], 'b n -> (n b)'))
        Assert.Equal(
            [Offset + 10, Offset + 20, Offset + 11, Offset + 21,
             Offset + 12, Offset + 22, Offset + 13, Offset + 23],
            got);
    }

    [Fact]
    public void DualTrack_SlicesAtOneHundredTokensPerSecond()
    {
        int[] vocal = Ramp(30);
        int[] instrumental = Ramp(30, 100);
        // [0.0 s, 0.1 s) at 100 tokens/s -> 10 tokens = the first 5 frames of BOTH tracks.
        int[] got = YueTokenizer.BuildAudioPromptCodec(vocal, instrumental, 0.0, 0.1);

        Assert.Equal(10, got.Length);
        Assert.Equal([Offset, Offset + 100, Offset + 1, Offset + 101, Offset + 2,
                      Offset + 102, Offset + 3, Offset + 103, Offset + 4, Offset + 104], got);
    }

    [Fact]
    public void DualTrack_OddStartIndexFlipsTrackParity()
    {
        int[] vocal = Ramp(30);            // 0..29
        int[] instrumental = Ramp(30, 100); // 100..129
        // 0.25 s x 100 tokens/s = index 25 exactly — ODD, so the window opens on an INSTRUMENTAL token.
        // Upstream slices the interleaved array without realigning, so this parity flip is reference behaviour.
        int[] got = YueTokenizer.BuildAudioPromptCodec(vocal, instrumental, 0.25, 0.5);

        Assert.Equal(25, got.Length);
        Assert.Equal(Offset + 100 + 12, got[0]);   // index 25 -> instrumental[12]
        Assert.Equal(Offset + 13, got[1]);         // index 26 -> vocal[13]
        Assert.Equal(Offset + 100 + 13, got[2]);   // index 27 -> instrumental[13]
        for (int j = 25; j < 50; j++)
        {
            int expected = (j & 1) == 0 ? Offset + (j >> 1) : Offset + 100 + (j >> 1);
            Assert.Equal(expected, got[j - 25]);
        }
    }

    [Fact]
    public void Window_ClampsPastTheEndOfTheClip()
    {
        int[] vocal = Ramp(10);   // 0.2 s
        // Python slicing tolerates an overrunning stop rather than throwing.
        int[] got = YueTokenizer.BuildAudioPromptCodec(vocal, [], 0.0, 30.0);
        Assert.Equal(10, got.Length);
    }

    [Fact]
    public void Window_EntirelyPastTheClipIsEmpty()
    {
        int[] vocal = Ramp(10);
        Assert.Empty(YueTokenizer.BuildAudioPromptCodec(vocal, [], 5.0, 30.0));
    }

    [Fact]
    public void Window_WithEndBeforeStartIsEmpty()
    {
        int[] vocal = Ramp(100);
        Assert.Empty(YueTokenizer.BuildAudioPromptCodec(vocal, [], 1.0, 0.5));
    }

    [Fact]
    public void DualTrack_MismatchedLengthsThrow()
    {
        // Upstream's rearrange raises on ragged inputs; fail with a message that names the fix instead.
        ArgumentException ex = Assert.Throws<ArgumentException>(
            () => YueTokenizer.BuildAudioPromptCodec(Ramp(10), Ramp(9), 0.0, 30.0));
        Assert.Contains("equal lengths", ex.Message);
    }

    [Fact]
    public void OffsetMatchesTheCodecManipulatorConstants()
    {
        Assert.Equal(45_334, YueTokenizer.XcodecGlobalOffset);
        Assert.Equal(50, YueTokenizer.XcodecFps);
    }
}
