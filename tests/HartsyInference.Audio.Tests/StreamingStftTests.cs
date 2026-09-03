using HartsyInference.Audio.Models.Vocoders;
using HartsyInference.Audio.Streaming;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Covers the streaming analysis/synthesis pair that spectral denoising sits on. The failure these
/// guard against is silent: a wrong overlap-add normalizer or a mis-mirrored conjugate half still produces
/// plausible-sounding audio, just with the wrong amplitude envelope or a comb artifact, which downstream shows
/// up only as slightly worse model accuracy.</summary>
public sealed class StreamingStftTests
{
    private const int NFft = 512;
    private const int Hop = 128;

    private static float[] MakeSignal(int length)
    {
        // Two incommensurate tones plus an offset — a single bin-aligned sine could hide frame-alignment bugs.
        float[] x = new float[length];
        for (int i = 0; i < length; i++)
        {
            x[i] = 0.6f * MathF.Sin(2f * MathF.PI * 440f * i / 16000f)
                 + 0.3f * MathF.Sin(2f * MathF.PI * 1173f * i / 16000f + 0.7f);
        }
        return x;
    }

    /// <summary>Feeds audio through analysis then synthesis untouched and requires the original back. Each sample
    /// is divided by the same window-square sum it was accumulated from, so this is exact, not approximate —
    /// a loose tolerance here would hide exactly the normalizer bugs this exists to catch.</summary>
    [Fact]
    public void RoundTrip_ReconstructsInput()
    {
        float[] input = MakeSignal(8192);
        StreamingStft stft = new StreamingStft(NFft, Hop);
        StreamingIstft istft = new StreamingIstft(NFft, Hop);

        float[] re = new float[stft.BinCount];
        float[] im = new float[stft.BinCount];
        float[] hopOut = new float[Hop];
        List<float> output = [];

        // Odd push size on purpose: real callers deliver arbitrary chunk lengths, not whole hops.
        const int PushSize = 97;
        for (int offset = 0; offset < input.Length; offset += PushSize)
        {
            int take = Math.Min(PushSize, input.Length - offset);
            stft.AddSamples(input.AsSpan(offset, take));
            while (stft.TryExtractFrame(re, im))
            {
                istft.PushFrame(re, im, hopOut);
                output.AddRange(hopOut);
            }
        }

        Assert.True(output.Count > 4000, $"expected a substantial reconstruction, got {output.Count} samples");

        // Skip p=0, where the periodic Hann's leading zero is the only contribution and nothing is recoverable.
        for (int i = Hop; i < output.Count; i++)
        {
            Assert.True(MathF.Abs(output[i] - input[i]) < 1e-4f,
                $"sample {i}: got {output[i]}, expected {input[i]}");
        }
    }

    /// <summary>Pins the streaming path to the offline primitive it was derived from. <c>IStft.Apply</c> trims
    /// <c>nFft/2</c> of center padding, so its sample <c>i</c> is the streaming path's <c>i + nFft/2</c>.</summary>
    [Fact]
    public void Streaming_MatchesOfflineIStft()
    {
        float[] input = MakeSignal(8192);
        StreamingStft stft = new StreamingStft(NFft, Hop);
        StreamingIstft istft = new StreamingIstft(NFft, Hop);

        int bins = stft.BinCount;
        float[] re = new float[bins];
        float[] im = new float[bins];
        float[] hopOut = new float[Hop];
        List<float> streamed = [];
        List<float> allRe = [];
        List<float> allIm = [];

        stft.AddSamples(input);
        while (stft.TryExtractFrame(re, im))
        {
            allRe.AddRange(re);
            allIm.AddRange(im);
            istft.PushFrame(re, im, hopOut);
            streamed.AddRange(hopOut);
        }

        int frames = allRe.Count / bins;
        float[] offline = IStft.Apply([.. allRe], [.. allIm], frames, NFft, Hop);

        // Compare a solidly interior slice: the offline result runs past what streaming has emitted at the tail.
        int pad = NFft / 2;
        int compareLength = Math.Min(offline.Length, streamed.Count - pad) - NFft;
        Assert.True(compareLength > 2000, $"comparison window too small: {compareLength}");

        for (int i = 0; i < compareLength; i++)
        {
            Assert.True(MathF.Abs(offline[i] - streamed[i + pad]) < 1e-4f,
                $"sample {i}: offline {offline[i]}, streamed {streamed[i + pad]}");
        }
    }

    /// <summary>A tail carried across a discontinuity would splice audio that never adjoined — the same class of
    /// error the wake pipeline resets for on a sequence gap.</summary>
    [Fact]
    public void Reset_ClearsOverlapTail()
    {
        StreamingStft stft = new StreamingStft(NFft, Hop);
        StreamingIstft istft = new StreamingIstft(NFft, Hop);
        float[] re = new float[stft.BinCount];
        float[] im = new float[stft.BinCount];
        float[] hopOut = new float[Hop];

        stft.AddSamples(MakeSignal(4096));
        while (stft.TryExtractFrame(re, im)) istft.PushFrame(re, im, hopOut);

        stft.Reset();
        istft.Reset();
        Assert.Equal(0, stft.FramesEmitted);
        Assert.Equal(0, istft.FramesConsumed);

        // Pure silence in must give pure silence out; any residue is a surviving overlap-add tail.
        stft.AddSamples(new float[4096]);
        while (stft.TryExtractFrame(re, im))
        {
            istft.PushFrame(re, im, hopOut);
            foreach (float s in hopOut) Assert.True(MathF.Abs(s) < 1e-6f, $"residual {s} after reset");
        }
    }

    /// <summary>The always-on wake path runs this ~12.5 times a second per device forever, so a per-frame
    /// allocation is a permanent GC treadmill rather than a one-off cost.</summary>
    [Fact]
    public void SteadyState_DoesNotAllocate()
    {
        StreamingStft stft = new StreamingStft(NFft, Hop);
        StreamingIstft istft = new StreamingIstft(NFft, Hop);
        float[] re = new float[stft.BinCount];
        float[] im = new float[stft.BinCount];
        float[] hopOut = new float[Hop];
        float[] chunk = MakeSignal(Hop);

        for (int warmup = 0; warmup < 20; warmup++)
        {
            stft.AddSamples(chunk);
            while (stft.TryExtractFrame(re, im)) istft.PushFrame(re, im, hopOut);
        }

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 200; i++)
        {
            stft.AddSamples(chunk);
            while (stft.TryExtractFrame(re, im)) istft.PushFrame(re, im, hopOut);
        }
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        Assert.True(allocated == 0, $"allocated {allocated} bytes across 200 steady-state frames");
    }

    [Fact]
    public void UndersizedSpans_AreRejected()
    {
        StreamingStft stft = new StreamingStft(NFft, Hop);
        StreamingIstft istft = new StreamingIstft(NFft, Hop);
        Assert.Equal(NFft / 2 + 1, stft.BinCount);
        Assert.Equal(NFft - Hop, istft.LatencySamples);

        float[] tooSmall = new float[stft.BinCount - 1];
        float[] ok = new float[stft.BinCount];
        stft.AddSamples(MakeSignal(NFft));
        Assert.Throws<ArgumentException>(() => stft.TryExtractFrame(tooSmall, tooSmall));
        Assert.Throws<ArgumentException>(() => istft.PushFrame(ok, ok, new float[Hop - 1]));
    }
}
