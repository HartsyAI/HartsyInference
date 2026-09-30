using HartsyInference.Audio.Io;
using SIPSorcery.Media;
using Xunit;

namespace HartsyInference.PhoneGateway.Tests;

/// <summary>The gateway's table codec must agree code-for-code with the encoder sipsorcery would otherwise apply, so
/// a frame we encode ourselves is indistinguishable on the wire from one sipsorcery encodes. Every 16-bit input is
/// checked. The single documented exception is <c>short.MinValue</c>: the NAudio-derived encoders negate it in
/// 16-bit arithmetic, which wraps back to -32768 and produces a near-zero code, while <see cref="G711"/> clips it to
/// the most negative code like every other over-range sample. That input is asserted separately, not skipped
/// silently.</summary>
public sealed class G711EquivalenceTests
{
    [Fact]
    public void MuLaw_MatchesSipsorceryForEveryInputButMinValue()
    {
        int mismatches = 0;
        for (int i = short.MinValue + 1; i <= short.MaxValue; i++)
        {
            short sample = (short)i;
            if (G711.EncodeSample(sample, G711Law.MuLaw) != MuLawEncoder.LinearToMuLawSample(sample))
            {
                mismatches++;
            }
        }
        Assert.Equal(0, mismatches);
    }

    [Fact]
    public void ALaw_MatchesSipsorceryForEveryInputButMinValue()
    {
        int mismatches = 0;
        for (int i = short.MinValue + 1; i <= short.MaxValue; i++)
        {
            short sample = (short)i;
            if (G711.EncodeSample(sample, G711Law.ALaw) != ALawEncoder.LinearToALawSample(sample))
            {
                mismatches++;
            }
        }
        Assert.Equal(0, mismatches);
    }

    [Fact]
    public void MinValue_ClipsToTheMostNegativeCodeWhereSipsorceryWraps()
    {
        // Ours: -32768 clips to the same code as -32767 (full-scale negative).
        Assert.Equal(G711.EncodeSample(-32767, G711Law.MuLaw), G711.EncodeSample(short.MinValue, G711Law.MuLaw));
        Assert.Equal(G711.EncodeSample(-32767, G711Law.ALaw), G711.EncodeSample(short.MinValue, G711Law.ALaw));
        // Theirs: the negation wraps, so the code is not the full-scale one. Documented, not a bug we mirror.
        Assert.NotEqual(G711.EncodeSample(short.MinValue, G711Law.MuLaw), MuLawEncoder.LinearToMuLawSample(short.MinValue));
    }

    [Fact]
    public void FrameEncode_MatchesPerSampleEncode()
    {
        short[] pcm = new short[160];
        for (int i = 0; i < pcm.Length; i++)
        {
            pcm[i] = (short)(Math.Sin(i * 0.1) * 12000);
        }
        byte[] mu = new byte[160];
        byte[] a = new byte[160];
        G711.Encode(pcm, mu, G711Law.MuLaw);
        G711.Encode(pcm, a, G711Law.ALaw);
        for (int i = 0; i < pcm.Length; i++)
        {
            Assert.Equal(MuLawEncoder.LinearToMuLawSample(pcm[i]), mu[i]);
            Assert.Equal(ALawEncoder.LinearToALawSample(pcm[i]), a[i]);
        }
    }
}
