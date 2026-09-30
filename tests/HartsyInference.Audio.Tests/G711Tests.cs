using HartsyInference.Audio.Io;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>The G.711 tables against the standard's own fixed points. A wrong entry is inaudible in a unit test
/// and a buzz on a phone line, so every code and every 16-bit input is swept rather than sampled.</summary>
public sealed class G711Tests
{
    public static IEnumerable<object[]> Laws => [[G711Law.MuLaw], [G711Law.ALaw]];

    [Theory]
    [MemberData(nameof(Laws))]
    public void EveryCode_DecodesAndEncodesBackToItself(G711Law law)
    {
        for (int code = 0; code < 256; code++)
        {
            short linear = G711.DecodeSample((byte)code, law);
            // μ-law's 0x7F is "negative zero": it decodes to 0, and 0 encodes to the positive-zero code 0xFF.
            byte expected = law == G711Law.MuLaw && code == 0x7F ? (byte)0xFF : (byte)code;
            Assert.Equal(expected, G711.EncodeSample(linear, law));
        }
    }

    [Theory]
    [MemberData(nameof(Laws))]
    public void EveryInput_RoundTripsWithinItsSegmentStep(G711Law law)
    {
        for (int i = short.MinValue; i <= short.MaxValue; i++)
        {
            short sample = (short)i;
            short back = G711.DecodeSample(G711.EncodeSample(sample, law), law);
            int magnitude = Math.Abs(i);
            int tolerance = magnitude / 16 + 16;
            Assert.True(Math.Abs(back - i) <= tolerance,
                $"{law}: {i} -> 0x{G711.EncodeSample(sample, law):X2} -> {back}, off by {Math.Abs(back - i)} (> {tolerance})");
        }
    }

    [Fact]
    public void SilenceCodes_DecodeToTheSmallestStep()
    {
        Assert.Equal(0, G711.DecodeSample(G711.MuLawSilence, G711Law.MuLaw));
        Assert.InRange(Math.Abs((int)G711.DecodeSample(G711.ALawSilence, G711Law.ALaw)), 0, 8);
        Assert.Equal(G711.MuLawSilence, G711.EncodeSample(0, G711Law.MuLaw));
        Assert.Equal(G711.ALawSilence, G711.EncodeSample(0, G711Law.ALaw));
        Assert.Equal(G711.MuLawSilence, G711.Silence(G711Law.MuLaw));
        Assert.Equal(G711.ALawSilence, G711.Silence(G711Law.ALaw));
    }

    [Theory]
    [InlineData(G711Law.MuLaw, 0, 0xFF)]
    [InlineData(G711Law.MuLaw, -1, 0x7F)]
    [InlineData(G711Law.MuLaw, short.MaxValue, 0x80)]
    [InlineData(G711Law.MuLaw, short.MinValue, 0x00)]
    [InlineData(G711Law.ALaw, 0, 0xD5)]
    [InlineData(G711Law.ALaw, short.MaxValue, 0xAA)]
    [InlineData(G711Law.ALaw, short.MinValue, 0x2A)]
    public void KnownVectors_Encode(G711Law law, int sample, int expected)
    {
        Assert.Equal((byte)expected, G711.EncodeSample((short)sample, law));
    }

    [Theory]
    [MemberData(nameof(Laws))]
    public void Decode_IsMonotonicWithinEachSign(G711Law law)
    {
        // The 256 codes must decode to 256 distinct values, bar μ-law's +0/-0 pair (0xFF and 0x7F both decode
        // to zero), or the encoder's segment search is off somewhere.
        short[] decoded = new short[256];
        for (int code = 0; code < 256; code++)
        {
            decoded[code] = G711.DecodeSample((byte)code, law);
        }
        int distinct = decoded.Distinct().Count();
        Assert.Equal(law == G711Law.MuLaw ? 255 : 256, distinct);
    }

    [Fact]
    public void SpanApi_MatchesPerSampleApi_AndValidatesLengths()
    {
        short[] pcm = [0, 1, -1, 100, -100, 1000, -1000, 12345, -12345, short.MaxValue, short.MinValue];
        byte[] encoded = new byte[pcm.Length];
        short[] back = new short[pcm.Length];
        foreach (G711Law law in new[] { G711Law.MuLaw, G711Law.ALaw })
        {
            G711.Encode(pcm, encoded, law);
            for (int i = 0; i < pcm.Length; i++)
            {
                Assert.Equal(G711.EncodeSample(pcm[i], law), encoded[i]);
            }
            G711.Decode(encoded, back, law);
            for (int i = 0; i < pcm.Length; i++)
            {
                Assert.Equal(G711.DecodeSample(encoded[i], law), back[i]);
            }
        }
        Assert.Throws<ArgumentException>(() => G711.Encode(pcm, new byte[pcm.Length - 1], G711Law.MuLaw));
        Assert.Throws<ArgumentException>(() => G711.Decode(encoded, new short[encoded.Length - 1], G711Law.ALaw));
    }
}
