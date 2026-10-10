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

    [Theory]
    [InlineData(G711Law.MuLaw, 0, 0xFF)]
    [InlineData(G711Law.ALaw, short.MinValue, 0x2A)]
    public void KnownVectors_Encode(G711Law law, int sample, int expected)
    {
        Assert.Equal((byte)expected, G711.EncodeSample((short)sample, law));
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
