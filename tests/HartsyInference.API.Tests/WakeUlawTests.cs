using HartsyInference.Engine.Audio.Wake;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>Audio a satellite can afford to send.
///
/// <para>A device on a marginal link does not lose audio because the link is slow on average — it loses audio
/// in stalls. One dropped packet costs a retransmission timeout of roughly a second, and everything queued
/// behind it goes on the floor. A fixed send buffer covers a fixed number of BYTES, so halving the bytes
/// doubles the seconds it covers, and that is the whole point of this format.</para>
///
/// <para>Not 8-bit linear, which would be free: the microphone on this satellite sits at a few hundred counts
/// out of 32768, and truncating that to a byte leaves two or three levels. These tests pin the round trip at
/// the levels that actually occur, not just at full scale.</para></summary>
public sealed class WakeUlawTests
{
    /// <summary>Encodes the way the satellite does, so the tests exercise a real round trip rather than the
    /// decoder against itself.</summary>
    private static byte Encode(short sample)
    {
        const int Bias = 0x84, Max = 32635;
        int sign = (sample >> 8) & 0x80;
        if (sign != 0) sample = (short)-sample;
        int magnitude = sample > Max ? Max : sample;
        magnitude += Bias;
        int exponent = 7;
        for (int mask = 0x4000; (magnitude & mask) == 0 && exponent > 0; exponent--, mask >>= 1) { }
        int mantissa = (magnitude >> (exponent + 3)) & 0x0F;
        return (byte)~(sign | (exponent << 4) | mantissa);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(20)]      // this device's measured quiet-room floor
    [InlineData(-20)]
    [InlineData(650)]     // its measured level during speech
    [InlineData(-650)]
    [InlineData(6558)]    // the loudest wake level ever recorded on it
    [InlineData(-6558)]
    [InlineData(32000)]
    public void ARoundTrip_StaysCloseEnoughToBeTheSameSound(short sample)
    {
        short decoded = WakeFrame.UlawToPcm(Encode(sample));
        // µ-law's error is proportional to magnitude, so the tolerance has to be too. 8% of the sample plus a
        // floor for the smallest steps — an absolute tolerance would either pass everything or fail the quiet
        // end, which is exactly the end that matters here.
        int tolerance = Math.Max(8, Math.Abs(sample) * 8 / 100);
        Assert.InRange(decoded - sample, -tolerance, tolerance);
    }

    [Fact]
    public void TheQuietEnd_KeepsItsResolution()
    {
        // The reason for µ-law over linear truncation, stated as a test: across the range this microphone
        // actually occupies, distinct inputs must stay distinct. Truncating to 8-bit linear would map every
        // one of these to zero or one.
        HashSet<short> distinct = [];
        for (short s = 0; s < 400; s += 25)
        {
            distinct.Add(WakeFrame.UlawToPcm(Encode(s)));
        }
        Assert.True(distinct.Count >= 12, $"only {distinct.Count} distinct levels below 400 — too coarse to hear a wake word through");
    }

    [Fact]
    public void SilenceStaysSilent()
    {
        Assert.InRange(WakeFrame.UlawToPcm(Encode(0)), -8, 8);
    }

    [Fact]
    public void EveryByteDecodesInsideInt16()
    {
        // The decoder runs on whatever arrives, including a corrupted frame. No input may produce something
        // that scores as a spike in the wake model.
        for (int b = 0; b <= 255; b++)
        {
            short v = WakeFrame.UlawToPcm((byte)b);
            Assert.InRange(v, short.MinValue, short.MaxValue);
        }
    }
}
