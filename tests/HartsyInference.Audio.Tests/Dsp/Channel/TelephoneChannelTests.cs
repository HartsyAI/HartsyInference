using HartsyInference.Audio.Dsp.Channel;
using HartsyInference.Audio.Io;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>The telephone channel's length, band shape, latency and G.711 stage.</summary>
public sealed class TelephoneChannelTests
{
    private const int Rate = 24000;
    private const int OutRate = 8000;
    private const int Edge = 200;

    public static IEnumerable<object[]> Codecs => [[TelephoneCodec.MuLaw], [TelephoneCodec.ALaw]];

    [Theory]
    [InlineData(24000)]
    [InlineData(24001)]
    public void Length_IsInputScaledToEightKilohertz(int samples)
    {
        float[] output = TelephoneChannel.Process(new float[samples], TelephoneCodec.MuLaw);

        Assert.True(Math.Abs(output.Length - samples * (double)OutRate / Rate) <= 1.0);
    }

    [Fact]
    public void Passband_OneKilohertzSineWithinOneDecibel()
    {
        float[] input = Sine(1000.0, 0.5, Rate);

        double gainDb = GainDb(TelephoneChannel.Process(input, TelephoneCodec.MuLaw), 0.5);

        Assert.True(Math.Abs(gainDb) < 1.0, $"1 kHz gain was {gainDb:F3} dB.");
    }

    [Fact]
    public void Stopband_HundredHertzSineAttenuatedByMoreThanThirtyDecibels()
    {
        float[] input = Sine(100.0, 0.5, Rate);

        double gainDb = GainDb(TelephoneChannel.Process(input, TelephoneCodec.MuLaw), 0.5);

        Assert.True(gainDb < -30.0, $"100 Hz gain was {gainDb:F3} dB.");
    }

    [Fact]
    public void AntiAlias_SixKilohertzEnergyFarBelowInputAfterBandLimit()
    {
        // The 6 kHz tone sits above the 3400 Hz edge (about -59 dB after the squared response); the 8 kHz
        // resampler then removes the rest. Threshold: output mean-square below 1e-3 of the input's (-30 dB).
        float[] input = Sine(6000.0, 1.0, Rate);

        float[] output = TelephoneChannel.Process(input, TelephoneCodec.MuLaw);

        double ratio = MeanSquare(output, Edge) / MeanSquare(input, 3 * Edge);
        Assert.True(ratio < 1e-3, $"Energy ratio was {ratio:E3}.");
    }

    [Fact]
    public void ZeroPhase_ClickStaysWithinOneOutputSampleOfItsScaledPosition()
    {
        const int click = 12000;
        float[] input = new float[Rate];
        input[click] = 1.0f;

        float[] output = TelephoneChannel.Process(input, TelephoneCodec.MuLaw);

        int peak = 0;
        for (int i = 1; i < output.Length; i++)
        {
            if (Math.Abs(output[i]) > Math.Abs(output[peak]))
            {
                peak = i;
            }
        }
        Assert.True(Math.Abs(peak - click / 3) <= 1, $"Peak at {peak}, expected {click / 3}.");
    }

    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(20)]
    public void ShortInput_ShorterThanPadding_IsFiniteWithScaledLength(int samples)
    {
        // Padding shrinks to length - 1 (no extension for one sample); the output keeps the scaled length.
        float[] input = Sine(1000.0, 0.5, Rate).AsSpan(0, samples).ToArray();

        float[] output = TelephoneChannel.Process(input, TelephoneCodec.MuLaw);

        Assert.Equal((samples + 2) / 3, output.Length);
        Assert.All(output, value => Assert.True(float.IsFinite(value)));
    }

    [Theory]
    [MemberData(nameof(Codecs))]
    public void Silence_EncodesToTheLawSilenceCode(TelephoneCodec codec)
    {
        byte expected = codec == TelephoneCodec.MuLaw ? G711.MuLawSilence : G711.ALawSilence;

        byte[] encoded = TelephoneChannel.Encode(new float[3000], codec);

        Assert.All(encoded, code => Assert.Equal(expected, code));
    }

    [Fact]
    public void HandCheckedCodes_MatchG711StandardTables()
    {
        // μ-law: silence 0xFF, full positive 0x80, full negative 0x00 (ITU-T G.711 Table 2a).
        Assert.Equal((byte)0xFF, G711.EncodeSample(0, G711Law.MuLaw));
        Assert.Equal((byte)0x80, G711.EncodeSample(32767, G711Law.MuLaw));
        Assert.Equal((byte)0x00, G711.EncodeSample(-32768, G711Law.MuLaw));

        // A-law: silence 0xD5, full positive 0xAA, full negative 0x2A.
        Assert.Equal((byte)0xD5, G711.EncodeSample(0, G711Law.ALaw));
        Assert.Equal((byte)0xAA, G711.EncodeSample(32767, G711Law.ALaw));
        Assert.Equal((byte)0x2A, G711.EncodeSample(-32768, G711Law.ALaw));
    }

    [Theory]
    [MemberData(nameof(Codecs))]
    public void RoundTrip_ProcessIsDecodedEncodeAndEncodeMatchesPerSampleFunction(TelephoneCodec codec)
    {
        G711Law law = codec == TelephoneCodec.MuLaw ? G711Law.MuLaw : G711Law.ALaw;
        float[] ramp = new float[4800];
        for (int i = 0; i < ramp.Length; i++)
        {
            ramp[i] = (i - 2400) / 2400f * 0.8f;
        }

        float[] processed = TelephoneChannel.Process(ramp, codec);
        byte[] encoded = TelephoneChannel.Encode(ramp, codec);

        Assert.Equal(processed.Length, encoded.Length);
        for (int i = 0; i < encoded.Length; i++)
        {
            Assert.Equal(G711.DecodeSample(encoded[i], law) / 32768f, processed[i]);
            short quantized = (short)Math.Round(processed[i] * 32768.0);
            // μ-law's negative-zero code 0x7F decodes to 0, which re-encodes as 0xFF; every other code is stable.
            byte expected = law == G711Law.MuLaw && encoded[i] == 0x7F ? (byte)0x7F : G711.EncodeSample(quantized, law);
            Assert.Equal(expected, encoded[i]);
        }
    }

    [Theory]
    [MemberData(nameof(Codecs))]
    public void EmptyInput_ReturnsEmpty(TelephoneCodec codec)
    {
        Assert.Empty(TelephoneChannel.Process(ReadOnlySpan<float>.Empty, codec));
        Assert.Empty(TelephoneChannel.Encode(ReadOnlySpan<float>.Empty, codec));
    }

    [Fact]
    public void UnsupportedSampleRate_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => TelephoneChannel.Process(new float[100], TelephoneCodec.MuLaw, 16000));
    }

    private static float[] Sine(double frequency, double amplitude, int rate)
    {
        float[] samples = new float[rate];
        for (int i = 0; i < samples.Length; i++)
        {
            samples[i] = (float)(amplitude * Math.Sin(2.0 * Math.PI * frequency * i / rate));
        }
        return samples;
    }

    /// <summary>Output RMS over the middle of the signal, against a sine of the given amplitude.</summary>
    private static double GainDb(float[] output, double amplitude)
    {
        double rms = Math.Sqrt(MeanSquare(output, Edge));
        return 20.0 * Math.Log10(rms / (amplitude / Math.Sqrt(2.0)));
    }

    private static double MeanSquare(float[] signal, int skip)
    {
        double sum = 0.0;
        int count = 0;
        for (int i = skip; i < signal.Length - skip; i++)
        {
            sum += (double)signal[i] * signal[i];
            count++;
        }
        return sum / count;
    }
}
