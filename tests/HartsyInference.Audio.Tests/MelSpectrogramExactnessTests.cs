using HartsyInference.Audio.Preprocessing;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Bit-exactness of the mel front-end against the textbook dense form. The extractor sums each filter only
/// over its nonzero bin range and, for a zero-padded input, fills a frame that lies wholly in the padding with that
/// frame's constant instead of transforming it; both are claimed to change no bit (a skipped product is exactly +0 and
/// the running sum never becomes -0), for every preset, so any drift here would silently move every mel-input model.</summary>
public sealed class MelSpectrogramExactnessTests
{
    public static IEnumerable<object[]> Presets =>
    [
        ["whisper80", MelSpectrogramExtractor.WhisperConfig()],
        ["whisper128", MelSpectrogramExtractor.WhisperConfig(128)],
        ["kokoro24k", MelSpectrogramExtractor.Kokoro24kConfig()],
        ["cosyvoice2flow", MelSpectrogramExtractor.CosyVoice2FlowConfig()],
        ["f5vocos", MelSpectrogramExtractor.F5VocosConfig()],
        ["zonos16k", MelSpectrogramExtractor.Zonos16kConfig()],
        ["hifigan22k", MelSpectrogramExtractor.HifiGan22kConfig()],
        ["rawAmplitude", MelSpectrogramExtractor.WhisperConfig() with
        {
            Norm = MelSpectrogramExtractor.Normalization.None, LogBase = MelSpectrogramExtractor.LogBase.None, PowerSpectrum = false,
        }],
    ];

    [Theory]
    [MemberData(nameof(Presets))]
    public void Compute_MatchesDenseReference_BitForBit(string name, MelSpectrogramExtractor.Config cfg)
    {
        float[] audio = Speechlike(cfg.SampleRate * 3 / 2 + 37, seed: (uint)name.Length);
        float[,] actual = new MelSpectrogramExtractor(cfg).Compute(audio);
        float[,] expected = DenseReference(cfg, audio);
        AssertBitEqual(expected, actual, name);
    }

    [Theory]
    [MemberData(nameof(Presets))]
    public void ComputeFrame_MatchesDenseReference_BitForBit(string name, MelSpectrogramExtractor.Config cfg)
    {
        float[] window = Speechlike(cfg.WinLength, seed: 7u + (uint)name.Length);
        MelSpectrogramExtractor extractor = new(cfg);
        float[] actual = new float[cfg.NMels];
        extractor.ComputeFrame(window, actual);
        // The streaming single-frame path always left-aligns the window, whatever CenterWindowInFft says.
        float[] expected = DenseFrame(cfg with { CenterWindowInFft = false }, window, 0, window.Length);
        for (int m = 0; m < cfg.NMels; m++)
        {
            Assert.True(BitConverter.SingleToInt32Bits(expected[m]) == BitConverter.SingleToInt32Bits(actual[m]),
                $"{name}: mel {m} expected {expected[m]:R} got {actual[m]:R}");
        }
    }

    /// <summary>Whisper's 30 s window at every boundary that matters: empty, a partial first frame, exactly one window,
    /// a mid-frame end, the gate's 2 / 5 / 10 s utterances and a full window with no padding at all.</summary>
    [Theory]
    [InlineData(80, 0)]
    [InlineData(80, 1)]
    [InlineData(80, 399)]
    [InlineData(80, 400)]
    [InlineData(80, 16_003)]
    [InlineData(80, 32_000)]
    [InlineData(80, 80_000)]
    [InlineData(80, 160_000)]
    [InlineData(80, 480_000)]
    [InlineData(128, 32_000)]
    [InlineData(128, 479_999)]
    public void ComputeZeroPadded_MatchesComputeOnThePaddedBuffer(int nMels, int length)
    {
        const int padded = 480_000;
        MelSpectrogramExtractor.Config cfg = MelSpectrogramExtractor.WhisperConfig(nMels);
        float[] audio = Speechlike(length, seed: 11u + (uint)length);
        float[] buffer = new float[padded];
        audio.CopyTo(buffer, 0);
        float[,] expected = new MelSpectrogramExtractor(cfg).Compute(buffer);

        MelSpectrogramExtractor extractor = new(cfg);
        int frames = extractor.OutputFrames(padded);
        float[] actual = new float[nMels * frames];
        extractor.ComputeZeroPadded(audio, padded, actual);
        for (int m = 0; m < nMels; m++)
        {
            for (int t = 0; t < frames; t++)
            {
                float e = expected[m, t], a = actual[m * frames + t];
                Assert.True(BitConverter.SingleToInt32Bits(e) == BitConverter.SingleToInt32Bits(a),
                    $"length {length}: mel [{m}, {t}] expected {e:R} got {a:R}");
            }
        }
    }

    [Fact]
    public void ComputeZeroPadded_RefusesCenteredPresetsAndOverlongAudio()
    {
        MelSpectrogramExtractor centered = new(MelSpectrogramExtractor.F5VocosConfig());
        Assert.Throws<InvalidOperationException>(() => centered.ComputeZeroPadded(new float[10], 24_000, new float[100 * 100]));
        MelSpectrogramExtractor whisper = new(MelSpectrogramExtractor.WhisperConfig());
        Assert.Throws<ArgumentException>(() => whisper.ComputeZeroPadded(new float[20], 10, new float[80 * 10]));
    }

    /// <summary>The pre-refactor algorithm, written out densely: every filterbank bin, inline log compression.</summary>
    private static float[,] DenseReference(MelSpectrogramExtractor.Config cfg, float[] audio)
    {
        MelSpectrogramExtractor sizing = new(cfg);
        int frames = sizing.OutputFrames(audio.Length);
        float[] src = cfg.Center ? SignalPadding.Reflect(audio, cfg.NFft / 2) : audio;
        float[,] output = new float[cfg.NMels, frames];
        float globalMax = float.MinValue;
        for (int t = 0; t < frames; t++)
        {
            float[] column = DenseFrame(cfg, src, t * cfg.HopLength, src.Length);
            for (int m = 0; m < cfg.NMels; m++)
            {
                output[m, t] = column[m];
                if (column[m] > globalMax) globalMax = column[m];
            }
        }
        if (cfg.Norm == MelSpectrogramExtractor.Normalization.WhisperDynamicRange)
        {
            float clampMin = globalMax - cfg.DynamicRangeDb;
            float invScale = 1f / cfg.NormScale;
            for (int m = 0; m < cfg.NMels; m++)
                for (int t = 0; t < frames; t++)
                    output[m, t] = (MathF.Max(output[m, t], clampMin) + cfg.NormOffset) * invScale;
        }
        return output;
    }

    private static float[] DenseFrame(MelSpectrogramExtractor.Config cfg, float[] src, int start, int srcLength)
    {
        int fftSize = Fft.NextPow2(cfg.NFft);
        int numBins = fftSize / 2 + 1;
        float[] window = HannWindow.Get(cfg.WinLength);
        float[,] fb = MelFilterbank.Get(cfg.SampleRate, fftSize, cfg.NMels, cfg.Fmin, cfg.Fmax, cfg.Scale, cfg.SlaneyNorm);
        int woff = cfg.CenterWindowInFft ? (fftSize - cfg.WinLength) / 2 : 0;
        float[] frame = new float[fftSize];
        for (int i = 0; i < cfg.WinLength; i++)
        {
            float sample = (start + woff + i) < srcLength ? src[start + woff + i] : 0f;
            frame[woff + i] = sample * window[i];
        }
        float[] re = new float[numBins], im = new float[numBins];
        Fft.RealTransform(frame, re, im, fftSize);
        float[] power = new float[numBins];
        for (int k = 0; k < numBins; k++)
            power[k] = cfg.PowerSpectrum ? re[k] * re[k] + im[k] * im[k] : MathF.Sqrt(re[k] * re[k] + im[k] * im[k]);
        float[] column = new float[cfg.NMels];
        float floor = cfg.LogFloor ?? 1e-10f;
        for (int m = 0; m < cfg.NMels; m++)
        {
            float acc = 0f;
            for (int k = 0; k < numBins; k++) acc += fb[m, k] * power[k];
            if (cfg.LogBase == MelSpectrogramExtractor.LogBase.None)
            {
                column[m] = acc;
                continue;
            }
            float v = cfg.AdditiveLogFloor ? acc + floor : MathF.Max(acc, floor);
            column[m] = cfg.LogBase == MelSpectrogramExtractor.LogBase.Log10 ? MathF.Log10(v) : MathF.Log(v);
        }
        return column;
    }

    /// <summary>Deterministic voiced-ish signal: two tones under a slow envelope plus low-level noise, in [-1, 1].</summary>
    private static float[] Speechlike(int length, uint seed)
    {
        float[] audio = new float[length];
        uint state = 0x9E3779B9u ^ seed;
        for (int i = 0; i < length; i++)
        {
            state = state * 1664525u + 1013904223u;
            float noise = ((state >> 8) & 0xFFFF) / 32768f - 1f;
            float t = i / 16_000f;
            float envelope = 0.5f + 0.5f * MathF.Sin(2 * MathF.PI * 3f * t);
            audio[i] = envelope * (0.4f * MathF.Sin(2 * MathF.PI * 190f * t) + 0.2f * MathF.Sin(2 * MathF.PI * 1330f * t))
                + 0.03f * noise;
        }
        return audio;
    }

    private static void AssertBitEqual(float[,] expected, float[,] actual, string name)
    {
        Assert.Equal(expected.GetLength(0), actual.GetLength(0));
        Assert.Equal(expected.GetLength(1), actual.GetLength(1));
        for (int m = 0; m < expected.GetLength(0); m++)
        {
            for (int t = 0; t < expected.GetLength(1); t++)
            {
                Assert.True(BitConverter.SingleToInt32Bits(expected[m, t]) == BitConverter.SingleToInt32Bits(actual[m, t]),
                    $"{name}: mel [{m}, {t}] expected {expected[m, t]:R} got {actual[m, t]:R}");
            }
        }
    }
}
