using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Numerics;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Bit-exactness of the mel front-end against the textbook dense form. The extractor sums each filter only
/// over its nonzero bin range, reads a centered preset's reflect padding through index arithmetic instead of building
/// it, fills a frame that reads only zero padding with that frame's constant instead of transforming it, and fans
/// frame blocks out across cores; all of it is claimed to change no bit (a skipped product is exactly +0 and the
/// running sum never becomes -0), for every preset, so any drift here would silently move every mel-input model.</summary>
public sealed class MelSpectrogramExactnessTests
{
    private const int WhisperWindow = 480_000;

    public static IEnumerable<object[]> Presets =>
    [
        ["whisper80", MelSpectrogramExtractor.WhisperConfig()],
        ["whisper128", MelSpectrogramExtractor.WhisperConfig(128)],
        ["whisperLegacyPow2_128", MelSpectrogramExtractor.WhisperLegacyPow2Config(128)],
        ["kokoro24k", MelSpectrogramExtractor.Kokoro24kConfig()],
        ["cosyvoice2flow", MelSpectrogramExtractor.CosyVoice2FlowConfig()],
        ["f5vocos", MelSpectrogramExtractor.F5VocosConfig()],
        ["zonos16k", MelSpectrogramExtractor.Zonos16kConfig()],
        ["hifigan22k", MelSpectrogramExtractor.HifiGan22kConfig()],
        ["rawAmplitude", MelSpectrogramExtractor.WhisperLegacyPow2Config() with
        {
            Norm = MelSpectrogramExtractor.Normalization.None, LogBase = MelSpectrogramExtractor.LogBase.None, PowerSpectrum = false,
        }],
    ];

    /// <summary>Zero-padded windows at every boundary that matters. Whisper's 30 s window: empty, a partial first
    /// frame, the last length whose third frame (the first one clear of the left reflection) still reads only padding,
    /// one window, a mid-frame end, the gate's 2 / 5 / 10 s utterances, the last length whose final frame reads only
    /// padding and one sample past it, and a full window whose right reflection reads real audio. The legacy layout's
    /// non-centered path, and a centered preset that keeps its last frame, where the mirrored read past the end
    /// reaches one sample further back than the frame's own start. A log floor or Whisper's clamp would hide a
    /// boundary frame's only real sample, which the window weights near zero, so raw-amplitude twins pin the
    /// padding test itself: at those lengths that frame is nonzero.</summary>
    public static IEnumerable<object[]> ZeroPaddedCases =>
    [
        ["whisperRaw", WhisperRaw(), WhisperWindow, 122],
        ["whisperRaw", WhisperRaw(), WhisperWindow, 479_642],
        ["f5vocosRaw", MelSpectrogramExtractor.F5VocosConfig() with { LogBase = MelSpectrogramExtractor.LogBase.None }, 25_600, 25_088],
        ["whisper80", MelSpectrogramExtractor.WhisperConfig(), WhisperWindow, 0],
        ["whisper80", MelSpectrogramExtractor.WhisperConfig(), WhisperWindow, 1],
        ["whisper80", MelSpectrogramExtractor.WhisperConfig(), WhisperWindow, 120],
        ["whisper80", MelSpectrogramExtractor.WhisperConfig(), WhisperWindow, 121],
        ["whisper80", MelSpectrogramExtractor.WhisperConfig(), WhisperWindow, 400],
        ["whisper80", MelSpectrogramExtractor.WhisperConfig(), WhisperWindow, 16_003],
        ["whisper80", MelSpectrogramExtractor.WhisperConfig(), WhisperWindow, 32_000],
        ["whisper80", MelSpectrogramExtractor.WhisperConfig(), WhisperWindow, 80_000],
        ["whisper80", MelSpectrogramExtractor.WhisperConfig(), WhisperWindow, 160_000],
        ["whisper80", MelSpectrogramExtractor.WhisperConfig(), WhisperWindow, 479_640],
        ["whisper80", MelSpectrogramExtractor.WhisperConfig(), WhisperWindow, 479_641],
        ["whisper80", MelSpectrogramExtractor.WhisperConfig(), WhisperWindow, WhisperWindow],
        ["whisper128", MelSpectrogramExtractor.WhisperConfig(128), WhisperWindow, 32_000],
        ["whisper128", MelSpectrogramExtractor.WhisperConfig(128), WhisperWindow, 479_999],
        ["whisperLegacyPow2_80", MelSpectrogramExtractor.WhisperLegacyPow2Config(), WhisperWindow, 399],
        ["whisperLegacyPow2_80", MelSpectrogramExtractor.WhisperLegacyPow2Config(), WhisperWindow, 32_000],
        ["f5vocos", MelSpectrogramExtractor.F5VocosConfig(), 25_600, 1_000],
        ["f5vocos", MelSpectrogramExtractor.F5VocosConfig(), 25_600, 25_087],
        ["f5vocos", MelSpectrogramExtractor.F5VocosConfig(), 25_600, 25_088],
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

    [Theory]
    [MemberData(nameof(ZeroPaddedCases))]
    public void ComputeZeroPadded_MatchesTheDenseFormOfThePaddedBuffer(string name, MelSpectrogramExtractor.Config cfg,
        int padded, int length)
    {
        float[] audio = Speechlike(length, seed: 11u + (uint)length);
        float[] buffer = new float[padded];
        audio.CopyTo(buffer, 0);
        float[,] expected = DenseReference(cfg, buffer);

        MelSpectrogramExtractor extractor = new(cfg);
        int frames = extractor.OutputFrames(padded);
        Assert.Equal(expected.GetLength(1), frames);
        float[] actual = new float[cfg.NMels * frames];
        extractor.ComputeZeroPadded(audio, padded, actual);
        for (int m = 0; m < cfg.NMels; m++)
        {
            for (int t = 0; t < frames; t++)
            {
                float e = expected[m, t], a = actual[m * frames + t];
                Assert.True(BitConverter.SingleToInt32Bits(e) == BitConverter.SingleToInt32Bits(a),
                    $"{name} length {length}: mel [{m}, {t}] expected {e:R} got {a:R}");
            }
        }
    }

    [Theory]
    [InlineData(80, 160_000)]
    [InlineData(128, WhisperWindow)]
    public void ComputeZeroPadded_FannedOut_EqualsInline(int nMels, int length)
    {
        MelSpectrogramExtractor extractor = new(MelSpectrogramExtractor.WhisperConfig(nMels));
        float[] audio = Speechlike(length, seed: 3u);
        int frames = extractor.OutputFrames(WhisperWindow);
        float[] fannedOut = new float[nMels * frames];
        float[] inline = new float[nMels * frames];
        extractor.ComputeZeroPadded(audio, WhisperWindow, fannedOut);
        using (CpuParallel.EnterInline())
        {
            extractor.ComputeZeroPadded(audio, WhisperWindow, inline);
        }
        for (int i = 0; i < inline.Length; i++)
        {
            Assert.True(BitConverter.SingleToInt32Bits(inline[i]) == BitConverter.SingleToInt32Bits(fannedOut[i]),
                $"element {i}: inline {inline[i]:R} fanned out {fannedOut[i]:R}");
        }
    }

    /// <summary>Whisper's front end runs once per utterance on the voice path; a warm extractor allocates nothing.</summary>
    [Fact]
    public void ComputeZeroPadded_Inline_AllocatesNothingOnceWarm()
    {
        MelSpectrogramExtractor extractor = new(MelSpectrogramExtractor.WhisperConfig());
        float[] audio = Speechlike(80_000, seed: 5u);
        float[] output = new float[80 * extractor.OutputFrames(WhisperWindow)];
        using CpuParallel.InlineScope inline = CpuParallel.EnterInline();
        extractor.ComputeZeroPadded(audio, WhisperWindow, output);
        long before = GC.GetAllocatedBytesForCurrentThread();
        extractor.ComputeZeroPadded(audio, WhisperWindow, output);
        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
    }

    [Fact]
    public void ComputeZeroPadded_RefusesOverlongAudio()
    {
        MelSpectrogramExtractor whisper = new(MelSpectrogramExtractor.WhisperConfig());
        Assert.Throws<ArgumentException>(() => whisper.ComputeZeroPadded(new float[20], 10, new float[80 * 10]));
    }

    [Fact]
    public void ExactFftSize_RefusesASizeTheMixedRadixPlanCannotRun()
    {
        MelSpectrogramExtractor.Config cfg = MelSpectrogramExtractor.WhisperConfig() with { NFft = 406, WinLength = 406 };
        Assert.Throws<ArgumentException>(() => new MelSpectrogramExtractor(cfg));
    }

    private static MelSpectrogramExtractor.Config WhisperRaw() => MelSpectrogramExtractor.WhisperConfig() with
    {
        Norm = MelSpectrogramExtractor.Normalization.None, LogBase = MelSpectrogramExtractor.LogBase.None,
    };

    /// <summary>The algorithm written out densely: materialized reflect padding, every filterbank bin, inline log
    /// compression, one frame after another.</summary>
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
        int fftSize = cfg.ExactFftSize ? cfg.NFft : Fft.NextPow2(cfg.NFft);
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
        if ((fftSize & (fftSize - 1)) != 0)
        {
            new FftPlan(fftSize).ForwardReal(frame, re, im);
        }
        else
        {
            Fft.RealTransform(frame, re, im, fftSize);
        }
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
