using HartsyInference.Audio.Io;
using HartsyInference.Audio.Preprocessing;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>End-to-end round-trip tests on synthetic audio: generate a known sine
/// wave, run it through the I/O + preprocessing pipeline (WAV write/read, resample,
/// mel-spectrogram), and verify properties of the output that any production audio
/// pipeline depends on.
///
/// <para>These don't require any model weights — they're guaranteed to run on every
/// developer's machine. If they ever break, something foundational has shifted in
/// the audio I/O / preprocessing stack and the model-level integration tests
/// downstream would all break in mysterious ways.</para></summary>
public sealed class SyntheticAudioRoundTripTests
{
    private static float[] GenerateSine(int sampleRate, float frequencyHz, float durationSec, float amplitude = 0.5f)
    {
        int n = (int)(sampleRate * durationSec);
        float[] samples = new float[n];
        for (int i = 0; i < n; i++)
            samples[i] = amplitude * MathF.Sin(2f * MathF.PI * frequencyHz * i / sampleRate);
        return samples;
    }

    [Fact]
    public void WavFile_RoundTrip_PreservesWithinPcm16Quantization()
    {
        int sr = 16_000;
        float[] sine = GenerateSine(sr, 880f, 0.05f);

        string tmp = Path.GetTempFileName();
        try
        {
            WavFile.WriteMono16(tmp, sine, sr);
            WavFile.DecodedAudio decoded = WavFile.Read(tmp);
            Assert.Equal(sr, decoded.SampleRate);
            Assert.Single(decoded.Channels);
            float[] read = decoded.Channels[0];
            Assert.Equal(sine.Length, read.Length);
            // 16-bit PCM has quantization step 1/32768 ≈ 3e-5; allow ~1 LSB slack.
            for (int i = 0; i < sine.Length; i++)
                Assert.True(MathF.Abs(sine[i] - read[i]) < 4e-5f, $"sample {i}: orig={sine[i]} read={read[i]}");
        }
        finally
        {
            if (File.Exists(tmp)) File.Delete(tmp);
        }
    }

    [Fact]
    public void WavFile_RoundTrip_LongSilence_StaysZero()
    {
        // 1 s of silence round-trips to silence.
        int sr = 16_000;
        float[] silence = new float[sr];

        string tmp = Path.GetTempFileName();
        try
        {
            WavFile.WriteMono16(tmp, silence, sr);
            WavFile.DecodedAudio decoded = WavFile.Read(tmp);
            float[] read = decoded.Channels[0];
            Assert.Equal(silence.Length, read.Length);
            for (int i = 0; i < silence.Length; i++)
                Assert.Equal(0f, read[i], precision: 5);
        }
        finally
        {
            if (File.Exists(tmp)) File.Delete(tmp);
        }
    }

    [Fact]
    public void Resampler_24kTo16k_PreservesDurationApproximately()
    {
        int srIn = 24_000;
        int srOut = 16_000;
        float[] sine = GenerateSine(srIn, 440f, 0.5f);
        Resampler resampler = Resampler.Create(srIn, srOut);
        float[] resampled = resampler.Resample(sine);

        // Output length should be approximately input.Length * srOut / srIn.
        int expected = (int)(sine.Length * (long)srOut / srIn);
        Assert.InRange(resampled.Length, expected - 16, expected + 16);     // allow polyphase tap padding
    }

    [Fact]
    public void Resampler_IdentityRate_PreservesEnergyAndShape()
    {
        // Same-rate resample produces the same length and preserves bulk RMS energy
        // (the polyphase low-pass filter introduces ~half-tap-count phase delay and a
        // small DC/Nyquist attenuation, so a sample-by-sample equality check isn't the
        // right invariant). What we DO require: the output is finite, the length
        // matches, and bulk RMS stays within ±20% of input (the filter passband attenuation
        // tolerance at 64 taps).
        int sr = 16_000;
        float[] sine = GenerateSine(sr, 440f, 0.1f);
        Resampler resampler = Resampler.Create(sr, sr);
        float[] resampled = resampler.Resample(sine);
        Assert.Equal(sine.Length, resampled.Length);

        double rmsIn = 0d, rmsOut = 0d;
        for (int i = 0; i < sine.Length; i++)
        {
            rmsIn += (double)sine[i] * sine[i];
            rmsOut += (double)resampled[i] * resampled[i];
        }
        rmsIn = Math.Sqrt(rmsIn / sine.Length);
        rmsOut = Math.Sqrt(rmsOut / resampled.Length);
        double ratio = rmsOut / rmsIn;
        Assert.InRange(ratio, 0.8, 1.2);
    }

    [Fact]
    public void MelSpectrogramExtractor_ProducesFiniteOutputForSineWave()
    {
        // 440 Hz sine at 16 kHz → log-mel must be finite (no NaN, no Inf) and
        // produce at least one non-near-zero value (signal has energy).
        int sr = 16_000;
        float[] sine = GenerateSine(sr, 440f, 1f);
        MelSpectrogramExtractor extractor = new(MelSpectrogramExtractor.WhisperConfig());
        float[,] mel = extractor.Compute(sine);

        Assert.Equal(80, mel.GetLength(0));
        Assert.True(mel.GetLength(1) > 0);

        bool sawNonZero = false;
        for (int m = 0; m < mel.GetLength(0); m++)
            for (int t = 0; t < mel.GetLength(1); t++)
            {
                float v = mel[m, t];
                Assert.False(float.IsNaN(v), $"mel[{m},{t}] is NaN");
                Assert.False(float.IsInfinity(v), $"mel[{m},{t}] is Inf");
                if (MathF.Abs(v) > 0.01f) sawNonZero = true;
            }
        Assert.True(sawNonZero, "mel output is entirely near-zero — preprocessing produced no signal");
    }

    [Fact]
    public void MelSpectrogramExtractor_FrameCountIsConsistentWithStftMath()
    {
        // For Whisper preset (n_fft=400, hop=160) and a 1-second 16 kHz clip = 16000 samples:
        // expected frame count = ((16000 - 400) / 160) + 1 = 98, minus 1 if drop-last-frame.
        int sr = 16_000;
        float[] sine = GenerateSine(sr, 440f, 1f);
        MelSpectrogramExtractor extractor = new(MelSpectrogramExtractor.WhisperConfig());
        int frames = extractor.OutputFrames(sine.Length);
        // Whisper preset has drop-last-frame so we expect 97.
        Assert.Equal(97, frames);
    }

    [Fact]
    public void EndToEnd_SineThroughResampleAndMel_StaysFinite()
    {
        // The "real" use case for these primitives: take 24 kHz audio (typical TTS
        // output), resample to 16 kHz (Whisper's preferred rate), run mel preprocessing.
        // Every numerical value must remain finite — if anything overflows or NaN's, the
        // STT pipeline downstream gets junk.
        int srTts = 24_000;
        int srStt = 16_000;
        float[] sine = GenerateSine(srTts, 440f, 1f);
        Resampler resampler = Resampler.Create(srTts, srStt);
        float[] resampled = resampler.Resample(sine);

        MelSpectrogramExtractor extractor = new(MelSpectrogramExtractor.WhisperConfig());
        float[,] mel = extractor.Compute(resampled);

        Assert.True(mel.GetLength(1) > 0, "mel produced zero frames");
        bool sawNonZero = false;
        for (int m = 0; m < mel.GetLength(0); m++)
            for (int t = 0; t < mel.GetLength(1); t++)
            {
                float v = mel[m, t];
                Assert.False(float.IsNaN(v));
                Assert.False(float.IsInfinity(v));
                if (MathF.Abs(v) > 0.01f) sawNonZero = true;
            }
        Assert.True(sawNonZero, "mel output is entirely near-zero — preprocessing produced no signal");
    }
}
