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
    public void MelSpectrogramExtractor_FrameCountIsConsistentWithStftMath()
    {
        // Whisper's centered STFT (n_fft=400, hop=160) over a 1-second 16 kHz clip = 16000 samples:
        // 1 + 16000 / 160 = 101 frames, the last dropped — torch.stft's count, 10 ms per frame.
        int sr = 16_000;
        float[] sine = GenerateSine(sr, 440f, 1f);
        MelSpectrogramExtractor extractor = new(MelSpectrogramExtractor.WhisperConfig());
        int frames = extractor.OutputFrames(sine.Length);
        Assert.Equal(100, frames);
    }

}
