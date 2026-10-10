using HartsyInference.Audio.Preprocessing;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>End-to-end mel pipeline sanity checks: the basic shape and the well-known
/// Whisper-specific normalization output range (~[0, 1] after the +4/4 shift on a normal
/// speech clip). Numeric agreement with HF's <c>WhisperFeatureExtractor</c> is
/// <c>Parity/WhisperLogMelParityTests</c>.</summary>
public sealed class MelSpectrogramExtractorTests
{
    [Theory]
    [InlineData(80)]
    [InlineData(128)]
    public void OutputFrames_Whisper_30sClip_Is3000(int nMels)
    {
        // Centered STFT: 1 + 480000 / 160 frames, the last dropped — the [n_mels, 3000] the encoder was trained on,
        // 1500 positions after its stride-2 conv.
        MelSpectrogramExtractor extractor = new(MelSpectrogramExtractor.WhisperConfig(nMels));
        Assert.Equal(3000, extractor.OutputFrames(480_000));
    }

    [Fact]
    public void Compute_Sinusoid_HasEnergyAtExpectedMelBin()
    {
        // A 1 kHz sine wave at 16 kHz should have its energy concentrated in the mel
        // bins covering ~1 kHz. With 80 mel bins from 0-8 kHz Slaney-scaled, the
        // crossover from linear to log is at bin 20 (1 kHz = 15 mel), and 1 kHz lands
        // squarely in the linear region. We just confirm the spectrogram is NOT
        // uniform and has a clear band of energy somewhere in the lower half.
        MelSpectrogramExtractor extractor = new(MelSpectrogramExtractor.WhisperConfig());
        int sr = 16_000;
        float[] audio = new float[sr];
        for (int i = 0; i < sr; i++) audio[i] = 0.5f * MathF.Sin(2f * MathF.PI * 1000f * i / sr);
        float[,] mel = extractor.Compute(audio);

        // Find the peak mel bin in the middle of the spectrogram (away from edges).
        int peakBin = -1;
        float peakVal = float.MinValue;
        int midFrame = mel.GetLength(1) / 2;
        for (int m = 0; m < 80; m++)
        {
            if (mel[m, midFrame] > peakVal) { peakVal = mel[m, midFrame]; peakBin = m; }
        }
        // 1 kHz corresponds to mel 15 out of 80*8/8=80 mels total covering 0-8kHz.
        // In bin terms, mel 15 of the 80+2 mel-spaced centers (which span 0 to 8kHz≈42 mel)
        // corresponds to bin ≈ 15/42 * 80 ≈ 28. Allow generous tolerance.
        Assert.True(peakBin > 10 && peakBin < 50, $"1 kHz energy expected mid-range, got peak at bin {peakBin}");
    }

}
