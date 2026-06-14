using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.Vocoders;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelHandler.SafeTensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Vocos vocoder tests. The unit tests exercise config + iSTFT shape math.
/// The integration test runs the real Vocos model on a saved mel and verifies the
/// reconstructed waveform has the expected length and a non-pathological RMS.
///
/// <para>The mel is pre-saved as <c>jfk_mel_vocos.bin</c> (raw float32, 100×N) in the
/// model cache. Generate it with the Python script in <c>tools/</c> if missing.</para></summary>
public sealed class VocosTests
{
    [Fact]
    public void Mel24kPreset_MatchesUpstreamConfig()
    {
        VocosConfig c = VocosConfig.Mel24k;
        Assert.Equal(100, c.InputChannels);
        Assert.Equal(512, c.HiddenDim);
        Assert.Equal(1536, c.IntermediateDim);
        Assert.Equal(8, c.NumLayers);
        Assert.Equal(7, c.DwConvKernel);
        Assert.Equal(1e-6f, c.LayerNormEps);
        Assert.Equal(1024, c.NFft);
        Assert.Equal(256, c.HopLength);
        Assert.Equal(24_000, c.SampleRate);
    }

    [Fact]
    public void IStft_OutputLength_MatchesFormula()
    {
        // For F frames at hop=256, n_fft=1024, center=True:
        // raw overlap-add length = (F - 1) * hop + n_fft
        // trimmed = raw - n_fft = (F - 1) * hop
        int frames = 10, nFft = 1024, hop = 256;
        int half = nFft / 2 + 1;
        float[] re = new float[frames * half];
        float[] im = new float[frames * half];
        float[] outAudio = IStft.Apply(re, im, frames, nFft, hop);
        Assert.Equal((frames - 1) * hop, outAudio.Length);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void Vocos_ReconstructsAudio_FromCachedMel()
    {
        string repoDir = AudioModelCache.GetRepoDirectory("charactr/vocos-mel-24khz");
        string modelPath = Path.Combine(repoDir, "model.safetensors");
        // The mel fixture is generated offline (see tools/dump_vocos_mel.py) — skip if
        // neither the cache nor the fixture is present.
        string melPath = Path.Combine(AudioModelCache.CacheRoot, "test-clips", "jfk_mel_vocos.bin");
        if (!File.Exists(modelPath) || !File.Exists(melPath))
        {
            return;
        }

        byte[] raw = File.ReadAllBytes(melPath);
        int floats = raw.Length / 4;
        int melBins = 100;
        int frames = floats / melBins;
        Assert.True(frames > 100, $"expected >100 mel frames, got {frames}");

        Tensor mel = new(new TensorShape(1, melBins, frames), DType.F32);
        unsafe
        {
            float* dst = (float*)mel.DataPointer;
            for (int i = 0; i < floats; i++) dst[i] = BitConverter.ToSingle(raw, i * 4);
        }

        SafeTensorsLoader loader = new();
        loader.Load(modelPath);
        Dictionary<string, Tensor> weights = loader.GetAllTensors();

        // OutputGain=44.53 is the per-checkpoint workaround for the mel-scale mismatch
        // between our C# MelSpectrogramExtractor and the torchaudio mel that the published
        // Vocos checkpoints were trained on. The fixture <c>jfk_mel_vocos.bin</c> is a mel
        // dumped from upstream Python and is already on the "correct" training-time scale,
        // so the gain workaround does NOT apply here — disable it.
        using Vocos vocos = new(VocosConfig.Mel24k with { OutputGain = 1.0f });
        vocos.LoadWeights(weights);

        using CpuBackend backend = new();
        float[] audio = vocos.Forward(backend, mel);
        mel.Dispose();
        loader.Dispose();

        // Length matches the iSTFT formula for the input frame count.
        Assert.Equal((frames - 1) * 256, audio.Length);

        // RMS should be in a "speech-like" range — between roughly 0.02 (very quiet) and
        // 0.5 (peak). The JFK clip is normalized speech so we expect ~0.13.
        double rms = Math.Sqrt(audio.Sum(x => (double)x * x) / audio.Length);
        Assert.InRange(rms, 0.02, 0.5);

        // Peak audio is in [-1, 1].
        float peak = audio.Max(Math.Abs);
        Assert.InRange(peak, 0.1f, 1.0f);
    }
}
