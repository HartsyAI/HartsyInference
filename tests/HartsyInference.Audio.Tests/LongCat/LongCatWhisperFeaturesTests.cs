using HartsyInference.Audio.Models.LongCat;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests.LongCat;

/// <summary>The chunked Whisper feature driver on a 32-layer, hidden-8 random-weight encoder with 128 mel bins: the wiring
/// (30 s chunks, 33 states, 5-block pooling, truncation to 2 x video frames, 25 fps resample) against a plain
/// composition of the same pieces. Real-checkpoint values are the Integration test's job.</summary>
public sealed class LongCatWhisperFeaturesTests
{
    private const int Hidden = 8;

    private static float[] Speech(int samples)
    {
        float[] audio = new float[samples];
        for (int i = 0; i < samples; i++)
            audio[i] = 0.3f * MathF.Sin(i * 0.031f) + 0.2f * MathF.Sin(i * 0.0071f + 1f) * MathF.Sin(i * 0.00013f);
        return audio;
    }

    // Stack every chunk's 33 states into [frames, 33, hidden] the long way, then pool the whole thing at once.
    private static float[] Reference(WhisperEncoder encoder, MelSpectrogramExtractor mel, IBackend backend, float[] speech, int fps)
    {
        int videoFrames = LongCatAudioFeatures.VideoFrames(speech.Length, 16_000, fps);
        List<float[]> chunkStates = [];
        int frames = 0;
        for (int start = 0; start < speech.Length; start += LongCatWhisperFeatures.MelChunkSamples)
        {
            int length = Math.Min(LongCatWhisperFeatures.MelChunkSamples, speech.Length - start);
            float[] melValues = new float[128 * LongCatWhisperFeatures.EncoderChunkFrames];
            mel.ComputeZeroPadded(speech.AsSpan(start, length), LongCatWhisperFeatures.MelChunkSamples, melValues);
            using Tensor melTensor = new(new TensorShape(1, 128, LongCatWhisperFeatures.EncoderChunkFrames), DType.F32);
            melValues.CopyTo(melTensor.AsSpan<float>());
            Tensor[] states = encoder.ForwardAllHiddenStates(backend, melTensor);
            int t = (int)states[0].Shape[1];
            float[] stacked = new float[t * 33 * Hidden];
            for (int f = 0; f < t; f++)
                for (int s = 0; s < 33; s++)
                    states[s].AsReadOnlySpan<float>().Slice(f * Hidden, Hidden).CopyTo(stacked.AsSpan((f * 33 + s) * Hidden, Hidden));
            foreach (Tensor state in states) state.Dispose();
            chunkStates.Add(stacked);
            frames += t;
        }
        float[] all = chunkStates.SelectMany(c => c).ToArray();
        int kept = Math.Min(frames, 2 * videoFrames);
        float[] pooled = new float[kept * 5 * Hidden];
        LongCatAudioFeatures.PoolWhisperStates(all.AsSpan(0, kept * 33 * Hidden), kept, Hidden, pooled);
        return LongCatAudioFeatures.ResampleFrames(pooled, kept, 5, Hidden, videoFrames);
    }

    [Theory]
    [InlineData(19_200)]
    [InlineData(31 * 16_000)]
    public void Extract_MatchesAPlainCompositionOfTheSamePieces(int samples)
    {
        WhisperConfig cfg = TinyWhisperEncoder.Config(layers: 32, melBins: 128, hidden: Hidden);
        (WhisperEncoder encoder, List<Tensor> weights) = TinyWhisperEncoder.Build(cfg);
        IBackend backend = new CpuBackend();
        MelSpectrogramExtractor mel = new(MelSpectrogramExtractor.WhisperConfig(128));
        float[] speech = Speech(samples);

        float[] features = LongCatWhisperFeatures.Extract(backend, encoder, mel, speech);
        float[] expected = Reference(encoder, mel, backend, speech, 25);

        Assert.Equal(LongCatAudioFeatures.VideoFrames(samples, 16_000, 25) * 5 * Hidden, features.Length);
        Assert.Equal(expected.Length, features.Length);
        double sumSq = 0, errSq = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            sumSq += (double)expected[i] * expected[i];
            errSq += (double)(features[i] - expected[i]) * (features[i] - expected[i]);
            Assert.True(float.IsFinite(features[i]));
        }
        Assert.True(Math.Sqrt(errSq / sumSq) < 1e-6, $"relRMS {Math.Sqrt(errSq / sumSq)}");
        encoder.Dispose();
        foreach (Tensor w in weights) w.Dispose();
    }

    [Fact]
    public void Extract_RejectsAnEncoderThatIsNotThirtyThreeStates()
    {
        WhisperConfig cfg = TinyWhisperEncoder.Config(layers: 4, melBins: 128, hidden: Hidden);
        (WhisperEncoder encoder, List<Tensor> weights) = TinyWhisperEncoder.Build(cfg);
        MelSpectrogramExtractor mel = new(MelSpectrogramExtractor.WhisperConfig(128));

        Assert.Throws<ArgumentException>(() => LongCatWhisperFeatures.Extract(new CpuBackend(), encoder, mel, new float[16_000]));
        encoder.Dispose();
        foreach (Tensor w in weights) w.Dispose();
    }

    [Fact]
    public void Extract_RejectsAnEightyBinMelExtractor()
    {
        WhisperConfig cfg = TinyWhisperEncoder.Config(layers: 32, melBins: 128, hidden: Hidden);
        (WhisperEncoder encoder, List<Tensor> weights) = TinyWhisperEncoder.Build(cfg);
        MelSpectrogramExtractor mel = new(MelSpectrogramExtractor.WhisperConfig(80));

        Assert.Throws<ArgumentException>(() => LongCatWhisperFeatures.Extract(new CpuBackend(), encoder, mel, new float[16_000]));
        encoder.Dispose();
        foreach (Tensor w in weights) w.Dispose();
    }
}
