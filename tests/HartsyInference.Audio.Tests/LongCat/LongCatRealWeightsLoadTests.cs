using HartsyInference.Audio.Models.LongCat;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests.LongCat;

/// <summary>The real checkpoints load into the encoders and give finite, non-degenerate features. This is a load check, not
/// parity: no reference values are compared. Skips unless the weight files named by the variables exist.</summary>
[Trait("Category", "Integration")]
public sealed class LongCatRealWeightsLoadTests(ITestOutputHelper output)
{
    private static float[] Tone(int samples)
    {
        float[] audio = new float[samples];
        for (int i = 0; i < samples; i++)
            audio[i] = 0.3f * MathF.Sin(i * 0.031f) * (0.6f + 0.4f * MathF.Sin(i * 0.0004f)) + 0.1f * MathF.Sin(i * 0.17f);
        return audio;
    }

    private static void AssertLive(ReadOnlySpan<float> values)
    {
        double sum = 0, sumSq = 0;
        foreach (float v in values)
        {
            Assert.True(float.IsFinite(v));
            sum += v;
            sumSq += (double)v * v;
        }
        double mean = sum / values.Length;
        Assert.True(sumSq / values.Length - mean * mean > 1e-8, "features are constant");
    }

    [Trait("Category", "Slow")]
    [LongCatWeightsFact("HARTSY_LONGCAT_WHISPER_SAFETENSORS")]
    public void WhisperLargeV3_Loads_AndGivesFiveBlocksOf1280()
    {
        IBackend backend = new CpuBackend();
        using SafeTensorsLoader loader = new();
        loader.Load(Environment.GetEnvironmentVariable("HARTSY_LONGCAT_WHISPER_SAFETENSORS")!);
        WhisperEncoder encoder = new(WhisperConfig.LargeV3);
        encoder.LoadWeights(loader.GetAllTensors());
        MelSpectrogramExtractor mel = new(MelSpectrogramExtractor.WhisperConfig(128));

        System.Diagnostics.Stopwatch clock = System.Diagnostics.Stopwatch.StartNew();
        float[] features = LongCatWhisperFeatures.Extract(backend, encoder, mel, Tone(16_000 * 3));

        output.WriteLine($"3 s clip -> [75, 5, 1280] on the CPU backend in {clock.Elapsed.TotalSeconds:0.0} s");
        Assert.Equal(75 * 5 * 1280, features.Length);
        AssertLive(features);
        encoder.Dispose();
    }

    [LongCatWeightsFact("HARTSY_LONGCAT_W2V_CHECKPOINT")]
    public void ChineseWav2Vec2Base_PickleLoads_AndGivesThirteenStates()
    {
        IBackend backend = new CpuBackend();
        using AnyFormatCheckpointLoader loader = new();
        loader.Load(Environment.GetEnvironmentVariable("HARTSY_LONGCAT_W2V_CHECKPOINT")!);
        Dictionary<string, Tensor> weights = new();
        foreach ((string name, Tensor tensor) in loader.GetAllTensors())
            weights[name.StartsWith("wav2vec2.", StringComparison.Ordinal) ? name["wav2vec2.".Length..] : name] = tensor;
        Wav2Vec2Encoder encoder = new(Wav2Vec2EncoderConfig.LongCatAvatarV10);
        encoder.LoadWeights(weights);

        float[] speech = Tone(16_000 * 2);
        int frames = LongCatAudioFeatures.VideoFrames(speech.Length, 16_000, 32);
        using Tensor states = encoder.EncodeAllLayers(backend, speech, frames);

        Assert.Equal(new TensorShape(frames, 13, 768), states.Shape);
        AssertLive(states.AsReadOnlySpan<float>());
        output.WriteLine($"{weights.Count} tensors loaded; [{frames}, 13, 768] states");
    }
}
