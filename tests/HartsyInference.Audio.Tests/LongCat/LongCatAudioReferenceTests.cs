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
using Files = HartsyInference.Audio.Tests.LongCat.LongCatReferenceFiles;

namespace HartsyInference.Audio.Tests.LongCat;

/// <summary>Comparisons against the dumps of tests/python-reference/longcat_audio (see its README). Each skips, never passes,
/// when the dumps or weights it needs are absent. Every stage is measured and printed before the single assertion, so one run
/// shows where a difference starts. Tolerance: relRMS 1e-5 on CPU F32.</summary>
[Trait("Category", "Integration")]
public sealed class LongCatAudioReferenceTests(ITestOutputHelper output)
{
    private const double Tolerance = 1e-5;

    // The mel stage holds the same relRMS bound as the rest; its per-element max difference is printed, not asserted.
    private const double MelTolerance = Tolerance;
    private readonly List<(string Stage, double RelRms)> _stages = [];

    private readonly Dictionary<string, double> _limits = [];

    private void Stage(string name, ReadOnlySpan<float> actual, ReadOnlySpan<float> expected, double limit = Tolerance)
    {
        _stages.Add((name, Files.RelRms(actual, expected)));
        _limits[name] = limit;
    }

    private void AssertStages()
    {
        foreach ((string stage, double rms) in _stages) output.WriteLine($"{stage}: relRMS {rms:E2}");
        string[] failed = _stages.Where(s => !(s.RelRms < _limits[s.Stage])).Select(s => s.Stage).ToArray();
        Assert.True(failed.Length == 0, "above the stage limit: " + string.Join(", ", failed));
    }

    private void StageLayout(string tag, LongCatAudioLayout layout, float[] features, int frames, int segments)
    {
        for (int k = 0; k < segments; k++)
        {
            int start = LongCatAudioFeatures.SegmentStartFrame(k, layout);
            float[] windows = LongCatAudioFeatures.BuildWindows(features, frames, layout, start, 93);
            (float[] first, float[] later) = LongCatAudioFeatures.GroupForAudioProj(windows, 93, layout);
            Stage($"{tag} segment {k} windows", windows, Files.Read($"{tag}_windows_seg{k}.f32"));
            Stage($"{tag} segment {k} first-frame group", first, Files.Read($"{tag}_first_seg{k}.f32"));
            Stage($"{tag} segment {k} later groups", later, Files.Read($"{tag}_later_seg{k}.f32"));
        }
    }

    [LongCatReferenceFact("hm_whisper_states.f32", "hm_whisper_features.f32", "hm_whisper_windows_seg1.f32",
        "hm_whisper_later_seg1.f32", "meta.json")]
    public void HostMath_Whisper_MatchesTheReference()
    {
        const int Channels = 16;
        int videoFrames = Files.MetaInt("hm_video_length");
        float[] states = Files.Read("hm_whisper_states.f32");
        int kept = 2 * videoFrames;
        float[] pooled = new float[kept * 5 * Channels];
        LongCatAudioFeatures.PoolWhisperStates(states.AsSpan(0, kept * 33 * Channels), kept, Channels, pooled);
        float[] features = LongCatAudioFeatures.ResampleFrames(pooled, kept, 5, Channels, videoFrames);

        Stage("pooled + resampled", features, Files.Read("hm_whisper_features.f32"));
        StageLayout("hm_whisper", new LongCatAudioLayout { Blocks = 5, Channels = Channels, VideoFps = 25, AudioStride = 1 },
            features, videoFrames, 2);
        AssertStages();
    }

    [LongCatReferenceFact("hm_w2v_features.f32", "hm_w2v_windows_seg1.f32", "hm_w2v_later_seg1.f32")]
    public void HostMath_Wav2Vec2Layout_MatchesTheReference()
    {
        float[] features = Files.Read("hm_w2v_features.f32");
        StageLayout("hm_w2v", new LongCatAudioLayout { Blocks = 12, Channels = 8, VideoFps = 16, AudioStride = 2 },
            features, features.Length / (12 * 8), 2);
        AssertStages();
    }

    [LongCatReferenceFact("speech16k.f32", "mel128_chunks.f32")]
    public void Mel128_MatchesHfWhisperFeatureExtractor()
    {
        float[] speech = Files.Read("speech16k.f32");
        float[] expected = Files.Read("mel128_chunks.f32");
        const int Chunk = LongCatWhisperFeatures.MelChunkSamples;
        int chunks = (speech.Length + Chunk - 1) / Chunk;
        Assert.Equal(chunks * 128 * 3000, expected.Length);
        MelSpectrogramExtractor mel = new(MelSpectrogramExtractor.WhisperConfig(128));
        for (int c = 0; c < chunks; c++)
        {
            float[] actual = new float[128 * 3000];
            mel.ComputeZeroPadded(speech.AsSpan(c * Chunk, Math.Min(Chunk, speech.Length - c * Chunk)), Chunk, actual);
            ReadOnlySpan<float> reference = expected.AsSpan(c * 128 * 3000, 128 * 3000);
            float maxAbs = 0;
            for (int i = 0; i < actual.Length; i++) maxAbs = Math.Max(maxAbs, Math.Abs(actual[i] - reference[i]));
            output.WriteLine($"mel128 chunk {c}: max |diff| {maxAbs:E2}");
            Stage($"mel128 chunk {c}", actual, reference, MelTolerance);
        }
        AssertStages();
    }

    [LongCatWeightsFact("HARTSY_LONGCAT_WHISPER_SAFETENSORS", "speech16k.f32", "ws_features.f32", "ws_states_first.f32", "ws_states_last.f32",
        "ws_windows_seg1.f32", "ws_later_seg1.f32", "meta.json")]
    public void WhisperLargeV3Features_MatchTheReference()
    {
        float[] speech = Files.Read("speech16k.f32");
        IBackend backend = new CpuBackend();
        using SafeTensorsLoader loader = new();
        loader.Load(Environment.GetEnvironmentVariable("HARTSY_LONGCAT_WHISPER_SAFETENSORS")!);
        WhisperEncoder encoder = new(WhisperConfig.LargeV3);
        encoder.LoadWeights(loader.GetAllTensors());
        MelSpectrogramExtractor mel = new(MelSpectrogramExtractor.WhisperConfig(128));

        // Chunk 0's first and last hidden states, apart from the pooling, so an encoder difference shows before the host math.
        float[] melValues = new float[128 * 3000];
        mel.ComputeZeroPadded(speech.AsSpan(0, Math.Min(speech.Length, LongCatWhisperFeatures.MelChunkSamples)),
            LongCatWhisperFeatures.MelChunkSamples, melValues);
        using Tensor melTensor = new(new TensorShape(1, 128, 3000), DType.F32);
        melValues.CopyTo(melTensor.AsSpan<float>());
        Tensor[] states = encoder.ForwardAllHiddenStates(backend, melTensor);
        Stage("chunk 0 state 0", states[0].AsReadOnlySpan<float>(), Files.Read("ws_states_first.f32").AsSpan(0, 1500 * 1280));
        Stage("chunk 0 state 32 (post-LN)", states[32].AsReadOnlySpan<float>(), Files.Read("ws_states_last.f32").AsSpan(0, 1500 * 1280));
        foreach (Tensor state in states) state.Dispose();

        float[] features = LongCatWhisperFeatures.Extract(backend, encoder, mel, speech);
        int frames = Files.MetaInt("ws_video_length");
        Assert.Equal(frames * 5 * 1280, features.Length);
        Stage("pooled [T, 5, 1280]", features, Files.Read("ws_features.f32"));
        StageLayout("ws", LongCatAudioLayout.Whisper, features, frames, 2);
        encoder.Dispose();
        AssertStages();
    }

    [LongCatWeightsFact("HARTSY_LONGCAT_W2V_CHECKPOINT", "w2v_waveform.f32", "w2v_input_values.f32", "w2v_state0.f32", "w2vr_features.f32",
        "w2vr_windows_seg1.f32", "w2vr_later_seg1.f32", "meta.json")]
    public void Wav2Vec2V10Features_MatchTheReference()
    {
        float[] waveform = Files.Read("w2v_waveform.f32");
        Assert.True(Files.MetaBool("w2v_do_normalize"), "the reference extractor did not normalise: NormalizeInput must be off");
        float[] inputValues = Files.Read("w2v_input_values.f32");
        int frames = Files.MetaInt("w2v_video_length");
        IBackend backend = new CpuBackend();
        using AnyFormatCheckpointLoader loader = new();
        loader.Load(Environment.GetEnvironmentVariable("HARTSY_LONGCAT_W2V_CHECKPOINT")!);
        Dictionary<string, Tensor> weights = new();
        foreach ((string name, Tensor tensor) in loader.GetAllTensors())
            weights[name.StartsWith("wav2vec2.", StringComparison.Ordinal) ? name["wav2vec2.".Length..] : name] = tensor;
        float[] expectedStates = Files.Read("w2vr_features.f32");

        // Run 1 starts from the extractor's output, so the encoder alone is measured; run 2 from the waveform, adding the z-normalisation.
        Wav2Vec2Encoder encoder = new(Wav2Vec2EncoderConfig.LongCatAvatarV10 with { NormalizeInput = false });
        encoder.LoadWeights(weights);
        using (Tensor fromValues = encoder.EncodeAllLayers(backend, inputValues, frames))
            StageWav2Vec2("encoder on input_values", fromValues, frames, expectedStates);

        Wav2Vec2Encoder full = new(Wav2Vec2EncoderConfig.LongCatAvatarV10);
        full.LoadWeights(weights);
        using Tensor fromWaveform = full.EncodeAllLayers(backend, waveform, frames);
        float[] features = StageWav2Vec2("z-normalise + encoder on the waveform", fromWaveform, frames, expectedStates);
        StageLayout("w2vr", LongCatAudioLayout.Wav2Vec2, features, frames, 2);
        AssertStages();

        float[] StageWav2Vec2(string name, Tensor states, int count, float[] expected)
        {
            float[] stacked = states.AsReadOnlySpan<float>().ToArray();
            float[] layers = LongCatAudioFeatures.SkipEmbeddingState(stacked, count, 13, 768);
            float[] state0 = new float[count * 768];
            for (int t = 0; t < count; t++) stacked.AsSpan(t * 13 * 768, 768).CopyTo(state0.AsSpan(t * 768, 768));
            Stage($"{name}: state 0", state0, Files.Read("w2v_state0.f32"));
            Stage($"{name}: states 1-12", layers, expected);
            return layers;
        }
    }
}
