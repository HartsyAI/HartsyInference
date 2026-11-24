using HartsyInference.Audio.Models.LongCat;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;
using Files = HartsyInference.Audio.Tests.LongCat.LongCatReferenceFiles;

namespace HartsyInference.Audio.Tests.LongCat;

/// <summary>Comparisons against the dumps of tests/python-reference/longcat_audio (see its README). Each skips, never passes,
/// when the dumps or weights it needs are absent. Tolerance: relRMS 1e-5 on CPU F32.</summary>
[Trait("Category", "Integration")]
public sealed class LongCatAudioReferenceTests(ITestOutputHelper output)
{
    private const double Tolerance = 1e-5;

    private void CompareLayout(string tag, LongCatAudioLayout layout, float[] features, int frames, int segments)
    {
        for (int k = 0; k < segments; k++)
        {
            int start = LongCatAudioFeatures.SegmentStartFrame(k, layout);
            float[] windows = LongCatAudioFeatures.BuildWindows(features, frames, layout, start, 93);
            (float[] first, float[] later) = LongCatAudioFeatures.GroupForAudioProj(windows, 93, layout);

            double w = Files.RelRms(windows, Files.Read($"{tag}_windows_seg{k}.f32"));
            double f = Files.RelRms(first, Files.Read($"{tag}_first_seg{k}.f32"));
            double l = Files.RelRms(later, Files.Read($"{tag}_later_seg{k}.f32"));
            output.WriteLine($"{tag} segment {k}: windows {w:E2}, first {f:E2}, later {l:E2}");
            Assert.True(w < Tolerance && f < Tolerance && l < Tolerance, $"{tag} segment {k}");
        }
    }

    [LongCatReferenceFact("hm_whisper_states.f32", "hm_whisper_features.f32", "hm_whisper_windows_seg1.f32", "hm_whisper_later_seg1.f32", "meta.json")]
    public void HostMath_Whisper_MatchesTheReference()
    {
        const int Channels = 16;
        int videoFrames = Files.MetaInt("hm_video_length");
        float[] states = Files.Read("hm_whisper_states.f32");
        int encoderFrames = states.Length / (33 * Channels);
        int kept = 2 * videoFrames;
        float[] pooled = new float[kept * 5 * Channels];
        LongCatAudioFeatures.PoolWhisperStates(states.AsSpan(0, kept * 33 * Channels), kept, Channels, pooled);
        float[] features = LongCatAudioFeatures.ResampleFrames(pooled, kept, 5, Channels, videoFrames);

        double rms = Files.RelRms(features, Files.Read("hm_whisper_features.f32"));
        output.WriteLine($"pooled+resampled [{videoFrames}, 5, {Channels}] from {encoderFrames} encoder frames: relRMS {rms:E2}");
        Assert.True(rms < Tolerance);
        CompareLayout("hm_whisper", new LongCatAudioLayout { Blocks = 5, Channels = Channels, VideoFps = 25, AudioStride = 1 },
            features, videoFrames, 2);
    }

    [LongCatReferenceFact("hm_w2v_features.f32", "hm_w2v_windows_seg1.f32", "hm_w2v_later_seg1.f32")]
    public void HostMath_Wav2Vec2Layout_MatchesTheReference()
    {
        float[] features = Files.Read("hm_w2v_features.f32");
        CompareLayout("hm_w2v", new LongCatAudioLayout { Blocks = 12, Channels = 8, VideoFps = 16, AudioStride = 2 },
            features, features.Length / (12 * 8), 2);
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
            double rms = Files.RelRms(actual, reference);
            float maxAbs = 0;
            for (int i = 0; i < actual.Length; i++) maxAbs = Math.Max(maxAbs, Math.Abs(actual[i] - reference[i]));
            output.WriteLine($"mel128 chunk {c}: relRMS {rms:E2}, max |diff| {maxAbs:E2}");
            Assert.True(rms < Tolerance, $"chunk {c} relRMS {rms}");
        }
    }

    [LongCatWeightsFact("HARTSY_LONGCAT_WHISPER_SAFETENSORS", "speech16k.f32", "ws_features.f32", "ws_windows_seg1.f32", "ws_later_seg1.f32", "meta.json")]
    public void WhisperLargeV3Features_MatchTheReference()
    {
        float[] speech = Files.Read("speech16k.f32");
        IBackend backend = new CpuBackend();
        using SafeTensorsLoader loader = new();
        loader.Load(Environment.GetEnvironmentVariable("HARTSY_LONGCAT_WHISPER_SAFETENSORS")!);
        WhisperEncoder encoder = new(WhisperConfig.LargeV3);
        encoder.LoadWeights(loader.GetAllTensors());
        MelSpectrogramExtractor mel = new(MelSpectrogramExtractor.WhisperConfig(128));

        float[] features = LongCatWhisperFeatures.Extract(backend, encoder, mel, speech);

        int frames = Files.MetaInt("ws_video_length");
        Assert.Equal(frames * 5 * 1280, features.Length);
        double rms = Files.RelRms(features, Files.Read("ws_features.f32"));
        output.WriteLine($"Whisper-large-v3 features [{frames}, 5, 1280]: relRMS {rms:E2}");
        Assert.True(rms < Tolerance);
        CompareLayout("ws", LongCatAudioLayout.Whisper, features, frames, 2);
        encoder.Dispose();
    }

    [LongCatReferenceFact("chinese-wav2vec2-base.safetensors", "w2v_waveform.f32", "w2v_input_values.f32", "w2v_state0.f32",
        "w2vr_features.f32", "w2vr_windows_seg1.f32", "w2vr_later_seg1.f32", "meta.json")]
    public void Wav2Vec2V10Features_MatchTheReference()
    {
        float[] waveform = Files.Read("w2v_waveform.f32");
        int frames = Files.MetaInt("w2v_video_length");
        IBackend backend = new CpuBackend();
        using SafeTensorsLoader loader = new();
        loader.Load(Files.FullPath("chinese-wav2vec2-base.safetensors"));
        Wav2Vec2Encoder encoder = new(Wav2Vec2EncoderConfig.LongCatAvatarV10);
        encoder.LoadWeights(loader.GetAllTensors());

        using Tensor states = encoder.EncodeAllLayers(backend, waveform, frames);
        float[] all = states.AsReadOnlySpan<float>().ToArray();
        float[] features = LongCatAudioFeatures.SkipEmbeddingState(all, frames, 13, 768);

        float[] firstState = new float[frames * 768];
        for (int t = 0; t < frames; t++) all.AsSpan(t * 13 * 768, 768).CopyTo(firstState.AsSpan(t * 768, 768));
        double first = Files.RelRms(firstState, Files.Read("w2v_state0.f32"));
        double rms = Files.RelRms(features, Files.Read("w2vr_features.f32"));
        output.WriteLine($"wav2vec2 features [{frames}, 12, 768]: relRMS {rms:E2}, state 0 {first:E2}");
        Assert.True(first < Tolerance && rms < Tolerance);
        CompareLayout("w2vr", LongCatAudioLayout.Wav2Vec2, features, frames, 2);
    }
}
