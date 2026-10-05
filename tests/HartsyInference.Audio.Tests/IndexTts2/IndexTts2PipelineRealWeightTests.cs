using HartsyInference.Audio.Io;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Cpu;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests.IndexTts2;

/// <summary>End-to-end real-weight generation through <see cref="IndexTts2Pipeline"/> — the capstone check for
/// every component built this session (semantic features, CAM++, semantic codec, GPT conditioning, S2Mel DiT,
/// BigVGAN). Gated on <c>INDEXTTS2_V25_DIR</c> (the main IndexTTS-2.5 repo's files: tiktoken, gpt.pth,
/// s2mel.pth, codec.pth, wav2vec2bert_stats.pt — feat1.pt/feat2.pt and qwen0.6bemo4-merge/ optional),
/// <c>INDEXTTS2_W2VBERT_SAFETENSORS_PATH</c>, <c>INDEXTTS2_CAMPPLUS_PATH</c>, <c>INDEXTTS2_BIGVGAN_PT_PATH</c>,
/// and <c>INDEXTTS2_REF_WAV</c> (any real speech clip). Confirms a short real synthesis call runs the entire
/// pipeline without crashing and produces finite, audibly-non-silent 22050 Hz PCM — says nothing about
/// perceptual quality (that is swarm.hartsy.ai's job, same posture as every other model shipped this way).</summary>
[Trait("Category", "Integration")]
public sealed class IndexTts2PipelineRealWeightTests(ITestOutputHelper output)
{
    [Fact]
    public async Task Synthesize_ProducesFiniteNonSilentAudio_AgainstRealCheckpoints()
    {
        string? v25Dir = Environment.GetEnvironmentVariable("INDEXTTS2_V25_DIR");
        string? w2vBertPath = Environment.GetEnvironmentVariable("INDEXTTS2_W2VBERT_SAFETENSORS_PATH");
        string? campplusPath = Environment.GetEnvironmentVariable("INDEXTTS2_CAMPPLUS_PATH");
        string? bigVganPath = Environment.GetEnvironmentVariable("INDEXTTS2_BIGVGAN_PT_PATH");
        string? refWavPath = Environment.GetEnvironmentVariable("INDEXTTS2_REF_WAV");
        if (string.IsNullOrEmpty(v25Dir) || !Directory.Exists(v25Dir)) { output.WriteLine("INDEXTTS2_V25_DIR missing — skip."); return; }
        if (string.IsNullOrEmpty(w2vBertPath) || !File.Exists(w2vBertPath)) { output.WriteLine("INDEXTTS2_W2VBERT_SAFETENSORS_PATH missing — skip."); return; }
        if (string.IsNullOrEmpty(campplusPath) || !File.Exists(campplusPath)) { output.WriteLine("INDEXTTS2_CAMPPLUS_PATH missing — skip."); return; }
        if (string.IsNullOrEmpty(bigVganPath) || !File.Exists(bigVganPath)) { output.WriteLine("INDEXTTS2_BIGVGAN_PT_PATH missing — skip."); return; }
        if (string.IsNullOrEmpty(refWavPath) || !File.Exists(refWavPath)) { output.WriteLine("INDEXTTS2_REF_WAV missing — skip."); return; }

        string tiktokenPath = Path.Combine(v25Dir, "multilingual_zh_ja_yue_char_del.tiktoken");
        string gptPath = Path.Combine(v25Dir, "gpt.pth");
        string s2melPath = Path.Combine(v25Dir, "s2mel.pth");
        string codecPath = Path.Combine(v25Dir, "codec.pth");
        string statsPath = Path.Combine(v25Dir, "wav2vec2bert_stats.pt");
        foreach (string p in new[] { tiktokenPath, gptPath, s2melPath, codecPath, statsPath })
            if (!File.Exists(p)) { output.WriteLine($"{p} missing — skip."); return; }

        using IndexTts2Pipeline pipeline = await IndexTts2Pipeline.LoadAsync(
            tiktokenPath, gptPath, s2melPath, codecPath, w2vBertPath, statsPath, campplusPath, bigVganPath);

        WavFile.DecodedAudio refAudio = WavFile.Read(refWavPath);
        using CpuBackend backend = new();

        IndexTts2Options opts = new() { MaxMelTokens = 150, Seed = 7 };
        float[] pcm = pipeline.Synthesize(backend, "Hello there.", refAudio.Channels[0], refAudio.SampleRate, opts);

        Assert.True(pcm.Length > 0);
        foreach (float v in pcm) Assert.True(float.IsFinite(v) && v is >= -1f and <= 1f);

        double sumSq = 0;
        foreach (float v in pcm) sumSq += (double)v * v;
        double rms = Math.Sqrt(sumSq / pcm.Length);
        output.WriteLine($"Generated {pcm.Length / 22_050.0:F2}s, RMS {rms:F5}");
        Assert.True(rms > 1e-4, $"Generated audio is suspiciously close to silence (RMS {rms:F6}).");

        string outPath = "/tmp/indextts2_e2e_test.wav";
        WavFile.WriteMono16(outPath, pcm, 22_050);
        output.WriteLine($"Wrote {outPath}");
    }
}
