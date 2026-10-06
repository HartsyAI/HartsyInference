using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.IndexTts2;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Cpu;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests.IndexTts2;

/// <summary>End-to-end real-weight generation through <see cref="IndexTts2Pipeline"/> in IndexTTS-2.0 mode — the
/// conformer_perceiver speaker conditioning, the MaskGCT RepCodec, and the second-GPT-pass + <c>gpt_layer</c> handoff
/// that only 2.0 has. Gated on <c>INDEXTTS2_V20_DIR</c> (the <c>IndexTeam/IndexTTS-2</c> files: bpe.model, gpt.pth,
/// s2mel.pth, wav2vec2bert_stats.pt; feat1.pt/feat2.pt optional), <c>INDEXTTS2_MASKGCT_CODEC_PATH</c>
/// (<c>amphion/MaskGCT</c>'s <c>semantic_codec/model.safetensors</c>), plus the same shared auxiliaries and
/// <c>INDEXTTS2_REF_WAV</c> as the 2.5 test. Confirms the whole 2.0 pipeline runs and yields finite, non-silent
/// 22050 Hz PCM; whether it is intelligible is checked by transcribing the written WAV (see the PR notes).</summary>
[Trait("Category", "Integration")]
public sealed class IndexTts2V20PipelineRealWeightTests(ITestOutputHelper output)
{
    [Fact]
    public async Task Synthesize_ProducesFiniteNonSilentAudio_AgainstRealV20Checkpoints()
    {
        string? v20Dir = Environment.GetEnvironmentVariable("INDEXTTS2_V20_DIR");
        string? codecPath = Environment.GetEnvironmentVariable("INDEXTTS2_MASKGCT_CODEC_PATH");
        string? w2vBertPath = Environment.GetEnvironmentVariable("INDEXTTS2_W2VBERT_SAFETENSORS_PATH");
        string? campplusPath = Environment.GetEnvironmentVariable("INDEXTTS2_CAMPPLUS_PATH");
        string? bigVganPath = Environment.GetEnvironmentVariable("INDEXTTS2_BIGVGAN_PT_PATH");
        string? refWavPath = Environment.GetEnvironmentVariable("INDEXTTS2_REF_WAV");
        string? outDir = Environment.GetEnvironmentVariable("INDEXTTS2_OUT_DIR");
        if (string.IsNullOrEmpty(v20Dir) || !Directory.Exists(v20Dir)) { output.WriteLine("INDEXTTS2_V20_DIR missing — skip."); return; }
        foreach ((string name, string? path) in new[]
        {
            ("INDEXTTS2_MASKGCT_CODEC_PATH", codecPath), ("INDEXTTS2_W2VBERT_SAFETENSORS_PATH", w2vBertPath),
            ("INDEXTTS2_CAMPPLUS_PATH", campplusPath), ("INDEXTTS2_BIGVGAN_PT_PATH", bigVganPath), ("INDEXTTS2_REF_WAV", refWavPath),
        })
        {
            if (string.IsNullOrEmpty(path) || !File.Exists(path)) { output.WriteLine($"{name} missing — skip."); return; }
        }

        string bpePath = Path.Combine(v20Dir, "bpe.model");
        string gptPath = Path.Combine(v20Dir, "gpt.pth");
        string s2melPath = Path.Combine(v20Dir, "s2mel.pth");
        string statsPath = Path.Combine(v20Dir, "wav2vec2bert_stats.pt");
        foreach (string p in new[] { bpePath, gptPath, s2melPath, statsPath })
            if (!File.Exists(p)) { output.WriteLine($"{p} missing — skip."); return; }
        string feat1 = Path.Combine(v20Dir, "feat1.pt"), feat2 = Path.Combine(v20Dir, "feat2.pt");
        bool haveFeat = File.Exists(feat1) && File.Exists(feat2);
        string? qwenDir = Environment.GetEnvironmentVariable("INDEXTTS2_QWEN_DIR");
        bool haveQwen = !string.IsNullOrEmpty(qwenDir) && Directory.Exists(qwenDir);
        string? emoRefPath = Environment.GetEnvironmentVariable("INDEXTTS2_EMO_REF_WAV");
        bool haveEmoRef = !string.IsNullOrEmpty(emoRefPath) && File.Exists(emoRefPath);
        using CpuBackend backend = new();

        using IndexTts2Pipeline pipeline = await IndexTts2Pipeline.LoadAsync(
            bpePath, gptPath, s2melPath, codecPath!, w2vBertPath!, statsPath, campplusPath!, bigVganPath!,
            feat1Path: haveFeat ? feat1 : null, feat2Path: haveFeat ? feat2 : null,
            qwenEmoDir: haveQwen ? qwenDir : null, qwenBackend: haveQwen ? backend : null, cfg: IndexTts2Config.V2_0);
        Assert.Equal("indextts2-2.0", pipeline.ModelName);

        WavFile.DecodedAudio refAudio = WavFile.Read(refWavPath!);
        string text = Environment.GetEnvironmentVariable("INDEXTTS2_TEXT") ?? "Hello there, this is a test.";

        float[] pcm = pipeline.Synthesize(backend, text, refAudio.Channels[0], refAudio.SampleRate, new IndexTts2Options { MaxMelTokens = 200, Seed = 7 });
        AssertAudible(pcm, "neutral");
        Write(outDir, "indextts2_v20_neutral.wav", pcm);

        if (haveFeat)
        {
            // Explicit emotion-vector mode: happy, through the feat1/feat2 exemplar lookup.
            float[] happy = pipeline.Synthesize(backend, text, refAudio.Channels[0], refAudio.SampleRate,
                new IndexTts2Options { MaxMelTokens = 200, Seed = 7, EmoVector = [0.9f, 0, 0, 0, 0, 0, 0, 0] });
            AssertAudible(happy, "emo-vector");
            Write(outDir, "indextts2_v20_happy.wav", happy);
        }

        if (haveEmoRef)
        {
            // Separate emotion-reference clip: the voice stays the speaker's, the emotion comes from the other clip.
            WavFile.DecodedAudio emoAudio = WavFile.Read(emoRefPath!);
            float[] fromRef = pipeline.Synthesize(backend, text, refAudio.Channels[0], refAudio.SampleRate,
                new IndexTts2Options { MaxMelTokens = 200, Seed = 7, EmoAudioReference = (emoAudio.Channels[0], emoAudio.SampleRate), EmoAlpha = 0.8f });
            AssertAudible(fromRef, "emo-reference");
            Write(outDir, "indextts2_v20_emoref.wav", fromRef);
        }

        if (haveFeat && haveQwen)
        {
            // Free-text emotion through the lazily-loaded QwenEmotion classifier.
            float[] fromText = pipeline.Synthesize(backend, text, refAudio.Channels[0], refAudio.SampleRate,
                new IndexTts2Options { MaxMelTokens = 200, Seed = 7, UseEmoText = true, EmoText = "I am absolutely furious about this!" });
            AssertAudible(fromText, "emo-text");
            Write(outDir, "indextts2_v20_angry_text.wav", fromText);
        }
    }

    private void AssertAudible(float[] pcm, string label)
    {
        Assert.True(pcm.Length > 0);
        double sumSq = 0;
        foreach (float v in pcm)
        {
            Assert.True(float.IsFinite(v) && v is >= -1f and <= 1f);
            sumSq += (double)v * v;
        }
        double rms = Math.Sqrt(sumSq / pcm.Length);
        output.WriteLine($"[{label}] {pcm.Length / 22_050.0:F2}s, RMS {rms:F5}");
        Assert.True(rms > 1e-4, $"[{label}] audio is suspiciously close to silence (RMS {rms:F6}).");
    }

    private void Write(string? dir, string name, float[] pcm)
    {
        if (string.IsNullOrEmpty(dir)) return;
        Directory.CreateDirectory(dir);
        string path = Path.Combine(dir, name);
        WavFile.WriteMono16(path, pcm, 22_050);
        output.WriteLine($"Wrote {path}");
    }
}
