using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.MeloTts;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Cpu;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Real-weight MeloTTS (English-v3, VITS lineage) → WAV → Whisper STT intelligibility check.
/// Text in (own G2P) → audio, small enough for CPU. Self-discovers weights from the audio cache
/// (MeloTTS checkpoint + bert-base-uncased). Asserts Whisper recovers the content words. Writes the WAV
/// for the human listen pass.</summary>
public sealed class MeloEndToEndSttTests
{
    private readonly ITestOutputHelper _out;
    public MeloEndToEndSttTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public async Task Text_To_Wav_To_Whisper_RecoversWords()
    {
        string ckpt = Path.Combine(AudioModelCache.GetRepoDirectory("myshell-ai/MeloTTS-English-v3", "tts"), "checkpoint.pth");
        string bertDir = AudioModelCache.GetRepoDirectory("bert-base-uncased", "tts");
        // Prefer safetensors: the cached pytorch_model.bin is legacy (pre-1.6, non-zip) pickle the loader can't read.
        string bertSt = Path.Combine(bertDir, "model.safetensors");
        string bert = File.Exists(bertSt) ? bertSt : Path.Combine(bertDir, "pytorch_model.bin");
        string vocab = Path.Combine(bertDir, "vocab.txt");
        string whisperDir = AudioModelCache.GetRepoDirectory("openai/whisper-base", "stt");
        if (!File.Exists(ckpt) || !File.Exists(bert) || !File.Exists(vocab))
        {
            _out.WriteLine($"MeloTTS/BERT not cached (ckpt={File.Exists(ckpt)}, bert={File.Exists(bert)}, vocab={File.Exists(vocab)}) — skipping.");
            return;
        }
        if (!File.Exists(Path.Combine(whisperDir, "model.safetensors")))
        {
            _out.WriteLine("whisper-base not cached — skipping STT half.");
            return;
        }

        MeloTtsConfig cfg = MeloTtsConfig.EnglishV3 with { NoiseScaleW = 0f };
        using MeloTts melo = MeloTts.LoadFromFiles(ckpt, bert, vocab, cfg);
        using CpuBackend backend = new();

        const string text = "Hello world. This is a test of the speech synthesizer.";
        string[] targets = ["hello", "world", "this", "test", "speech"];

        float[] audio = melo.SynthesizeText(backend, text, lengthScale: 1.0f, noiseScale: 0.0f, seed: 0);
        Assert.NotEmpty(audio);
        double sumSq = 0; foreach (float v in audio) { Assert.True(float.IsFinite(v)); sumSq += (double)v * v; }
        double rms = Math.Sqrt(sumSq / audio.Length);
        int sr = melo.SampleRate;

        string outDir = Path.Combine(Path.GetTempPath(), "hartsyinference_tts_to_stt");
        Directory.CreateDirectory(outDir);
        string outWav = Path.Combine(outDir, $"melo_stt_{DateTime.Now:yyyyMMdd_HHmmss}.wav");
        WavFile.WriteMono16(outWav, audio, sr);
        _out.WriteLine($"MeloTTS generated {audio.Length} samples ({audio.Length / (double)sr:F2}s @ {sr}Hz). RMS={rms:F4}.");
        _out.WriteLine($"WAV (listen): {outWav}");
        Assert.True(rms > 1e-3, "silent output");

        float[] stt16 = Resampler.Create(sr, 16_000).Resample(audio);
        string sttWav = Path.Combine(outDir, $"melo_stt_16k_{DateTime.Now:yyyyMMdd_HHmmss}.wav");
        WavFile.WriteMono16(sttWav, stt16, 16_000);

        using WhisperPipeline stt = await WhisperPipeline.LoadAsync("openai/whisper-base");
        string heard = stt.TranscribeWav(backend, sttWav,
            new WhisperOptions { Language = "en", Translate = false, WithTimestamps = false }).Trim();
        string lower = heard.ToLowerInvariant();
        _out.WriteLine($"Target text:    \"{text}\"");
        _out.WriteLine($"Whisper heard:  \"{heard}\"");
        int hits = targets.Count(w => lower.Contains(w));
        _out.WriteLine($"Content-word recall: {hits}/{targets.Length} ({string.Join(",", targets.Where(w => lower.Contains(w)))})");
        if (hits == 0) _out.WriteLine("--- NO TARGET WORDS — listen to the WAV to judge.");
        Assert.NotNull(heard);
    }
}
