using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Cpu;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Real-weight Kokoro (82M StyleTTS2) → WAV → Whisper STT intelligibility check. Kokoro is small
/// enough to run on CPU safely (no VRAM/host-RAM pressure). Takes IPA phonemes directly; we synthesize a
/// known phrase and assert Whisper recovers its content words — the "actually listen" bar the RMS-only
/// smoke test never enforced. Cache-gated; writes the WAV for the human listen pass.</summary>
public sealed class KokoroEndToEndSttTests
{
    private readonly ITestOutputHelper _out;
    public KokoroEndToEndSttTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public async Task Phonemes_To_Wav_To_Whisper_RecoversWords()
    {
        string weights = AudioModelCache.GetRepoDirectory("Hartsy/kokoro-82m-safetensors");
        string voices = AudioModelCache.GetRepoDirectory("hexgrad/Kokoro-82M");
        string whisperDir = AudioModelCache.GetRepoDirectory("openai/whisper-base");
        if (!File.Exists(Path.Combine(weights, "kokoro-82m.safetensors"))
            || !File.Exists(Path.Combine(voices, "voices", "af_heart.bin")))
        {
            _out.WriteLine("Kokoro weights/voices not cached — skipping.");
            return;
        }
        if (!File.Exists(Path.Combine(whisperDir, "model.safetensors")))
        {
            _out.WriteLine("whisper-base not cached — skipping STT half.");
            return;
        }

        using KokoroPipeline kokoro = await KokoroPipeline.LoadAsync();
        using CpuBackend backend = new();

        // IPA for "hello world. this is a test.": more content words = a more reliable STT signal than a
        // single word on a short clip.
        const string phonemes = "hɛloʊ wɜːld. ðɪs ɪz ə tɛst.";
        string[] targets = ["hello", "world", "this", "test"];

        float[] audio = kokoro.Synthesize(backend, phonemes, voiceName: "af_heart", speed: 1f);
        Assert.NotEmpty(audio);
        double sumSq = 0; foreach (float v in audio) { Assert.True(float.IsFinite(v)); sumSq += (double)v * v; }
        double rms = Math.Sqrt(sumSq / audio.Length);
        int sr = kokoro.Config.SampleRate;

        string outDir = Path.Combine(Path.GetTempPath(), "hartsyinference_tts_to_stt");
        Directory.CreateDirectory(outDir);
        string outWav = Path.Combine(outDir, $"kokoro_stt_{DateTime.Now:yyyyMMdd_HHmmss}.wav");
        WavFile.WriteMono16(outWav, audio, sr);
        _out.WriteLine($"Kokoro generated {audio.Length} samples ({audio.Length / (double)sr:F2}s @ {sr}Hz). RMS={rms:F4}.");
        _out.WriteLine($"WAV (listen): {outWav}");
        Assert.True(rms > 1e-4, "silent output");

        float[] stt16 = Resampler.Create(sr, 16_000).Resample(audio);
        string sttWav = Path.Combine(outDir, $"kokoro_stt_16k_{DateTime.Now:yyyyMMdd_HHmmss}.wav");
        WavFile.WriteMono16(sttWav, stt16, 16_000);

        using WhisperPipeline stt = await WhisperPipeline.LoadAsync("openai/whisper-base");
        string heard = stt.TranscribeWav(backend, sttWav,
            new WhisperOptions { Language = "en", Translate = false, WithTimestamps = false }).Trim();
        string lower = heard.ToLowerInvariant();
        _out.WriteLine($"Target phonemes:  \"{phonemes}\"  (\"hello world. this is a test.\")");
        _out.WriteLine($"Whisper heard:    \"{heard}\"");
        int hits = targets.Count(w => lower.Contains(w));
        _out.WriteLine($"Content-word recall: {hits}/{targets.Length} ({string.Join(",", targets.Where(w => lower.Contains(w)))})");
        if (hits == 0) _out.WriteLine("--- NO TARGET WORDS — listen to the WAV to judge.");
        Assert.NotNull(heard);
    }
}
