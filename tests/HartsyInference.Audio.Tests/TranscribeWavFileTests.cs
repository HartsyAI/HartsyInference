using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Cpu;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Reusable STT harness for the audio-model grind: transcribes any WAV with Whisper and reports
/// content-word recall against an expected set. Lets a model's existing (RMS-only) e2e test produce a WAV,
/// then this proves whether it's actually intelligible speech — no per-model STT wiring needed.
///
/// <para>Gated on <c>TRANSCRIBE_WAV</c> (path to the WAV). Optional <c>TRANSCRIBE_TARGETS</c> (comma-separated
/// content words to score recall against). Runs on CPU with whisper-base.</para></summary>
public sealed class TranscribeWavFileTests
{
    private readonly ITestOutputHelper _out;
    public TranscribeWavFileTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public async Task Transcribe()
    {
        string? wav = Environment.GetEnvironmentVariable("TRANSCRIBE_WAV");
        if (string.IsNullOrEmpty(wav) || !File.Exists(wav)) { _out.WriteLine("TRANSCRIBE_WAV not set/found — skipping."); return; }
        string whisperDir = AudioModelCache.GetRepoDirectory("openai/whisper-base");
        if (!File.Exists(Path.Combine(whisperDir, "model.safetensors"))) { _out.WriteLine("whisper-base not cached — skipping."); return; }

        // Ensure 16 kHz mono for Whisper.
        WavFile.DecodedAudio a = WavFile.Read(wav);
        float[] mono = a.Channels[0];
        float[] in16 = a.SampleRate == 16_000 ? mono : Resampler.Create(a.SampleRate, 16_000).Resample(mono);
        string sttWav = wav.EndsWith(".16k.wav") ? wav : wav + ".16k.wav";
        WavFile.WriteMono16(sttWav, in16, 16_000);

        using CpuBackend backend = new();
        using WhisperPipeline stt = await WhisperPipeline.LoadAsync("openai/whisper-base");
        string heard = stt.TranscribeWav(backend, sttWav,
            new WhisperOptions { Language = "en", Translate = false, WithTimestamps = false }).Trim();
        _out.WriteLine($"WAV: {wav} ({mono.Length / (double)a.SampleRate:F2}s @ {a.SampleRate}Hz)");
        _out.WriteLine($"Whisper heard: \"{heard}\"");

        string? targetsEnv = Environment.GetEnvironmentVariable("TRANSCRIBE_TARGETS");
        if (!string.IsNullOrEmpty(targetsEnv))
        {
            string lower = heard.ToLowerInvariant();
            string[] targets = targetsEnv.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            int hits = targets.Count(w => lower.Contains(w.ToLowerInvariant()));
            _out.WriteLine($"Content-word recall: {hits}/{targets.Length} ({string.Join(",", targets.Where(w => lower.Contains(w.ToLowerInvariant())))})");
        }
        Assert.NotNull(heard);
    }
}
