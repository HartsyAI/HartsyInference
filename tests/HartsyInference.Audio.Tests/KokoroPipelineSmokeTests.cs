using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Cpu;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>End-to-end smoke tests for <see cref="KokoroPipeline"/>. These tests verify
/// that the full pipeline (tokenizer → PLBERT → TextEncoder → ProsodyPredictor →
/// IStftNetDecoder) loads all 548 Kokoro tensors cleanly and runs forward without
/// crashing on a real IPA phoneme string.
///
/// <para><b>Status note</b>: the decoder forward pass is currently the
/// <see cref="HartsyInference.Audio.Models.Kokoro.KokoroIStftNetDecoder"/> placeholder —
/// it synthesizes F0-modulated audio so the prosody pipeline can be audibly QA'd, but
/// the full HiFi-GAN+iSTFT generator is staged for a follow-up. The tests therefore
/// only assert structural facts (non-empty audio, finite samples, weight load success);
/// quality assertions will follow once the generator is wired.</para></summary>
public sealed class KokoroPipelineSmokeTests
{
    private readonly ITestOutputHelper _out;
    public KokoroPipelineSmokeTests(ITestOutputHelper output) => _out = output;

    private static bool CacheReady()
    {
        // Weights load from the single-file repack repo; voice packs still come from hexgrad.
        string weights = AudioModelCache.GetRepoDirectory("Hartsy/kokoro-82m-safetensors");
        string voices = AudioModelCache.GetRepoDirectory("hexgrad/Kokoro-82M");
        return File.Exists(Path.Combine(weights, "kokoro-82m.safetensors"))
            && File.Exists(Path.Combine(weights, "config.json"))
            && File.Exists(Path.Combine(voices, "voices", "af_heart.bin"));
    }

    [Fact]
    public async Task LoadsAndProducesAudio_ForSimpleIpaPhrase()
    {
        if (!CacheReady())
        {
            _out.WriteLine("Kokoro cache missing — skipping.");
            return;
        }

        using KokoroPipeline kokoro = await KokoroPipeline.LoadAsync();
        using CpuBackend backend = new();

        // "hello" in IPA from the Kokoro vocab: h ɛ l oʊ → /h/ /ɛ/ /l/ /oʊ/.
        // Using the printed vocab: h=50, ɛ=86, l=54, o=57, ʊ=135.
        // For now keep it simple — Kokoro takes any visible-token IPA string.
        string phonemes = "hɛloʊ wɜːld";     // "hello world"-ish for the smoke test
        _out.WriteLine($"Synthesizing: \"{phonemes}\"");

        float[] audio = kokoro.Synthesize(backend, phonemes, voiceName: "af_heart", speed: 1f);

        Assert.NotEmpty(audio);
        _out.WriteLine($"Audio length: {audio.Length} samples ({audio.Length / (double)kokoro.Config.SampleRate:F2} s).");

        float maxAbs = 0f;
        int nFinite = 0;
        for (int i = 0; i < audio.Length; i++)
        {
            float a = audio[i];
            if (float.IsNaN(a) || float.IsInfinity(a))
                Assert.Fail($"audio[{i}] is non-finite: {a}");
            nFinite++;
            float abs = MathF.Abs(a);
            if (abs > maxAbs) maxAbs = abs;
        }
        Assert.Equal(audio.Length, nFinite);
        _out.WriteLine($"Audio max-abs: {maxAbs:F4} ({nFinite}/{audio.Length} finite samples).");

        // Write the WAV out so the user can listen and QA the predictor's prosody curves.
        string outDir = Path.Combine(Path.GetTempPath(), "hartsyinference_tts_to_stt");
        Directory.CreateDirectory(outDir);
        string outPath = Path.Combine(outDir, $"kokoro_smoke_{DateTime.Now:yyyyMMdd_HHmmss}.wav");
        WavFile.WriteMono16(outPath, audio, kokoro.Config.SampleRate);
        _out.WriteLine($"Wrote Kokoro smoke audio: {outPath}");
    }

    [Fact]
    public async Task DurationsScaleWithSpeedKnob()
    {
        if (!CacheReady())
        {
            _out.WriteLine("Kokoro cache missing — skipping.");
            return;
        }

        using KokoroPipeline kokoro = await KokoroPipeline.LoadAsync();
        using CpuBackend backend = new();

        string phonemes = "hɛloʊ";
        float[] normal = kokoro.Synthesize(backend, phonemes, "af_heart", speed: 1f);
        float[] fast = kokoro.Synthesize(backend, phonemes, "af_heart", speed: 2f);
        _out.WriteLine($"normal: {normal.Length} samples, fast (speed=2): {fast.Length} samples.");
        // Fast should produce strictly fewer samples — even with placeholder synthesis the
        // duration count drives total length, so this verifies the speed knob propagates.
        Assert.True(fast.Length < normal.Length,
            $"fast ({fast.Length}) should be < normal ({normal.Length}).");
    }
}
