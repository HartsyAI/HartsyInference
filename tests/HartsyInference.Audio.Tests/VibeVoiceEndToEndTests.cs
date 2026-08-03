using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.VibeVoice;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Cpu;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>End-to-end VibeVoice TTS smoke tests. Drives the full pipeline (acoustic VAE
/// + semantic VAE + Qwen2.5-1.5B LM + diffusion head + multi-speaker processor) from a
/// short script + reference audio → 24 kHz mono WAV.
///
/// <para><b>Cost note:</b> each AR LM step on CPU is ~10 s at f32, and each emitted
/// <c>speech_diffusion</c> token triggers 20 forward passes of the diffusion head plus an
/// acoustic VAE decode. These tests are <b>gated behind an env var</b>
/// (<c>HARTSYINFERENCE_RUN_VIBEVOICE_E2E=1</c>) so the default regression suite stays
/// memory-bounded. The point is to prove the pipeline runs end-to-end and produces finite
/// audio — quality validation against the Python reference is the follow-up.</para>
///
/// <para>Tests skip cleanly when the VibeVoice-1.5B + test-clip caches are missing.</para></summary>
public sealed class VibeVoiceEndToEndTests
{
    private const int VibeVoiceSampleRate = 24_000;

    private readonly ITestOutputHelper _out;
    public VibeVoiceEndToEndTests(ITestOutputHelper output) => _out = output;

    private static bool CacheReady()
    {
        string vvRepo = AudioModelCache.GetRepoDirectory("vibevoice/VibeVoice-1.5B", "tts");
        string jfk = Path.Combine(AudioModelCache.CacheRoot, "test-clips", "jfk.wav");
        return File.Exists(Path.Combine(vvRepo, "model-00001-of-00003.safetensors"))
            && File.Exists(Path.Combine(vvRepo, "model-00002-of-00003.safetensors"))
            && File.Exists(Path.Combine(vvRepo, "model-00003-of-00003.safetensors"))
            && File.Exists(jfk);
    }

    private bool Skip()
    {
        if (!CacheReady()) { _out.WriteLine("VibeVoice cache missing — skipping."); return true; }
        if (Environment.GetEnvironmentVariable("HARTSYINFERENCE_RUN_VIBEVOICE_E2E") != "1")
        {
            _out.WriteLine("Set HARTSYINFERENCE_RUN_VIBEVOICE_E2E=1 to enable. Skipping.");
            return true;
        }
        return false;
    }

    [Fact]
    public async Task PipelineLoads_AllSubmodulesReady()
    {
        if (!CacheReady()) { _out.WriteLine("VibeVoice cache missing — skipping."); return; }
        using VibeVoicePipeline pipeline = await VibeVoicePipeline.LoadAsync();
        Assert.NotNull(pipeline);
        Assert.Equal("vibevoice", pipeline.Config.ModelType);
        _out.WriteLine("VibeVoicePipeline loaded successfully (all submodules + scaling/bias scalars).");
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
    }

    [Fact]
    [Trait("Category", "Slow")]
    public void Processor_BuildsMultiSpeakerPrompt()
    {
        if (!CacheReady()) { _out.WriteLine("VibeVoice cache missing — skipping."); return; }
        using VibeVoiceTokenizer tokenizer = new();
        VibeVoiceProcessor processor = new(tokenizer);
        string jfk = Path.Combine(AudioModelCache.CacheRoot, "test-clips", "jfk.wav");
        VibeVoiceProcessor.PreparedPrompt prep = processor.Prepare(
            lines: ["Speaker 0: Hello there."],
            voiceAudioPaths: [jfk]);

        // Basic sanity: prompt is non-empty, mask is the same length, voices array has 1 entry.
        Assert.NotEmpty(prep.TokenIds);
        Assert.Equal(prep.TokenIds.Length, prep.SpeechInputMask.Length);
        Assert.Single(prep.Voices);

        int trueMaskCount = prep.SpeechInputMask.Count(b => b);
        Assert.Equal(prep.Voices[0].LatentCount, trueMaskCount);
        _out.WriteLine($"Prompt: {prep.TokenIds.Length} tokens, {trueMaskCount} speech-input-mask slots, voice latent count = {prep.Voices[0].LatentCount}.");
        _out.WriteLine($"First / last 10 token IDs: [{string.Join(", ", prep.TokenIds.Take(10))} … {string.Join(", ", prep.TokenIds.TakeLast(10))}].");
    }

    [Fact]
    [Trait("Category", "Slow")]
    public async Task PrefillOnly_RunsWithoutCrash()
    {
        if (Skip()) return;
        _out.WriteLine("Starting prefill-only test (no AR loop).");
        using VibeVoicePipeline pipeline = await VibeVoicePipeline.LoadAsync();
        using CpuBackend backend = new();
        string jfk = Path.Combine(AudioModelCache.CacheRoot, "test-clips", "jfk.wav");

        DateTime t0 = DateTime.Now;
        // maxNewTokens=0 → pipeline runs prefill, doesn't enter the AR loop.
        float[] audio = pipeline.Synthesize(backend, ["Speaker 0: Hi."], [jfk], maxNewTokens: 0);
        TimeSpan dt = DateTime.Now - t0;
        _out.WriteLine($"Prefill-only completed in {dt.TotalSeconds:F1} s, audio length = {audio.Length}.");
        // With maxNewTokens=0 we expect zero audio chunks emitted.
        Assert.Empty(audio);
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
    }

    [Fact]
    [Trait("Category", "Slow")]
    public async Task SynthesizeShortScript_ProducesFiniteAudio()
    {
        if (Skip()) return;

        using VibeVoicePipeline pipeline = await VibeVoicePipeline.LoadAsync();
        using CpuBackend backend = new();

        string jfkPath = Path.Combine(AudioModelCache.CacheRoot, "test-clips", "jfk.wav");
        string[] lines = ["Speaker 0: Hello there."];
        string[] voices = [jfkPath];

        DateTime t0 = DateTime.Now;
        float[] audio = pipeline.Synthesize(backend, lines, voices, maxNewTokens: 4);
        TimeSpan dt = DateTime.Now - t0;

        _out.WriteLine($"VibeVoice synthesized {audio.Length} samples in {dt.TotalSeconds:F1} s.");
        Assert.NotEmpty(audio);
        for (int i = 0; i < audio.Length; i++)
        {
            float v = audio[i];
            if (float.IsNaN(v) || float.IsInfinity(v))
                Assert.Fail($"audio[{i}] is non-finite: {v}");
        }

        double sumSq = 0d;
        for (int i = 0; i < audio.Length; i++) sumSq += (double)audio[i] * audio[i];
        double rms = Math.Sqrt(sumSq / audio.Length);
        _out.WriteLine($"Output RMS = {rms:F4}.");
        Assert.True(rms > 1e-6, $"audio is essentially silent (RMS {rms}).");

        string outDir = Path.Combine(Path.GetTempPath(), "hartsyinference_tts_to_stt");
        Directory.CreateDirectory(outDir);
        string outPath = Path.Combine(outDir, $"vibevoice_smoke_{DateTime.Now:yyyyMMdd_HHmmss}.wav");
        WavFile.WriteMono16(outPath, audio, VibeVoiceSampleRate);
        _out.WriteLine($"Wrote VibeVoice smoke audio: {outPath}");
    }
}
