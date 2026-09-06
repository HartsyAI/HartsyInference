using System.Diagnostics;
using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Transcription wall-clock, the counterpart to <see cref="TtsBenchTests"/>.
///
/// <para>Speech-to-text sits on the critical path of a voice turn just as synthesis does — a wake word is not
/// answered until the utterance has been transcribed — but it had no benchmark, so kernel work could speed up
/// synthesis and leave transcription untouched without anyone noticing. Same shape as the TTS benches: model
/// load excluded, one warmup, best of three, CPU unless <c>BENCH_CUDA=1</c>, and a clean skip when the weights
/// are not cached rather than a failure on a machine that never downloaded them.</para>
///
/// <para>Reported as seconds and as a real-time factor against the clip's own duration. Whisper pads every clip
/// to 30 s before encoding, so its cost is nearly independent of how long the audio actually is — which is
/// exactly why the RTF of a short command utterance is the number worth watching.</para></summary>
public sealed class SttBenchTests
{
    private readonly ITestOutputHelper _out;
    public SttBenchTests(ITestOutputHelper o) => _out = o;

    private static IBackend MakeBackend(out string name)
    {
        if (Environment.GetEnvironmentVariable("BENCH_CUDA") == "1")
        {
            string ptx = Environment.GetEnvironmentVariable("BENCH_PTX") ?? throw new InvalidOperationException("BENCH_CUDA=1 needs BENCH_PTX");
            name = "CUDA";
            return new HartsyInference.Cuda.CudaBackend(0, ptx);
        }
        name = "CPU";
        return new CpuBackend();
    }

    private static (double best, double rtf) Time(Func<string> run, double audioSeconds, int runs = 3)
    {
        run(); // warmup: first call pays JIT and any lazy allocation
        double best = double.MaxValue;
        for (int i = 0; i < runs; i++)
        {
            Stopwatch sw = Stopwatch.StartNew();
            run();
            sw.Stop();
            best = Math.Min(best, sw.Elapsed.TotalSeconds);
        }
        return (best, best / audioSeconds);
    }

    /// <summary>A short command utterance — the shape a wake-word satellite actually sends.</summary>
    private static float[] SpokenCommandClip(out double seconds, int sampleRate = 16_000)
    {
        // Synthetic rather than a fixture: the decoded text is irrelevant here (this times the forward pass,
        // it does not check accuracy), and a committed audio file would make an untagged unit-lane benchmark
        // depend on gitignored reference data.
        int n = sampleRate * 3;
        float[] audio = new float[n];
        uint state = 0x9E3779B9;
        for (int i = 0; i < n; i++)
        {
            state = state * 1664525u + 1013904223u;
            float noise = ((state >> 8) & 0xFFFF) / 32768f - 1f;
            // A couple of formant-ish tones with an envelope, enough to give the encoder real structure.
            float t = i / (float)sampleRate;
            float env = MathF.Min(1f, MathF.Sin(MathF.PI * t / 3f) * 2f);
            audio[i] = env * (0.35f * MathF.Sin(2 * MathF.PI * 220f * t)
                            + 0.25f * MathF.Sin(2 * MathF.PI * 740f * t)
                            + 0.05f * noise);
        }
        seconds = n / (double)sampleRate;
        return audio;
    }

    [Fact]
    public async Task Bench_WhisperBase()
    {
        string dir = Path.Combine(AudioModelCache.CacheRoot, "stt", "openai--whisper-base");
        if (!Directory.Exists(dir) || Directory.GetFiles(dir, "*.safetensors").Length == 0)
        {
            _out.WriteLine($"SKIP Bench_WhisperBase: no cached weights at {dir}");
            return;
        }
        using IBackend backend = MakeBackend(out string device);
        using WhisperPipeline pipeline = await WhisperPipeline.LoadAsync("openai/whisper-base");
        float[] audio = SpokenCommandClip(out double seconds);

        (double best, double rtf) = Time(
            () => pipeline.TranscribeAudio(backend, audio, 16_000, new WhisperOptions { Language = "en" }),
            seconds);
        _out.WriteLine($"OURS Whisper-base ({device}): {seconds:0.00}s clip | gen {best:0.000}s | RTF {rtf:0.000}");
    }
}
