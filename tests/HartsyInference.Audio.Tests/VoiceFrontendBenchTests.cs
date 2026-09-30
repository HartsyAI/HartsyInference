using System.Diagnostics;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.Denoise;
using HartsyInference.Audio.Models.Wake;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Runtime;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>The voice agent's front-end budget: RNNoise over one 20 ms frame of 16 kHz audio plus one Silero VAD
/// chunk, serial on one core, must fit in 2 ms.
///
/// <para>Conservative on purpose: Silero consumes 512 samples, so in the real stream it runs on 0.625 of frames, and
/// here it runs on every one. The work is what the audio thread does per frame, scale conversions included: ±1 in,
/// ×32768 for RNNoise, which keeps upstream's absolute thresholds, and back to ±1 for Silero. The thread pins itself
/// to one CPU and enters <see cref="CpuParallel.EnterInline"/>, so no kernel can fan out. The clip is jfk.wav with
/// white noise mixed in, so RNNoise never meets a frame quiet enough to skip its network.</para>
///
/// <para>Asserts p99 ≤ 2 ms; p50, max, allocations, GC count and process CPU time against wall time are reported
/// beside it, so a pass on a machine that was not quiet, or not single-core, is visible as such. Opt in with
/// <c>HARTSY_VOICE_FRONTEND_BENCH=1</c>; <c>HARTSY_VOICE_FRONTEND_BENCH_CPU</c> picks the core (default the last).
/// Needs the RNNoise and Silero weights under the wake model root, or <c>HARTSYINFERENCE_RNNOISE_WEIGHTS</c> and
/// <c>HARTSYINFERENCE_SILERO_WEIGHTS</c>. Run it alone: any other benchmark or test run on the box moves it.</para></summary>
public sealed class VoiceFrontendBenchTests(ITestOutputHelper log)
{
    private const int Rate = 16_000;
    private const int FrameSamples = Rate / 50;
    // Long enough for tiered compilation to settle every method on the path before timing starts.
    private const int WarmupFrames = 500;
    private const int TimedFrames = 2_000;
    private const double BudgetMs = 2.0;
    private const float Int16Scale = 32768f;

    [Fact]
    [Trait("Category", "Integration")]
    public void SileroPlusRnnoise_PerTwentyMsFrame_OnOneCore_FitsTwoMilliseconds()
    {
        if (Environment.GetEnvironmentVariable("HARTSY_VOICE_FRONTEND_BENCH") != "1")
        {
            log.WriteLine("SKIPPED: set HARTSY_VOICE_FRONTEND_BENCH=1 to run this benchmark");
            return;
        }
        string rnnoisePath = RnnoiseRealSpeechTests.WeightsPath();
        string sileroPath = Environment.GetEnvironmentVariable("HARTSYINFERENCE_SILERO_WEIGHTS")
            ?? Path.Combine(TestPaths.ModelsDir, "audio", "wake", "vad", "silero_vad_16k.safetensors");
        string clipPath = Path.Combine(RepoRoot.Path, "tests", "python-reference", "silerovad_reference", "jfk.wav");
        if (!RealWeightGate.Require(log.WriteLine, rnnoisePath, sileroPath, clipPath)) return;

        float[] audio = LoopWithNoise(WavFile.Read(clipPath).ToMono(), (WarmupFrames + TimedFrames) * FrameSamples);
        int cpu = int.TryParse(Environment.GetEnvironmentVariable("HARTSY_VOICE_FRONTEND_BENCH_CPU"), out int c)
            ? c : Environment.ProcessorCount - 1;

        Result? result = null;
        Exception? failure = null;
        Thread thread = new(() =>
        {
            try
            {
                result = Measure(rnnoisePath, sileroPath, audio, cpu);
            }
            catch (Exception ex)
            {
                failure = ex;
            }
        });
        thread.Start();
        thread.Join();
        if (failure is not null) throw new InvalidOperationException("front-end benchmark thread failed", failure);

        Result r = result!;
        long[] sorted = [.. r.FrameNs];
        Array.Sort(sorted);
        double p50 = Ms(sorted[sorted.Length / 2]), p99 = Ms(sorted[(int)(sorted.Length * 0.99)]);
        double max = Ms(sorted[^1]), mean = Ms((long)sorted.Average());
        log.WriteLine($"{TimedFrames} frames of 20 ms: p50 {p50:F3} ms, p99 {p99:F3} ms, max {max:F3} ms, mean {mean:F3} ms");
        log.WriteLine($"pinned to CPU {cpu}: {(r.Pinned ? "yes" : "no, " + r.PinReason)}; RNNoise network ran on "
            + $"{r.NetworkFrames}/{TimedFrames} frames; {r.AllocatedBytes} bytes allocated, {r.Gen0Collections} gen-0 GCs "
            + $"while timed; process CPU / wall {r.CpuOverWall:F2}; load average {File.ReadAllText("/proc/loadavg").Trim()}");
        foreach ((string label, double ms) in r.Stages)
            log.WriteLine($"  {label}: {ms:F3} ms/frame mean");

        // A frame whose gains saturate can repeat the previous speech probability exactly, so this is a floor
        // against the silence-floor passthrough rather than an exact count.
        Assert.True(r.NetworkFrames >= TimedFrames * 0.95,
            $"RNNoise ran its network on only {r.NetworkFrames}/{TimedFrames} frames");
        Assert.True(p99 <= BudgetMs,
            $"p99 {p99:F3} ms exceeds the {BudgetMs} ms front-end budget (p50 {p50:F3}, max {max:F3})");
    }

    private static Result Measure(string rnnoisePath, string sileroPath, float[] audio, int cpu)
    {
        bool pinned = RealtimeScheduling.TryPinToCpu(cpu, out string pinReason);
        using CpuParallel.InlineScope inline = CpuParallel.EnterInline();
        using CpuBackend backend = new();
        using RnnoiseWeights weights = LoadRnnoise(rnnoisePath);
        using RnnoiseStream denoiser = new(weights, Rate);
        using SileroVad vad = LoadSilero(sileroPath);
        SileroVadStream endpointer = new(vad);

        float[] scaled = new float[FrameSamples];
        float[] denoised = new float[FrameSamples + denoiser.FrameSize];
        float[] window = new float[SileroVad.WindowSamples];
        long[] frameNs = new long[TimedFrames];
        long[] stageTicks = new long[3];
        int networkFrames = 0;
        float lastProbability = float.NaN;
        long allocatedBefore = 0, wallStart = 0;
        int gen0Before = 0;
        TimeSpan cpuStart = TimeSpan.Zero;

        for (int f = 0; f < WarmupFrames + TimedFrames; f++)
        {
            int timed = f - WarmupFrames;
            if (timed == 0)
            {
                cpuStart = Process.GetCurrentProcess().TotalProcessorTime;
                gen0Before = GC.CollectionCount(0);
                wallStart = Stopwatch.GetTimestamp();
                allocatedBefore = GC.GetAllocatedBytesForCurrentThread();
            }
            ReadOnlySpan<float> frame = audio.AsSpan(f * FrameSamples, FrameSamples);
            long t0 = Stopwatch.GetTimestamp();
            for (int i = 0; i < FrameSamples; i++) scaled[i] = frame[i] * Int16Scale;
            int written = denoiser.Process(backend, scaled, denoised);
            long t1 = Stopwatch.GetTimestamp();
            // Newest 512 denoised samples, back at ±1: Silero's scale.
            window.AsSpan(written).CopyTo(window);
            for (int i = 0; i < written; i++) window[SileroVad.WindowSamples - written + i] = denoised[i] / Int16Scale;
            long t2 = Stopwatch.GetTimestamp();
            endpointer.Push(backend, window, out _);
            long t3 = Stopwatch.GetTimestamp();
            if (timed < 0) continue;
            frameNs[timed] = (long)((t3 - t0) * (1e9 / Stopwatch.Frequency));
            stageTicks[0] += t1 - t0;
            stageTicks[1] += t2 - t1;
            stageTicks[2] += t3 - t2;
            if (denoiser.SpeechProbability != lastProbability) networkFrames++;
            lastProbability = denoiser.SpeechProbability;
        }
        long allocated = GC.GetAllocatedBytesForCurrentThread() - allocatedBefore;
        long wallTicks = Stopwatch.GetTimestamp() - wallStart;
        int collections = GC.CollectionCount(0) - gen0Before;
        double cpuOverWall = (Process.GetCurrentProcess().TotalProcessorTime - cpuStart).TotalSeconds
            / (wallTicks / (double)Stopwatch.Frequency);
        (string, double)[] stages =
        [
            ("scale + RNNoise (2 x 10 ms, 16k <-> 48k)", StageMs(stageTicks[0])),
            ("rescale into Silero window", StageMs(stageTicks[1])),
            ("Silero 512-sample chunk + endpointing", StageMs(stageTicks[2])),
        ];
        return new Result(frameNs, stages, networkFrames, pinned, pinReason, allocated, collections, cpuOverWall);
    }

    /// <summary>The clip repeated to <paramref name="length"/> samples with white noise ~30 dB under the speech, so
    /// every frame carries energy.</summary>
    private static float[] LoopWithNoise(float[] clip, int length)
    {
        Random rng = new(99);
        float[] audio = new float[length];
        for (int i = 0; i < length; i++)
            audio[i] = clip[i % clip.Length] + (float)((rng.NextDouble() * 2 - 1) * 0.01);
        return audio;
    }

    private static RnnoiseWeights LoadRnnoise(string path)
    {
        using SafeTensorsLoader loader = new();
        loader.Load(path);
        Dictionary<string, Tensor> tensors = loader.GetAllTensors();
        RnnoiseWeights weights = new();
        weights.Load(tensors);
        foreach (Tensor tensor in tensors.Values) tensor.Dispose();
        return weights;
    }

    private static SileroVad LoadSilero(string path)
    {
        using SafeTensorsLoader loader = new();
        loader.Load(path);
        SileroVad vad = new();
        vad.LoadWeights(loader.GetAllTensors());
        return vad;
    }

    private static double StageMs(long ticks) => ticks * 1000.0 / Stopwatch.Frequency / TimedFrames;

    private static double Ms(long ns) => ns / 1e6;

    private sealed record Result(long[] FrameNs, (string Label, double Ms)[] Stages, int NetworkFrames, bool Pinned,
        string PinReason, long AllocatedBytes, int Gen0Collections, double CpuOverWall);
}
