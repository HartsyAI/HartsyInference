using System.Diagnostics;
using HartsyInference.Audio.Models.Denoise;
using HartsyInference.Audio.Models.Wake;
using HartsyInference.Core.Numerics;
using HartsyInference.Cpu;
using HartsyInference.Engine.Audio.Wake;
using HartsyInference.Tests.Common;
using HartsyInference.Voice.Audio;
using HartsyInference.Voice.Tests.Fakes;
using HartsyInference.Voice.Turns;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Voice.Tests;

/// <summary>The audio thread's front-end on real weights (Silero, and RNNoise when its weights are installed) over the
/// JFK clip, run the way the audio thread runs it: CPU backend, inline scope, one 20 ms frame at a time. The utterances
/// must cover the clip's speech. Silero alone stays inside <see cref="SileroFrameBudgetMs"/>; RNNoise (int8) + Silero is
/// checked against the plan's redefined quiet gate, p50/p99 over <see cref="LatencyRepeats"/> back-to-back passes — the
/// 4-thread contention variant of that gate is the Audio package's own int8 bench, not this test. The managed bytes per
/// frame are reported (the CPU kernels, not the session, allocate their dispatch closures). Weights come from the
/// models root's <c>audio/wake</c> (set <c>HARTSYINFERENCE_MODELS_DIR</c>); missing ones skip unless
/// <c>HARTSY_REQUIRE_REAL_WEIGHTS=1</c>.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class TurnEndpointerRealVadTests
{
    private const double SileroFrameBudgetMs = 2.0;

    /// <summary>Quiet p50 budget for RNNoise (int8) + Silero per 20 ms frame, back-to-back (the plan's redefined
    /// front-end gate; the live-20ms-cadence and 4-thread-contention variants are measured in the Audio package).</summary>
    private const double RnnoiseP50BudgetMs = 3.0;

    /// <summary>Quiet p99 budget for the same pair.</summary>
    private const double RnnoiseP99BudgetMs = 5.0;

    /// <summary>Back-to-back passes over the padded clip the measured run repeats, so the JFK clip's ~625 frames give
    /// a steadier p99 than a single pass would. Only the first pass's utterances are checked against the clip.</summary>
    private const int LatencyRepeats = 3;

    private readonly ITestOutputHelper _output;

    public TurnEndpointerRealVadTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void SileroSplitsJfkIntoUtterancesThatCoverItsSpeech()
    {
        if (!RealWeightGate.Require(_output.WriteLine, VoiceAssets.SileroWeights, VoiceAssets.Jfk))
        {
            return;
        }
        // Denoise off explicitly: this is the Silero-only baseline, and VoiceAgentOptions.Denoise now defaults on.
        Result result = Run(VoiceAssets.Jfk16k(), new VoiceAgentOptions { Denoise = false });
        Report("Silero", result);
        AssertCoversJfk(result);
        Assert.True(result.MeanMs <= SileroFrameBudgetMs, $"Silero took {result.MeanMs:F3} ms per 20 ms frame on average.");
    }

    [Fact]
    public void RnnoiseAndSileroSplitJfkWithinTheFrameBudget()
    {
        if (!RealWeightGate.Require(_output.WriteLine, VoiceAssets.SileroWeights, VoiceAssets.RnnoiseWeights, VoiceAssets.RnnoiseInt8Tables, VoiceAssets.Jfk))
        {
            return;
        }
        Result result = Run(VoiceAssets.Jfk16k(), new VoiceAgentOptions { Denoise = true });
        Report("RNNoise (int8) + Silero", result);
        AssertCoversJfk(result);
        Assert.Equal(RnnoisePrecision.Int8, result.DenoisePrecision);
        Assert.True(result.P50Ms <= RnnoiseP50BudgetMs, $"RNNoise + Silero p50 was {result.P50Ms:F3} ms per 20 ms frame.");
        Assert.True(result.P99Ms <= RnnoiseP99BudgetMs, $"RNNoise + Silero p99 was {result.P99Ms:F3} ms per 20 ms frame.");
    }

    /// <summary>The utterances a session with <paramref name="options"/> would close on <paramref name="audio"/> plus 1.5 s
    /// of trailing silence: a warm pass, then <see cref="LatencyRepeats"/> measured passes (each after a
    /// <see cref="VoiceAudioFrontend.Reset"/>) pooled into one back-to-back latency sample; only the first measured pass's
    /// utterances are returned, since repeating the clip would double-count endpoints.</summary>
    internal static Result Run(float[] audio, VoiceAgentOptions options)
    {
        using WakeModelSet wake = VoiceModelSet.LoadFrontEnd(VoiceAssets.WakeRoot, options, out Func<IVadModel> createVad,
            out Func<RnnoiseStream>? createDenoiser);
        using CpuBackend cpu = new();
        using CpuParallel.InlineScope inline = CpuParallel.EnterInline();
        using VoiceAudioFrontend frontend = new(cpu, createVad(), createDenoiser?.Invoke(), new VoiceTurnSignals(), options);
        int frames = (audio.Length + 24_000) / VoiceAudioFrontend.FrameSamples;
        float[] padded = new float[frames * VoiceAudioFrontend.FrameSamples];
        audio.AsSpan(0, Math.Min(audio.Length, padded.Length)).CopyTo(padded);

        Drive(frontend, padded, null, null);
        List<Utterance> utterances = [];
        double[] frameMs = new double[frames * LatencyRepeats];
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int rep = 0; rep < LatencyRepeats; rep++)
        {
            frontend.Reset();
            double[] pass = new double[frames];
            Drive(frontend, padded, rep == 0 ? utterances : null, pass);
            Array.Copy(pass, 0, frameMs, rep * frames, frames);
        }
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Array.Sort(frameMs);
        int total = frameMs.Length;
        return new Result(utterances, frameMs.Average(), frameMs[total / 2], frameMs[(int)(total * 0.99)], frameMs[^1],
            allocated / (double)total, frontend.Denoising, wake.DenoisePrecision);
    }

    private static void Drive(VoiceAudioFrontend frontend, float[] audio, List<Utterance>? utterances, double[]? frameMs)
    {
        for (int f = 0; f < audio.Length / VoiceAudioFrontend.FrameSamples; f++)
        {
            long started = Stopwatch.GetTimestamp();
            VoiceFrameEvents events = frontend.ProcessFrame(audio.AsSpan(f * VoiceAudioFrontend.FrameSamples, VoiceAudioFrontend.FrameSamples));
            if (frameMs is not null)
            {
                frameMs[f] = Stopwatch.GetElapsedTime(started).TotalMilliseconds;
            }
            if (utterances is not null && (events & VoiceFrameEvents.Endpoint) != 0)
            {
                utterances.Add(new Utterance(frontend.UtteranceStartSample, frontend.UtteranceSamples, frontend.HangoverSamples));
            }
        }
    }

    private void Report(string label, Result result)
    {
        foreach (Utterance utterance in result.Utterances)
        {
            _output.WriteLine($"{label}: utterance {utterance.Start / 16_000.0:F2}-{(utterance.Start + utterance.Length) / 16_000.0:F2} s, "
                + $"hangover {utterance.Hangover / 16.0:F0} ms");
        }
        _output.WriteLine($"{label}: per 20 ms frame mean {result.MeanMs:F3} ms, p50 {result.P50Ms:F3} ms, p99 {result.P99Ms:F3} ms, "
            + $"max {result.MaxMs:F3} ms; {result.BytesPerFrame:F1} managed bytes per frame; denoising {result.Denoised} ({result.DenoisePrecision})");
    }

    private static void AssertCoversJfk(Result result)
    {
        Assert.NotEmpty(result.Utterances);
        Utterance first = result.Utterances[0];
        Utterance last = result.Utterances[^1];
        Assert.True(first.Start < 1.5 * 16_000, $"the first utterance starts at {first.Start / 16_000.0:F2} s.");
        Assert.True(last.Start + last.Length > 8 * 16_000, $"the last utterance ends at {(last.Start + last.Length) / 16_000.0:F2} s.");
        Assert.True(result.Utterances.Sum(u => u.Length) > 6 * 16_000, "the utterances hold less than 6 s of the 11 s clip.");
    }

    internal readonly record struct Utterance(long Start, int Length, long Hangover);

    internal sealed record Result(List<Utterance> Utterances, double MeanMs, double P50Ms, double P99Ms, double MaxMs, double BytesPerFrame,
        bool Denoised, RnnoisePrecision? DenoisePrecision);
}
