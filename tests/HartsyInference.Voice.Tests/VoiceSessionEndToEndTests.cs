using System.Diagnostics;
using HartsyInference.Core.Logging;
using HartsyInference.Engine;
using HartsyInference.Tests.Common;
using HartsyInference.Voice.Gpu;
using HartsyInference.Voice.Tests.Fakes;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Voice.Tests;

/// <summary>A call on the RTX 3060 with real speech models: the JFK clip through the audio thread, Whisper small.en and
/// Kokoro af_heart leases on the GPU thread, and a scripted language model (the LLM is not what this measures). Both
/// directions must be intelligible (Whisper recall of JFK and Whisper-verify of the spoken reply, at least 80 % each),
/// the per-turn line must carry every <c>voice.*</c> metric, and the plan's model gates are checked on this card:
/// Whisper small.en at most 350 ms per utterance and Kokoro at most 250 ms per 15-word sentence. The front-end's frame
/// times are logged, not asserted: here the audio thread shares the cores with the speech models' host work, and its
/// serial 2 ms gate is <see cref="TurnEndpointerRealVadTests"/>'s. Run alone, in a quiet window:
/// <c>CUDA_VISIBLE_DEVICES=1 HARTSY_REQUIRE_REAL_WEIGHTS=1 HARTSYINFERENCE_MODELS_DIR=/mnt/model-storage/Models dotnet test tests/HartsyInference.Voice.Tests -c Release --filter "FullyQualifiedName~VoiceSessionEndToEndTests"</c>.</summary>
[Trait("Category", "GpuIntegration")]
[Trait("Category", "RealWeights")]
public sealed class VoiceSessionEndToEndTests
{
    private const string Reply = "The weather tomorrow is clear and mild, with a gentle breeze in the afternoon, so enjoy your walk.";
    private const string FifteenWords = "Thanks for calling, I can see your appointment is booked for Tuesday afternoon at three.";
    private const double SttGateMs = 350;
    private const double KokoroGateMs = 250;

    private static readonly string[] ReplyWords = ["weather", "tomorrow", "clear", "mild", "gentle", "breeze", "afternoon", "enjoy", "walk"];

    private readonly ITestOutputHelper _output;

    public VoiceSessionEndToEndTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task JfkIsHeardAnsweredAndTheReplyIsIntelligible()
    {
        if (!RealWeightGate.Require(_output.WriteLine, GpuVoiceRig.Assets()))
        {
            return;
        }
        using InferenceEngine? engine = GpuVoiceRig.OpenAudioEngine(_output.WriteLine);
        if (engine is null)
        {
            return;
        }
        List<string> log = [];
        Logs.SetLogger((_, message) =>
        {
            lock (log)
            {
                log.Add(message);
            }
        });
        try
        {
            VoiceAgentOptions options = GpuVoiceRig.Options(engine);
            await using VoiceModelSet models = await VoiceModelSet.LoadAsync(engine, options, VoiceAssets.WakeRoot);
            ScriptedTextService text = new ScriptedTextService { DefaultReply = "Okay." }.Reply(Reply);
            await models.WarmAsync(text);
            string warm;
            lock (log)
            {
                warm = Assert.Single(log, message => message.StartsWith("[Voice] Warm-up on ", StringComparison.Ordinal));
            }
            _output.WriteLine(warm);

            double[] kokoroMs = new double[5];
            for (int run = 0; run < kokoroMs.Length; run++)
            {
                kokoroMs[run] = await models.Gpu.RunAsync(VoiceGpuJobKind.Synthesize, () =>
                {
                    long started = Stopwatch.GetTimestamp();
                    models.Synthesize(FifteenWords);
                    return Stopwatch.GetElapsedTime(started).TotalMilliseconds;
                }, CancellationToken.None);
            }
            double kokoroMedian = kokoroMs.Order().ElementAt(kokoroMs.Length / 2);
            _output.WriteLine($"Kokoro 15-word sentence on the GPU thread: median {kokoroMedian:F1} ms ({string.Join(", ", kokoroMs.Select(ms => ms.ToString("F1")))})");

            float[] jfk = VoiceAssets.Jfk16k();
            int turns = TurnEndpointerRealVadTests.Run(jfk, options).Utterances.Count;
            await using VoiceHarness harness = await VoiceHarness.StartAsync(models, text, TimeSpan.FromMilliseconds(20));
            harness.Push(jfk);
            harness.PushSilence(1.5);
            List<VoiceTurnMetrics> metrics = [(await harness.TurnCompletedAsync(1, 120)).Metrics!.Value];
            float[] replyAudio = harness.Reader.Samples;
            for (int turn = 2; turn <= turns; turn++)
            {
                metrics.Add((await harness.TurnCompletedAsync(turn, 120)).Metrics!.Value);
            }
            string heard = string.Join(' ', harness.Events.Where(e => e.Kind == VoiceAgentEventKind.UserTranscript).Select(e => e.Text));
            string replyHeard = await models.Gpu.RunAsync(VoiceGpuJobKind.Transcribe, () => models.Transcribe(replyAudio), CancellationToken.None);
            foreach (VoiceTurnMetrics turn in metrics)
            {
                _output.WriteLine(turn.ToLogLine());
            }
            _output.WriteLine($"heard from the caller ({turns} turns): \"{heard}\"");
            _output.WriteLine($"reply, {replyAudio.Length / 16_000.0:F2} s played, heard back as: \"{replyHeard}\"");

            Assert.True(VoiceAssets.Recall(heard, VoiceAssets.JfkWords) >= 0.8, $"caller recall below 80 % on \"{heard}\"");
            Assert.True(VoiceAssets.Recall(replyHeard, ReplyWords) >= 0.8, $"reply recall below 80 % on \"{replyHeard}\"");
            VoiceTurnMetrics first = metrics[0];
            Assert.NotNull(first.EndpointMs);
            Assert.NotNull(first.LlmTtftMs);
            Assert.NotNull(first.LlmFirstSentenceMs);
            Assert.NotNull(first.TtsFirstChunkMs);
            Assert.NotNull(first.TransportMs);
            Assert.NotNull(first.TotalMs);
            Assert.All(metrics, turn => Assert.True(turn.SttMs <= SttGateMs, $"Whisper small.en took {turn.SttMs:F1} ms for turn {turn.TurnId}."));
            Assert.True(kokoroMedian <= KokoroGateMs, $"Kokoro took {kokoroMedian:F1} ms for a 15-word sentence.");
            string line;
            lock (log)
            {
                line = Assert.Single(log, message => message.StartsWith("[Voice] turn 1 (utterance):", StringComparison.Ordinal));
            }
            foreach (string key in VoiceTurnMetrics.MetricKeys)
            {
                Assert.Contains(" " + key + "=", line, StringComparison.Ordinal);
            }
        }
        finally
        {
            Logs.SetLogger(null!);
        }
    }
}
