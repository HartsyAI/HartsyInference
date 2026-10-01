using System.Diagnostics;
using HartsyInference.Core.Logging;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;
using HartsyInference.Tools;
using HartsyInference.Voice.Gpu;
using HartsyInference.Voice.Tests.Fakes;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Voice.Tests;

/// <summary>The production split with a real language model: the audio engine on the RTX 3060 (engine <c>cuda:1</c> with
/// every card visible) and Qwen3-4B Q4_K_M, asked for by its catalog id <c>qwen3</c>, on the RTX 4090 through
/// <see cref="TextRequest.Device"/> = <c>cuda:0</c>, tool calling installed. Runs <see cref="Turns"/> turns of the same
/// caller utterance back to back (conversation history grows each turn, so later turns template a longer prompt) and
/// reports every <c>voice.llm.*</c> metric per turn, plus two independent timelines for root-causing a gap between
/// <c>llm.ttft_ms</c> and <c>llm.first_sentence_ms</c>: <see cref="RecordingDiagnostics"/> (raw per-token events from
/// inside the engine, via <see cref="EngineOptions.Diagnostics"/>) and <see cref="TimestampingTextService"/> (the
/// chunks the session actually sees, after the tool-call stream filter). Turn 1 keeps the original correctness checks
/// (no literal <c>&lt;think&gt;</c> text, caller/reply recall); turns beyond it are reported, not gated — this is the
/// first time the multi-turn shape has been measured, so a hard budget assertion would gate on noise, not a known-
/// stable number. The 4090 is shared with SwarmUI, so beyond the Slow category this runs only with
/// <c>HARTSY_VOICE_LLM_GPU=1</c>, set by the orchestrator when it grants the card:
/// <c>HARTSY_VOICE_LLM_GPU=1 HARTSY_REQUIRE_REAL_WEIGHTS=1 HARTSYINFERENCE_MODELS_DIR=/mnt/model-storage/Models dotnet test tests/HartsyInference.Voice.Tests -c Release --filter "FullyQualifiedName~VoiceSessionQwen3EndToEndTests"</c>.</summary>
[Trait("Category", "GpuIntegration")]
[Trait("Category", "RealWeights")]
[Trait("Category", "Slow")]
public sealed class VoiceSessionQwen3EndToEndTests
{
    private const string GrantEnvVar = "HARTSY_VOICE_LLM_GPU";

    /// <summary>User turns driven back to back on one session: enough to tell a one-time cold cost from a recurring
    /// per-turn one (the plan's "Decisions delegated to the orchestrator" §5 asked for 3-4).</summary>
    private const int Turns = 4;

    private readonly ITestOutputHelper _output;

    public VoiceSessionQwen3EndToEndTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task ACallerIsAnsweredByQwen3OnTheOtherCard()
    {
        if (Environment.GetEnvironmentVariable(GrantEnvVar) != "1")
        {
            _output.WriteLine($"SKIPPED: set {GrantEnvVar}=1 once the 4090 is granted for this run.");
            return;
        }
        string checkpoint = TestPaths.Llm.Qwen3_4BQ4KM;
        if (!RealWeightGate.Require(_output.WriteLine, [.. GpuVoiceRig.Assets(), checkpoint]))
        {
            return;
        }
        ToolRegistry tools = new ToolRegistry().Add("get_time", () => DateTime.Now.ToString("h:mm tt"), "Tells the current local time.");
        RecordingDiagnostics diagnostics = new();
        EngineOptions engineOptions = new() { Diagnostics = diagnostics };
        ToolCalling.Install(engineOptions);
        using InferenceEngine? engine = GpuVoiceRig.OpenAudioEngine(_output.WriteLine, ordinal: 1, engineOptions);
        if (engine is null)
        {
            return;
        }
        VoiceAgentOptions options = GpuVoiceRig.Options(engine, llmDevice: "cuda:0") with { LlmModel = "qwen3" };
        ModelSpec llm = VoiceModelSet.ResolveLlm(options);
        _output.WriteLine($"qwen3 resolves to {llm.LocalPath}");
        Assert.Equal(Path.GetFullPath(checkpoint), Path.GetFullPath(llm.LocalPath ?? ""));
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
            await RunAsync(engine, options, tools, diagnostics, log);
        }
        finally
        {
            Logs.SetLogger(null!);
        }
    }

    private async Task RunAsync(InferenceEngine engine, VoiceAgentOptions options, ToolRegistry tools, RecordingDiagnostics diagnostics, List<string> log)
    {
        await using VoiceModelSet models = await VoiceModelSet.LoadAsync(engine, options, VoiceAssets.WakeRoot);
        // The session's own text service, wrapped so the chunks it actually streams (post tool-call filter) are
        // timestamped for comparison against RecordingDiagnostics' raw per-token view from inside the engine.
        TimestampingTextService timestamped = new(engine.Text);
        long warmStarted = Stopwatch.GetTimestamp();
        await models.WarmAsync(timestamped, tools.Definitions);
        _output.WriteLine($"warm-up (speech on the 3060 and {VoiceModelSet.WarmToolMaxTokens} tool-aware tokens on the 4090, in parallel): "
            + $"{Stopwatch.GetElapsedTime(warmStarted).TotalMilliseconds:F0} ms");
        lock (log)
        {
            _output.WriteLine(Assert.Single(log, message => message.StartsWith("[Voice] Warm-up on ", StringComparison.Ordinal)));
        }
        _output.WriteLine($"plan stats after warm-up: {CudaPlanStats.Describe(engine.Text, "cuda:0") ?? "(unavailable)"}");

        float[] jfk = VoiceAssets.Jfk16k();
        TurnEndpointerRealVadTests.Utterance first = TurnEndpointerRealVadTests.Run(jfk, options).Utterances[0];
        float[] slice = jfk[..(int)Math.Min(jfk.Length, first.Start + first.Length)];
        await using VoiceHarness harness = await VoiceHarness.StartAsync(models, timestamped, TimeSpan.FromMilliseconds(2), tools);

        List<VoiceTurnMetrics> metrics = [];
        int repliedSamplesSoFar = 0;
        for (int turnId = 1; turnId <= Turns; turnId++)
        {
            // Cleared per turn so the recorded chunks for turns 1-2 below belong to that turn alone; warm-up's own
            // chunks (recorded through the same decorator) are discarded by the first clear.
            timestamped.Clear();
            harness.Push(slice);
            harness.PushSilence(1.5);
            VoiceTurnMetrics turn = (await harness.TurnCompletedAsync(turnId, 180)).Metrics!.Value;
            metrics.Add(turn);
            _output.WriteLine(turn.ToLogLine());
            _output.WriteLine($"  plan stats after turn {turnId}: {CudaPlanStats.Describe(engine.Text, "cuda:0") ?? "(unavailable)"}");

            bool sawThink = false;
            if (turnId <= 2)
            {
                foreach (TimestampingTextService.Entry chunkEvent in timestamped.Events)
                {
                    sawThink |= chunkEvent.Text?.Contains("<think", StringComparison.OrdinalIgnoreCase) == true;
                    _output.WriteLine($"    t+{chunkEvent.ElapsedMs:F1} ms {chunkEvent.Kind}"
                        + (chunkEvent.Stop is { } stop ? $" stop={stop}" : "")
                        + (string.IsNullOrEmpty(chunkEvent.Text) ? "" : $" text={Quote(chunkEvent.Text)}"));
                }
                if (sawThink)
                {
                    _output.WriteLine("  WARNING: a chunk contained literal \"<think\" even though EnableThinking=false.");
                }
            }

            float[] allReplies = harness.Reader.Samples;
            float[] thisReply = allReplies[repliedSamplesSoFar..];
            repliedSamplesSoFar = allReplies.Length;
            if (turnId == 1)
            {
                string caller = harness.Session.Transcript.Single(e => e.Role == TextRole.User && e.TurnId == turnId).Text;
                string reply = harness.Session.Transcript.Single(e => e.Role == TextRole.Assistant && e.TurnId == turnId).Text;
                string replyHeard = await models.Gpu.RunAsync(VoiceGpuJobKind.Transcribe, () => models.Transcribe(thisReply), CancellationToken.None);
                string[] replyWords = [.. VoiceAssets.Words(reply).Distinct()];
                _output.WriteLine($"  caller: \"{caller}\" (recall of \"fellow Americans\": {VoiceAssets.Recall(caller, ["fellow", "americans"]):P0})");
                _output.WriteLine($"  Qwen3: \"{reply}\"; heard back: \"{replyHeard}\" (recall of the reply's words: {VoiceAssets.Recall(replyHeard, replyWords):P0})");

                Assert.DoesNotContain("<think>", reply, StringComparison.Ordinal);
                Assert.NotNull(turn.LlmTtftMs);
                Assert.NotNull(turn.TotalMs);
                Assert.True(VoiceAssets.Words(replyHeard).Count() >= 3, $"the spoken reply was not intelligible: \"{replyHeard}\"");
            }
        }

        Assert.All(metrics, m => Assert.NotNull(m.TotalMs));

        IReadOnlyList<IReadOnlyList<RecordingDiagnostics.Entry>> byRequest = diagnostics.ByRequest();
        _output.WriteLine($"raw per-token diagnostics, {byRequest.Count} request(s) recorded (expect warm-up + {Turns} turns = {Turns + 1}):");
        for (int i = 0; i < byRequest.Count; i++)
        {
            string label = i == 0 ? "warm-up" : $"turn {i}";
            _output.WriteLine($"  {label}: {RecordingDiagnostics.Describe(byRequest[i])}");
        }

        _output.WriteLine("turn | llm.ttft_ms (budget 150) | llm.first_sentence_ms (budget 200) | turn.total_ms (budget 1300)");
        foreach (VoiceTurnMetrics turn in metrics)
        {
            _output.WriteLine($"  {turn.TurnId} | {Fmt(turn.LlmTtftMs)} | {Fmt(turn.LlmFirstSentenceMs)} | {Fmt(turn.TotalMs)}"
                + $" | ttft {Verdict(turn.LlmTtftMs, 150)}, first_sentence {Verdict(turn.LlmFirstSentenceMs, 200)}, total {Verdict(turn.TotalMs, 1300)}");
        }
    }

    private static string Fmt(double? ms) => ms is { } value ? value.ToString("F1") : "-";

    private static string Verdict(double? ms, double budget) => ms is { } value ? (value <= budget ? "MEETS" : $"MISSES by {value - budget:F1} ms") : "n/a";

    private static string Quote(string? text) => "\"" + (text ?? "").Replace("\n", "\\n").Replace("\r", "") + "\"";
}
