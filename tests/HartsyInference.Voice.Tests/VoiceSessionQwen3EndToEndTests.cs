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
/// <see cref="TextRequest.Device"/> = <c>cuda:0</c>, the host's real 7-tool set installed (<see cref="BuildRealToolSet"/>
/// — the same names/descriptions/schemas <c>VoiceHostTools</c> offers; this project has no <c>InternalsVisibleTo</c>
/// into <c>HartsyInference.VoiceHost</c>, and the handlers here are safe no-ops rather than VoiceHostTools' own
/// warm-up-only throw-if-invoked ones, since a real multi-turn conversation can plausibly call any of them). Runs
/// <see cref="Turns"/> turns of the same caller utterance back to back (conversation history grows each turn, so
/// later turns template a longer prompt, exactly as <c>MaxHistoryTokens</c>/<c>MaxReplyTokens</c> and the default
/// sampling the gate specifies) and asserts <c>voice.llm.ttft_ms</c> ≤ 150, <c>voice.llm.first_sentence_ms</c> ≤ 200
/// (tool-call turns exempted, matching the gate's own wording) and <c>voice.turn.total_ms</c> ≤ 1300 for EVERY turn —
/// this is the gate-verification run for perf/llm-voice-prefix-reuse's prefix-KV reuse, not just a measurement
/// report. Also logs the LLM's resident VRAM (<see cref="VramProbe"/>) before the first turn and after the last, to
/// show the footprint is bounded and does not grow with the conversation. Two independent timelines remain for
/// root-causing any remaining gap between <c>llm.ttft_ms</c> and <c>llm.first_sentence_ms</c>:
/// <see cref="RecordingDiagnostics"/> (raw per-token events from inside the engine, via
/// <see cref="EngineOptions.Diagnostics"/>) and <see cref="TimestampingTextService"/> (the chunks the session
/// actually sees, after the tool-call stream filter). Turn 1 keeps the original correctness checks (no literal
/// <c>&lt;think&gt;</c> text, caller/reply recall). The 4090 is shared with SwarmUI, so beyond the Slow category this
/// runs only with <c>HARTSY_VOICE_LLM_GPU=1</c>, set by the orchestrator when it grants the card:
/// <c>HARTSY_VOICE_LLM_GPU=1 HARTSY_REQUIRE_REAL_WEIGHTS=1 HARTSYINFERENCE_MODELS_DIR=/mnt/model-storage/Models dotnet test tests/HartsyInference.Voice.Tests -c Release --filter "FullyQualifiedName~VoiceSessionQwen3EndToEndTests"</c>.</summary>
[Trait("Category", "GpuIntegration")]
[Trait("Category", "RealWeights")]
[Trait("Category", "Slow")]
public sealed class VoiceSessionQwen3EndToEndTests
{
    private const string GrantEnvVar = "HARTSY_VOICE_LLM_GPU";

    /// <summary>User turns driven back to back on one session: the gate's own "10-turn conversation" wording.</summary>
    private const int Turns = 10;

    private const double TtftBudgetMs = 150;
    private const double FirstSentenceBudgetMs = 200;
    private const double TotalBudgetMs = 1300;
    private const long VramTargetBytes = 6L << 30;

    private readonly ITestOutputHelper _output;

    public VoiceSessionQwen3EndToEndTests(ITestOutputHelper output) => _output = output;

    /// <summary>The host's real 7 tools (<c>VoiceHostTools.Names</c>'s own set and schemas), with safe handlers that
    /// return a canned confirmation instead of touching a gateway or ending the call — unlike
    /// <c>VoiceModelSet.WarmAsync</c>'s throw-if-invoked warm-up definitions, a real multi-turn conversation can
    /// plausibly have the model call any of these for real, and must not crash the run if it does.</summary>
    private static ToolRegistry BuildRealToolSet()
    {
        const string noArgs = """{"type":"object","properties":{},"additionalProperties":false}""";
        ToolRegistry registry = new();
        registry.Add("hangup", "End the call.", noArgs, (_, _) => Task.FromResult("""{"ok":true}"""));
        registry.Add("send_dtmf", "Press keys on the phone keypad, for example to answer a phone menu.",
            """{"type":"object","properties":{"digits":{"type":"string","description":"Keys to press, from 0-9 * # A-D, for example \"1\" or \"123#\"."}},"required":["digits"],"additionalProperties":false}""",
            (_, _) => Task.FromResult("""{"ok":true}"""));
        registry.Add("transfer", "Transfer the caller to another phone number. Only numbers the phone system allows can be reached.",
            """{"type":"object","properties":{"target":{"type":"string","description":"The phone number to transfer the caller to."}},"required":["target"],"additionalProperties":false}""",
            (_, _) => Task.FromResult("""{"ok":true}"""));
        registry.Add("hold", "Put the caller on hold.", noArgs, (_, _) => Task.FromResult("""{"ok":true}"""));
        registry.Add("unhold", "Take the caller off hold.", noArgs, (_, _) => Task.FromResult("""{"ok":true}"""));
        registry.Add("play_prompt", "Play a short recorded prompt to the caller.",
            """{"type":"object","properties":{"name":{"type":"string","enum":["one-moment","goodbye"],"description":"Which prompt to play."}},"required":["name"],"additionalProperties":false}""",
            (_, _) => Task.FromResult("""{"ok":true}"""));
        registry.Add("get_time", () => DateTime.Now.ToString("h:mm tt"), "Tells the current local time.");
        return registry;
    }

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
        ToolRegistry tools = BuildRealToolSet();
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
        long? vramAfterWarm = VramProbe.UsedBytes(engine.Text, "cuda:0");
        _output.WriteLine($"VRAM (cuda:0) after warm-up: {VramProbe.Describe(vramAfterWarm)} (target <= {VramTargetBytes / (1024.0 * 1024):F0} MB)");

        float[] jfk = VoiceAssets.Jfk16k();
        TurnEndpointerRealVadTests.Utterance first = TurnEndpointerRealVadTests.Run(jfk, options).Utterances[0];
        float[] slice = jfk[..(int)Math.Min(jfk.Length, first.Start + first.Length)];
        await using VoiceHarness harness = await VoiceHarness.StartAsync(models, timestamped, TimeSpan.FromMilliseconds(2), tools);
        // StartAsync's prefix-cache priming request is fire-and-forget; give it a moment to land before reading
        // VRAM, purely so this log line attributes its cost correctly (production never waits on it).
        await Task.Delay(500).ConfigureAwait(false);
        long? vramAfterPriming = VramProbe.UsedBytes(engine.Text, "cuda:0");
        _output.WriteLine($"VRAM (cuda:0) after the session's priming request: {VramProbe.Describe(vramAfterPriming)}"
            + $" (+{Delta(vramAfterWarm, vramAfterPriming)} over warm-up: the retained sequence's own KV capacity, a real resident allocation, not pool slack)");

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
            _output.WriteLine($"  VRAM (cuda:0) after turn {turnId}: {VramProbe.Describe(VramProbe.UsedBytes(engine.Text, "cuda:0"))}");

            // The gate itself: every turn, not just a report. Tool-call turns are exempt from first_sentence,
            // matching the gate's own "on non-tool turns" wording (a tool round adds a full extra model
            // invocation before any reply text streams, which TTFT and total_ms still cover).
            AssertBudget("llm.ttft_ms", turnId, turn.LlmTtftMs, TtftBudgetMs);
            if (turn.ToolCalls == 0)
            {
                AssertBudget("llm.first_sentence_ms", turnId, turn.LlmFirstSentenceMs, FirstSentenceBudgetMs);
            }
            AssertBudget("turn.total_ms", turnId, turn.TotalMs, TotalBudgetMs);

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

        long? vramAfterLast = VramProbe.UsedBytes(engine.Text, "cuda:0");
        _output.WriteLine($"VRAM (cuda:0) after turn {Turns}: {VramProbe.Describe(vramAfterLast)}"
            + $" (warm-up {VramProbe.Describe(vramAfterWarm)} -> priming {VramProbe.Describe(vramAfterPriming)} -> turn {Turns} {VramProbe.Describe(vramAfterLast)})");
        if (vramAfterPriming is { } primedBytes && vramAfterLast is { } lastBytes)
        {
            // The gate this asserts: once the call is warm (model loaded, prefix primed), 10 real turns must not
            // creep the footprint up further -- the growth from warm-up to THIS point is the retained sequence's
            // own KV capacity (a deliberate, one-time, reported-below allocation, not something turns add to).
            // A couple hundred MB of activation/workspace slack sized for the longest prompt a turn's decode loop
            // happens to hit is expected; anything beyond that is the regression this assertion exists to catch.
            const long growthToleranceBytes = 256L << 20;
            Assert.True(lastBytes <= primedBytes + growthToleranceBytes,
                $"VRAM grew by {(lastBytes - primedBytes) / (1024.0 * 1024):F0} MB over {Turns} turns after priming ({VramProbe.Describe(primedBytes)} -> {VramProbe.Describe(lastBytes)}).");
            if (lastBytes > VramTargetBytes)
            {
                _output.WriteLine($"NOTE: VRAM steady-state ({VramProbe.Describe(lastBytes)}) is above the {VramTargetBytes / (1024.0 * 1024):F0} MB target by "
                    + $"{(lastBytes - VramTargetBytes) / (1024.0 * 1024):F0} MB (down from ~13.8 GB before CacheWeightCasts=false, ~7.27 GB before"
                    + " PreloadRedundantWeightSplits=false). Two known, unused levers to close the rest: (1) a smaller VoiceAgentSession.PrefixCacheCapacityHint"
                    + " trades some mid-call reallocation risk for less resident KV; (2) vram.kvF16 halves the retained cache's bytes but is a numerics change"
                    + " this PR does not make unilaterally. Not asserted here -- reported for a decision.");
            }
            else
            {
                _output.WriteLine($"VRAM steady-state ({VramProbe.Describe(lastBytes)}) meets the {VramTargetBytes / (1024.0 * 1024):F0} MB target "
                    + $"(margin {(VramTargetBytes - lastBytes) / (1024.0 * 1024):F0} MB) -- down from ~13.8 GB before CacheWeightCasts=false and ~7.27 GB"
                    + " before PreloadRedundantWeightSplits=false; see both default's doc comments on VoiceAgentOptions.");
            }
        }

        IReadOnlyList<IReadOnlyList<RecordingDiagnostics.Entry>> byRequest = diagnostics.ByRequest();
        _output.WriteLine($"raw per-token diagnostics, {byRequest.Count} request(s) recorded (expect warm-up + priming + {Turns} turns = {Turns + 2}):");
        for (int i = 0; i < byRequest.Count; i++)
        {
            string label = i switch { 0 => "warm-up", 1 => "priming", _ => $"turn {i - 1}" };
            _output.WriteLine($"  {label}: {RecordingDiagnostics.Describe(byRequest[i])}");
        }

        _output.WriteLine($"turn | llm.ttft_ms (budget {TtftBudgetMs:F0}) | llm.first_sentence_ms (budget {FirstSentenceBudgetMs:F0}) | turn.total_ms (budget {TotalBudgetMs:F0}) | tools");
        foreach (VoiceTurnMetrics turn in metrics)
        {
            _output.WriteLine($"  {turn.TurnId} | {Fmt(turn.LlmTtftMs)} | {Fmt(turn.LlmFirstSentenceMs)} | {Fmt(turn.TotalMs)} | {turn.ToolCalls}"
                + $" | ttft {Verdict(turn.LlmTtftMs, TtftBudgetMs)}, first_sentence {(turn.ToolCalls > 0 ? "n/a (tool turn)" : Verdict(turn.LlmFirstSentenceMs, FirstSentenceBudgetMs))}, total {Verdict(turn.TotalMs, TotalBudgetMs)}");
        }
    }

    /// <summary>Fails with every number in the message (not just pass/fail) when <paramref name="actualMs"/> is null
    /// or exceeds <paramref name="budgetMs"/>.</summary>
    private static void AssertBudget(string metric, int turnId, double? actualMs, double budgetMs)
    {
        Assert.True(actualMs is { } value && value <= budgetMs,
            $"turn {turnId}: {metric} = {Fmt(actualMs)} ms, budget {budgetMs:F0} ms.");
    }

    private static string Delta(long? before, long? after) =>
        before is { } b && after is { } a ? $"{(a - b) / (1024.0 * 1024):F0} MB" : "?";

    private static string Fmt(double? ms) => ms is { } value ? value.ToString("F1") : "-";

    private static string Verdict(double? ms, double budget) => ms is { } value ? (value <= budget ? "MEETS" : $"MISSES by {value - budget:F1} ms") : "n/a";

    private static string Quote(string? text) => "\"" + (text ?? "").Replace("\n", "\\n").Replace("\r", "") + "\"";
}
