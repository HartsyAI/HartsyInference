using System.Diagnostics;
using System.Text;
using HartsyInference.Core.Backends;
using HartsyInference.Cuda;
using HartsyInference.Engine;
using HartsyInference.Engine.Diagnostics;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Tools.Tests;

/// <summary>perf/llm-short-reply-decode investigation step 2+3: the REAL path a voice turn takes —
/// <see cref="ToolLoop.RunAsync"/> over <see cref="ITextService.StreamAsync"/> — with the session's own prompt
/// shape (system instructions + a tool + ~250 tokens of call history, same as <c>VoiceAgentSession.BuildRequest</c>
/// on <c>feat/voice-package</c>, not reproduced verbatim here since the Voice package is not on this branch) and
/// its own default sampling (Temperature=0.7, TopP=0.95, Greedy=false — <see cref="TextRequest"/>'s own defaults;
/// <c>VoiceAgentSession.Turns.cs</c>'s <c>BuildRequest</c> never overrides them).
///
/// <para>Records two independent timelines per run: <see cref="InferenceDiagnosticKind.TokenGenerated"/> events
/// (decode thread, inside the slot lock — fired at the TOP of <c>TextService.RunText</c>'s <c>onToken</c>, BEFORE
/// the incremental detokenizer / output parser / <see cref="HartsyInference.Tools.ToolCallStreamFilter"/> /
/// channel write run) via a trivial <see cref="IInferenceDiagnostics"/>, and the consumer's own <c>await foreach</c>
/// arrival time for each visible chunk. decode-thread gap ≈ consumer gap means the cost is already paid before the
/// channel hand-off (synchronous, on the decode thread itself — sampler or onToken-chain cost); a consumer gap
/// measurably larger than the decode-thread gap is <c>TextStreamPump</c>/channel/async-iterator transport cost.</para></summary>
[Trait("Category", "GpuIntegration")]
[Trait("Category", "RealWeights")]
public sealed class ShortReplyStreamTransportBenchTests
{
    private const string GateEnvVar = "HARTSY_SHORTREPLY_BENCH";
    private const string OutEnvVar = "HARTSY_SHORTREPLY_BENCH_OUT";
    private const int Repeats = 3;
    private const int MaxReplyTokens = 48;

    private readonly ITestOutputHelper _out;

    public ShortReplyStreamTransportBenchTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public async Task DecodeThreadVsConsumerGaps_RealisticVoiceTurn_SessionDefaultSampling()
    {
        if (Environment.GetEnvironmentVariable(GateEnvVar) != "1")
        {
            _out.WriteLine($"SKIPPED: set {GateEnvVar}=1 to run the short-reply stream-transport bench.");
            return;
        }
        if (!CudaContext.IsAvailable())
        {
            _out.WriteLine("SKIPPED: CUDA unavailable.");
            return;
        }
        string checkpoint = TestPaths.Llm.Qwen3_4BQ4KM;
        if (!RealWeightGate.Require(_out.WriteLine, checkpoint)) return;
        ModelSpec spec = ModelResolver.Resolve("qwen3", checkpoint, Modality.Text);

        ToolDefinition hangUp = ToolSchema.FromDelegate("hang_up", static () => "call ended", "Ends the current phone call immediately.");
        ToolRegistry registry = new ToolRegistry().Add("hang_up", () => "call ended", "Ends the current phone call immediately.");

        EngineOptions options = new();
        ToolCalling.Install(options);
        TimestampRecorder recorder = new();
        options.Diagnostics = recorder;
        using InferenceEngine engine = new("cuda", options);

        List<TextMessage> messages = BuildVoiceTurnHistory();
        int approxTokens = messages.Sum(m => Math.Max(1, (m.Content.Length + 3) / 4));
        _out.WriteLine($"history: {messages.Count} messages, ~{approxTokens} tokens (4-chars/token estimate; same heuristic "
            + "ITextService.CountTokens falls back to when no slot is loaded yet).");

        TextRequest request = new()
        {
            Messages = messages,
            Tools = [hangUp],
            EnableThinking = false,
            MaxTokens = MaxReplyTokens,
            // Deliberately NOT setting Greedy/Temperature/TopP: this is the voice session's own default shape
            // (VoiceAgentSession.Turns.cs's BuildRequest leaves them unset too) -> Temperature=0.7, TopP=0.95,
            // Greedy=false (TextRequest.cs's own defaults; see TextService.BuildSampling).
        };

        StringBuilder table = new StringBuilder();
        table.AppendLine("### Stream-transport bench — Qwen3-4B Q4_K_M (4090, `ToolLoop.RunAsync` -> `StreamAsync`, session-default sampling)");
        table.AppendLine();
        table.AppendLine("| run | tokens | decode-thread gap median ms | consumer gap median ms | consumer tok/s | decode-thread tok/s | reply |");
        table.AppendLine("|---|---:|---:|---:|---:|---:|---|");

        for (int run = 1; run <= Repeats; run++)
        {
            recorder.Events.Clear();
            List<double> consumerTicks = new(MaxReplyTokens + 4);
            Stopwatch clock = Stopwatch.StartNew();
            StringBuilder visible = new();
            // maxRounds: 1 — this measures ONE generation's per-token timeline; a plain spoken reply is what the
            // prompt below asks for, but if the model calls hang_up anyway, round 1 == maxRounds just ends the
            // loop on the completed call without dispatching it (see ToolLoop's round-limit doc) — either way
            // exactly one coherant decode sequence is what gets timestamped.
            await foreach (TextChunk chunk in ToolLoop.RunAsync(engine.Text, spec, request, registry, maxRounds: 1))
            {
                if (chunk.Kind is TextChunkKind.Chunk or TextChunkKind.NativeToolCall or TextChunkKind.ToolCallDelta)
                {
                    consumerTicks.Add(clock.Elapsed.TotalSeconds);
                }
                if (chunk.Kind == TextChunkKind.Chunk) visible.Append(chunk.Text);
            }
            List<double> decodeTicks = recorder.Events
                .Where(e => e.Kind == InferenceDiagnosticKind.TokenGenerated)
                .Select(e => e.Timestamp / (double)Stopwatch.Frequency)
                .ToList();

            List<double> decodeGaps = Gaps(decodeTicks);
            List<double> consumerGaps = Gaps(consumerTicks);
            double decodeRate = decodeGaps.Count > 0 ? decodeGaps.Count / decodeGaps.Sum() : 0.0;
            double consumerRate = consumerGaps.Count > 0 ? consumerGaps.Count / consumerGaps.Sum() : 0.0;
            _out.WriteLine($"run {run}: reply=\"{Flatten(visible.ToString())}\" decodeTokens={decodeTicks.Count} consumerChunks={consumerTicks.Count}");
            _out.WriteLine($"run {run}: decode-thread gaps(ms)=[{string.Join(", ", decodeGaps.Select(g => (g * 1000).ToString("F1")))}]");
            _out.WriteLine($"run {run}: consumer     gaps(ms)=[{string.Join(", ", consumerGaps.Select(g => (g * 1000).ToString("F1")))}]");
            table.AppendLine($"| {run} | {decodeTicks.Count} | {Median(decodeGaps) * 1000:F1} | {Median(consumerGaps) * 1000:F1} | {consumerRate:F1} | {decodeRate:F1} | {Flatten(visible.ToString())} |");
        }
        Emit(table);
    }

    /// <summary>~250 tokens of system instructions + a short multi-turn call history, matching
    /// VoiceAgentSession's "system prompt, tools, a short history" shape; the final user turn asks for a one-
    /// sentence spoken answer (not a tool call) so this measures the short-CONTENT-reply path the task reports as
    /// slow, not a tool-call turn.</summary>
    private static List<TextMessage> BuildVoiceTurnHistory() =>
    [
        new TextMessage
        {
            Role = TextRole.System,
            Content = "You are a phone agent handling live customer calls. Answer briefly and plainly, in short "
                + "spoken sentences a caller can follow aloud. You can end the call with the hang_up tool once "
                + "everything is resolved. Company policy: address changes are allowed until the parcel is "
                + "scanned at the depot, refunds take three to five business days, and any promise of a callback "
                + "must be logged before the call ends.",
        },
        new TextMessage
        {
            Role = TextRole.User,
            Content = "Hi, I'm calling about an order I placed last week. I want to change the delivery address "
                + "to my office instead of my home.",
        },
        new TextMessage
        {
            Role = TextRole.Assistant,
            Content = "Sure, I can help with that. Let me pull up your order. Can you confirm the name on the "
                + "account?",
        },
        new TextMessage { Role = TextRole.User, Content = "It's under Jordan Reyes." },
        new TextMessage
        {
            Role = TextRole.Assistant,
            Content = "Thanks, Jordan. I see one open order and one completed return on the account. The open "
                + "order hasn't reached the depot yet, so the address change is still possible. I've updated the "
                + "delivery address to your office. Is there anything else?",
        },
        new TextMessage
        {
            Role = TextRole.User,
            Content = "Yes — can the courier call ahead before dropping it off? And I never got a callback that "
                + "was promised to me on my return refund last month, which is frustrating since it's been almost "
                + "three weeks now and I was told five business days at most.",
        },
        new TextMessage
        {
            Role = TextRole.Assistant,
            Content = "I'm sorry about that missed callback — I've logged a note so it gets followed up today. "
                + "I've also flagged the delivery for a call-ahead. Your return refund shows as issued on our "
                + "side within the three-to-five-day window; it may just be your bank's posting delay.",
        },
        new TextMessage
        {
            Role = TextRole.User,
            Content = "Okay. Can you just summarize everything we agreed to do, in one short sentence?",
        },
    ];

    private static List<double> Gaps(List<double> timestamps)
    {
        List<double> gaps = new(Math.Max(0, timestamps.Count - 1));
        for (int i = 1; i < timestamps.Count; i++) gaps.Add(timestamps[i] - timestamps[i - 1]);
        return gaps;
    }

    private static double Median(List<double> values)
    {
        if (values.Count == 0) return 0.0;
        List<double> sorted = values.OrderBy(v => v).ToList();
        int mid = sorted.Count / 2;
        return sorted.Count % 2 == 0 ? (sorted[mid - 1] + sorted[mid]) / 2.0 : sorted[mid];
    }

    private static string Flatten(string text) => text.Trim().Replace('\n', ' ');

    private void Emit(StringBuilder table)
    {
        string text = table.ToString();
        _out.WriteLine(text);
        string? outPath = Environment.GetEnvironmentVariable(OutEnvVar);
        if (!string.IsNullOrEmpty(outPath))
        {
            File.AppendAllText(outPath, text + Environment.NewLine);
            _out.WriteLine($"appended to {outPath}");
        }
    }

    /// <summary>Allocation-light recorder: pre-sized list, no locks (one generation at a time in this test).</summary>
    private sealed class TimestampRecorder : IInferenceDiagnostics
    {
        public readonly List<InferenceDiagnosticEvent> Events = new(256);
        public void OnEvent(in InferenceDiagnosticEvent diagnostic) => Events.Add(diagnostic);
        public void OnBackendReady(long requestId, IBackend backend) { }
    }
}
