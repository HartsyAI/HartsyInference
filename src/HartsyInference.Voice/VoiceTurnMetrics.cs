using System.Globalization;
using System.Text;

namespace HartsyInference.Voice;

/// <summary>Timings of one turn, named after the voice latency budget (<c>voice.*</c>). A stage that did not run in
/// this turn is null: a DTMF or spoken-prompt turn has no recognition, a turn with no reply has no synthesis.</summary>
/// <remarks>Monotonic-clock times except <see cref="EndpointMs"/> and <see cref="UtteranceMs"/>, which are counted in
/// inbound samples, so they hold when audio arrives faster than real time. <see cref="TotalMs"/> therefore adds the
/// sample-clock hangover to the wall time from the endpoint to the first reply audio queued for
/// <see cref="VoiceAgentSession.ReadOutbound"/>. The gateway-side stages (jitter buffer, RTP pacing) are measured by
/// the phone gateway, not here.</remarks>
public readonly record struct VoiceTurnMetrics
{
    /// <summary>The turn.</summary>
    public int TurnId { get; init; }

    /// <summary>What started it.</summary>
    public VoiceTurnKind Kind { get; init; }

    /// <summary>Length of the caller's utterance, VAD padding included.</summary>
    public double? UtteranceMs { get; init; }

    /// <summary><c>voice.endpoint.ms</c>: from the end of the caller's speech to the endpoint decision.</summary>
    public double? EndpointMs { get; init; }

    /// <summary><c>voice.frontend.ms</c> median bucket: denoise plus VAD per 20 ms frame since the previous endpoint.</summary>
    public double? FrontendP50Ms { get; init; }

    /// <summary><c>voice.frontend.ms</c> 99th-percentile bucket.</summary>
    public double? FrontendP99Ms { get; init; }

    /// <summary><c>voice.frontend.ms</c> worst frame.</summary>
    public double? FrontendMaxMs { get; init; }

    /// <summary><c>voice.stt.ms</c>: recognition, GPU queue wait included.</summary>
    public double? SttMs { get; init; }

    /// <summary><c>voice.llm.ttft_ms</c>: from the request to the first reply text.</summary>
    public double? LlmTtftMs { get; init; }

    /// <summary><c>voice.llm.first_sentence_ms</c>: from the request to the first sentence ready for synthesis.</summary>
    public double? LlmFirstSentenceMs { get; init; }

    /// <summary><c>voice.tts.first_chunk_ms</c>: from the first sentence to its audio, GPU queue wait included.</summary>
    public double? TtsFirstChunkMs { get; init; }

    /// <summary><c>voice.transport.ms</c>: from the first audio to its resampled samples queued for playback.</summary>
    public double? TransportMs { get; init; }

    /// <summary><c>voice.turn.total_ms</c>: from the end of the caller's speech (or the turn's start for DTMF and
    /// spoken prompts) to the first reply audio queued for playback.</summary>
    public double? TotalMs { get; init; }

    /// <summary>Tool calls the model made.</summary>
    public int ToolCalls { get; init; }

    /// <summary>Whether the caller barged in.</summary>
    public bool Interrupted { get; init; }

    /// <summary><c>voice.bargein.stop_ms</c>: from the barge-in decision to the reader dropping the reply's queued audio.</summary>
    public double? BargeInStopMs { get; init; }

    /// <summary>Inbound samples dropped since the session started because the audio thread fell behind.</summary>
    public long InboundDroppedSamples { get; init; }

    /// <summary>Utterances not answered since the session started (<see cref="VoiceAgentEventKind.UtteranceDiscarded"/>).</summary>
    public int DiscardedUtterances { get; init; }

    /// <summary>The metric keys <see cref="ToLogLine"/> always writes, in order.</summary>
    internal static IReadOnlyList<string> MetricKeys { get; } =
    [
        "voice.frontend.ms.p50", "voice.frontend.ms.p99", "voice.frontend.ms.max", "voice.endpoint.ms", "voice.stt.ms",
        "voice.llm.ttft_ms", "voice.llm.first_sentence_ms", "voice.tts.first_chunk_ms", "voice.transport.ms",
        "voice.turn.total_ms", "voice.bargein.stop_ms",
    ];

    /// <summary>The single <c>[Voice] turn N: …</c> line logged when the turn ends; absent stages print as <c>-</c>.</summary>
    public string ToLogLine()
    {
        StringBuilder line = new(512);
        line.Append("[Voice] turn ").Append(TurnId.ToString(CultureInfo.InvariantCulture))
            .Append(" (").Append(Kind.ToString().ToLowerInvariant()).Append("):");
        double?[] values =
        [
            FrontendP50Ms, FrontendP99Ms, FrontendMaxMs, EndpointMs, SttMs, LlmTtftMs, LlmFirstSentenceMs, TtsFirstChunkMs,
            TransportMs, TotalMs, BargeInStopMs,
        ];
        for (int i = 0; i < values.Length; i++)
        {
            line.Append(' ').Append(MetricKeys[i]).Append('=')
                .Append(values[i] is double value ? value.ToString("0.00", CultureInfo.InvariantCulture) : "-");
        }
        line.Append(" utterance_ms=").Append(UtteranceMs is double ms ? ms.ToString("0", CultureInfo.InvariantCulture) : "-")
            .Append(" tools=").Append(ToolCalls.ToString(CultureInfo.InvariantCulture))
            .Append(" interrupted=").Append(Interrupted ? "true" : "false")
            .Append(" inbound_dropped=").Append(InboundDroppedSamples.ToString(CultureInfo.InvariantCulture))
            .Append(" discarded_utterances=").Append(DiscardedUtterances.ToString(CultureInfo.InvariantCulture));
        return line.ToString();
    }
}
