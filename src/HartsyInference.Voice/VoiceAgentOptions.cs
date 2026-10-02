using HartsyInference.Engine.Requests;

namespace HartsyInference.Voice;

/// <summary>Models, devices and conversation settings for a <see cref="VoiceModelSet"/> and the
/// <see cref="VoiceAgentSession"/>s it serves.</summary>
/// <remarks>The model and device fields (<see cref="SttModel"/>, <see cref="TtsModel"/>, <see cref="AudioDevice"/>,
/// <see cref="Denoise"/>, <see cref="CpuThreadCap"/>) are applied when the model set loads; a session built on that
/// set must carry the same values. Every other field is per session.</remarks>
public sealed record VoiceAgentOptions
{
    /// <summary>The instruction every conversation starts with when none is given.</summary>
    public const string DefaultSystemPrompt =
        "You are a helpful assistant speaking with a caller on the phone. Answer in one or two short sentences of "
        + "plain spoken English, with no lists, markdown, emoji or links. Use a tool when the caller asks for something "
        + "a tool can do.";

    /// <summary>Language model handed to the <c>ITextService</c>: a catalog id, local path or repository id.</summary>
    public string LlmModel { get; init; } = "qwen3";

    /// <summary>Device key the language model runs on (<see cref="TextRequest.Device"/>).</summary>
    public string LlmDevice { get; init; } = "cuda:0";

    /// <summary>Device the speech recognizer and synthesizer run on; the model set's GPU thread is its only user.</summary>
    public string AudioDevice { get; init; } = "cuda:1";

    /// <summary>Speech recognizer. English-only Whisper checkpoints need the full repository id.</summary>
    public string SttModel { get; init; } = "whisper:openai/whisper-small.en";

    /// <summary>Speech synthesizer and its voice.</summary>
    public string TtsModel { get; init; } = "kokoro:af_heart";

    /// <summary>Rate of the audio <see cref="VoiceAgentSession.ReadOutbound(Span{float})"/> returns.</summary>
    public int OutboundSampleRate { get; init; } = 16_000;

    /// <summary>Silence after speech that ends the caller's turn.</summary>
    public int EndOfTurnSilenceMs { get; init; } = 700;

    /// <summary>Longest utterance before the turn is cut and transcribed anyway.</summary>
    public int MaxUtteranceMs { get; init; } = 15_000;

    /// <summary>Whether caller speech over the agent's reply stops the reply.</summary>
    public bool BargeInEnabled { get; init; } = true;

    /// <summary>Speech probability a chunk must reach to count toward a barge-in.</summary>
    public float BargeInProbability { get; init; } = 0.6f;

    /// <summary>Consecutive speech needed to barge in.</summary>
    public int BargeInMinMs { get; init; } = 200;

    /// <summary>Time after the agent starts speaking during which barge-in is not armed, so the reply's own onset
    /// and the tail of the caller's turn cannot interrupt it.</summary>
    public int BargeInHoldoffMs { get; init; } = 300;

    /// <summary>Run RNNoise ahead of the VAD, at int8 (the voice front-end gate was only met at that precision; the
    /// wake stack's own loader always stays Float). Loading fails when either the F32 weights or the int8 tables
    /// beside them are absent; it never falls back to raw audio. Default true: adds 640 samples (40 ms) of
    /// algorithmic delay ahead of endpointing and barge-in detection.</summary>
    public bool Denoise { get; init; } = true;

    /// <summary>First message of every conversation.</summary>
    public string SystemPrompt { get; init; } = DefaultSystemPrompt;

    /// <summary>Language-model invocations per turn, tool rounds included.</summary>
    public int MaxToolRoundsPerTurn { get; init; } = 4;

    /// <summary>Token budget of the conversation sent with each request; the oldest turns are dropped first.</summary>
    public int MaxHistoryTokens { get; init; } = 3_000;

    /// <summary>Token budget of one reply.</summary>
    public int MaxReplyTokens { get; init; } = 200;

    /// <summary>Shortest first sentence synthesized on its own, so a reply starts playing early.</summary>
    public int FirstSentenceMinChars { get; init; } = 12;

    /// <summary>Longest sentence synthesized whole; longer ones are split at clauses.</summary>
    public int MaxSentenceChars { get; init; } = 180;

    /// <summary>When positive, caps the engine's CPU kernel threads (<c>numerics.cpuThreads</c>) while the model set
    /// is loaded.</summary>
    public int CpuThreadCap { get; init; }

    /// <summary>Partial transcripts while the caller is still speaking. Not supported yet: it needs a streaming
    /// recognizer, so <c>true</c> is rejected.</summary>
    public bool PartialTranscripts { get; init; }

    /// <summary>Reuses the language model's KV cache across a call's turns (<c>TextRequest.PrefixCacheKey</c>):
    /// each turn prefills only what diverges from the retained conversation instead of the whole growing history,
    /// and the system+tools prefix is pre-filled once at <see cref="VoiceAgentSession.StartAsync"/> so turn 1 is
    /// warm too. Default true; false restores the original per-turn-from-scratch behavior (e.g. to isolate whether
    /// a regression is reuse-related).</summary>
    public bool EnablePrefixCache { get; init; } = true;

    /// <summary>Whether the language model's device backend caches a dequantized copy of its quantized weights.
    /// Default false: measured on Qwen3-4B-Q4_K_M/4090, "on" costs ~7.3 GB resident once warm (the dominant
    /// share of the model's VRAM footprint) for a prefill that is a fixed ~50 ms faster; "off" keeps weights
    /// compressed with a transient per-GEMM dequant, trading that fixed ~50 ms of every prefill call (prompt
    /// length does not change it — decode's quantized GEMV path is unaffected either way, and so is tokens/sec)
    /// for staying off the model's own memory. True restores the backend's own default (on) — e.g. to isolate
    /// whether a regression is residency-related.</summary>
    public bool CacheWeightCasts { get; init; }

    /// <summary>Whether the language model's initial weight upload includes the load-time-fused Q/K/V and
    /// gate/up projections' original split tensors alongside their fused replacements (see
    /// <c>TextRequest.PreloadRedundantWeightSplits</c>). Default false: measured on Qwen3-4B-Q4_K_M/4090, the
    /// split originals are ~1.21 GiB of pure duplicate upload that nothing on this single-sequence decode/prefill
    /// path ever reads (only the batch scheduler's mixed-dtype split-projection path does, via its own lazy
    /// auto-promotion, unaffected by this flag). True restores the engine's long-standing default (included) —
    /// e.g. to isolate whether a regression is residency-related.</summary>
    public bool PreloadRedundantWeightSplits { get; init; }

    /// <summary>Throws when a field is out of range or asks for something this version cannot do.</summary>
    public void Validate()
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(LlmModel);
        ArgumentException.ThrowIfNullOrWhiteSpace(LlmDevice);
        ArgumentException.ThrowIfNullOrWhiteSpace(AudioDevice);
        ArgumentException.ThrowIfNullOrWhiteSpace(SttModel);
        ArgumentException.ThrowIfNullOrWhiteSpace(TtsModel);
        ArgumentNullException.ThrowIfNull(SystemPrompt);
        Require(OutboundSampleRate is >= 8_000 and <= 48_000, nameof(OutboundSampleRate), OutboundSampleRate, "must be 8000-48000 Hz");
        // The VAD pads a segment by 30 ms and scores 32 ms chunks, so a shorter hangover would end a turn inside
        // the padding of its own last word.
        Require(EndOfTurnSilenceMs is >= 100 and <= 10_000, nameof(EndOfTurnSilenceMs), EndOfTurnSilenceMs, "must be 100-10000 ms");
        Require(MaxUtteranceMs is >= 1_000 and <= 120_000, nameof(MaxUtteranceMs), MaxUtteranceMs, "must be 1000-120000 ms");
        Require(BargeInProbability is > 0f and <= 1f, nameof(BargeInProbability), BargeInProbability, "must be in (0, 1]");
        Require(BargeInMinMs >= 0, nameof(BargeInMinMs), BargeInMinMs, "must be non-negative");
        Require(BargeInHoldoffMs >= 0, nameof(BargeInHoldoffMs), BargeInHoldoffMs, "must be non-negative");
        Require(MaxToolRoundsPerTurn >= 1, nameof(MaxToolRoundsPerTurn), MaxToolRoundsPerTurn, "must be at least 1");
        Require(MaxHistoryTokens >= 1, nameof(MaxHistoryTokens), MaxHistoryTokens, "must be positive");
        Require(MaxReplyTokens >= 1, nameof(MaxReplyTokens), MaxReplyTokens, "must be positive");
        Require(FirstSentenceMinChars >= 0, nameof(FirstSentenceMinChars), FirstSentenceMinChars, "must be non-negative");
        Require(MaxSentenceChars >= 1, nameof(MaxSentenceChars), MaxSentenceChars, "must be positive");
        Require(CpuThreadCap >= 0, nameof(CpuThreadCap), CpuThreadCap, "must be non-negative");
        if (PartialTranscripts)
        {
            throw new NotSupportedException("Partial transcripts need a streaming recognizer, which the voice session does not have yet; set PartialTranscripts to false.");
        }
    }

    private static void Require(bool condition, string name, object value, string rule)
    {
        if (!condition)
        {
            throw new ArgumentOutOfRangeException(name, value, $"{name} {rule}.");
        }
    }
}
