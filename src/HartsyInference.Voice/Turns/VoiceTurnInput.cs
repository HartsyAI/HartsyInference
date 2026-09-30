using HartsyInference.Core.Numerics;

namespace HartsyInference.Voice.Turns;

/// <summary>What starts a turn: a closed utterance from the audio thread, a DTMF key, or text the host wants spoken.</summary>
internal sealed class VoiceTurnInput
{
    private VoiceTurnInput(VoiceTurnKind kind, long receivedNs)
    {
        Kind = kind;
        ReceivedNs = receivedNs;
    }

    /// <summary>What started the turn.</summary>
    public VoiceTurnKind Kind { get; }

    /// <summary>When the input arrived (the endpoint decision for an utterance), in monotonic nanoseconds.</summary>
    public long ReceivedNs { get; }

    /// <summary>The utterance at 16 kHz, ±1.</summary>
    public float[]? Audio { get; private init; }

    /// <summary>Samples between the end of the caller's speech and the endpoint decision.</summary>
    public long HangoverSamples { get; private init; }

    /// <summary>Per-frame front-end timings since the previous endpoint.</summary>
    public LatencyHistogram.Summary FrameTimes { get; private init; }

    /// <summary>The user message for a DTMF turn, or the text of a spoken prompt.</summary>
    public string? Text { get; private init; }

    /// <summary>Completed when a spoken prompt has played or was interrupted.</summary>
    public TaskCompletionSource? Completion { get; private init; }

    /// <summary>A closed utterance.</summary>
    public static VoiceTurnInput Utterance(float[] audio, long hangoverSamples, LatencyHistogram.Summary frameTimes, long endpointNs) =>
        new(VoiceTurnKind.Utterance, endpointNs) { Audio = audio, HangoverSamples = hangoverSamples, FrameTimes = frameTimes };

    /// <summary>A key press, carried to the model as the user message <c>[DTMF n]</c>.</summary>
    public static VoiceTurnInput Dtmf(char digit, long receivedNs) =>
        new(VoiceTurnKind.Dtmf, receivedNs) { Text = "[DTMF " + digit + "]" };

    /// <summary>Text the agent says without asking the model.</summary>
    public static VoiceTurnInput Speak(string text, TaskCompletionSource completion, long receivedNs) =>
        new(VoiceTurnKind.Speak, receivedNs) { Text = text, Completion = completion };
}
