namespace HartsyInference.Engine.Audio.Wake;

/// <summary>What a satellite is being told the server is doing right now.
///
/// <para>A voice turn is several seconds of work spread across transcription, generation and synthesis, and
/// from the device's side all of it looks the same: it sent audio, and nothing has come back. So the light
/// stays on one colour and the user, who cannot see a progress bar, is left deciding whether to repeat
/// themselves. Latency work makes the wait shorter; this makes it legible, which is the other half of feeling
/// fast and the cheaper half.</para>
///
/// <para>These are sent as <c>status</c> frames on the wake socket, which the device is already holding open.
/// A satellite that does not know the type ignores it — the protocol has always skipped unknown frames — so
/// nothing here needs the device and the server to be upgraded together.</para></summary>
public static class WakeStatus
{
    /// <summary>The utterance has been captured; the speaker can stop talking.</summary>
    public const string Captured = "captured";

    /// <summary>Transcription is running.</summary>
    public const string Transcribing = "transcribing";

    /// <summary>The assistant is generating a reply. Sent by whatever owns the turn, not by the wake service,
    /// which has handed the transcript on by then.</summary>
    public const string Thinking = "thinking";

    /// <summary>Audio is on its way.</summary>
    public const string Speaking = "speaking";

    /// <summary>The turn is over, however it ended.</summary>
    public const string Done = "done";

    /// <summary>Something failed. The <c>detail</c> is for a log, not for a speaker.</summary>
    public const string Error = "error";

    /// <summary>Builds the <c>data</c> object of a status frame.
    ///
    /// <para>Here rather than inline at the call site because this is the wire contract a satellite parses,
    /// and the satellite is in another repository and another language. A test can pin the exact bytes.</para></summary>
    public static string Data(string state, string? detail = null) =>
        $"{{\"state\":{WakeFrameCodec.Escape(state)}"
        + (detail is null ? "" : $",\"detail\":{WakeFrameCodec.Escape(detail)}") + "}";

    /// <summary>Whether a string is one of the states above. The wire is not validated against this — an
    /// unknown state is simply ignored by the device — but a caller inside the process can check itself.</summary>
    public static bool IsKnown(string? state) => state is Captured or Transcribing or Thinking or Speaking or Done or Error;
}
