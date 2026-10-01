using HartsyInference.Tools;

namespace HartsyInference.VoiceHost.Tools;

/// <summary><c>hangup</c>: marks that it ran and answers at once. The gateway sends its BYE the moment it runs the
/// tool, while the model's goodbye is usually still being synthesized, so the host sends the real request only once
/// the turn that asked for it (named by the session's <c>ToolResult</c> event) has ended and its audio has drained to
/// the gateway, within a cap (see <see cref="Calls.VoiceCall"/>). A caller who barges in on that goodbye only cuts it
/// short: the call still ends.</summary>
internal sealed class HangupToolHandler(Action requestHangup) : IToolHandler
{
    public string Name => VoiceHostTools.Hangup;

    public string Description => "End the phone call. Say goodbye first: the call ends once your reply has finished playing.";

    public string JsonSchema => """{"type":"object","properties":{},"additionalProperties":false}""";

    public Task<string> InvokeAsync(string argumentsJson, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        requestHangup();
        return Task.FromResult(ToolOutcome.Write("Ok", "The call ends when your reply has finished playing."));
    }
}
