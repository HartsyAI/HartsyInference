using HartsyInference.PhoneLink;
using HartsyInference.Tools;

namespace HartsyInference.VoiceHost.Tools;

/// <summary>The tools a call's model is offered: the six telephony tools the gateway runs (<c>hangup</c>,
/// <c>send_dtmf</c>, <c>transfer</c>, <c>hold</c>, <c>unhold</c>, <c>play_prompt</c>) and <c>get_time</c>, answered here.</summary>
public static class VoiceHostTools
{
    public const string Hangup = "hangup";
    public const string SendDtmf = "send_dtmf";
    public const string Transfer = "transfer";
    public const string Hold = "hold";
    public const string Unhold = "unhold";
    public const string PlayPrompt = "play_prompt";
    public const string GetTime = "get_time";

    private const string NoArguments = """{"type":"object","properties":{},"additionalProperties":false}""";

    /// <summary>Every tool name the host knows, in the order they are offered.</summary>
    public static IReadOnlyList<string> Names { get; } = [Hangup, SendDtmf, Transfer, Hold, Unhold, PlayPrompt, GetTime];

    /// <summary>A registry with the <paramref name="enabled"/> tools for one call.</summary>
    /// <param name="enabled">Names from <see cref="Names"/>.</param>
    /// <param name="request">Sends a telephony request to the gateway for this call and returns its answer.</param>
    /// <param name="requestHangup">Arms the hang-up that follows the current reply.</param>
    /// <param name="clock">The time <c>get_time</c> reports.</param>
    internal static ToolRegistry Build(IReadOnlyList<string> enabled, Func<ToolRequestMessage, CancellationToken, Task<ToolResultMessage>> request,
        Action requestHangup, TimeProvider clock)
    {
        ArgumentNullException.ThrowIfNull(enabled);
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(requestHangup);
        ArgumentNullException.ThrowIfNull(clock);
        ToolRegistry registry = new();
        foreach (string name in Names)
        {
            if (!enabled.Contains(name, StringComparer.Ordinal))
            {
                continue;
            }
            registry.Add(name switch
            {
                Hangup => new HangupToolHandler(requestHangup),
                GetTime => new GetTimeToolHandler(clock),
                _ => new TelephonyToolHandler(name, Description(name), Schema(name), request),
            });
        }
        return registry;
    }

    private static string Description(string name) => name switch
    {
        SendDtmf => "Press keys on the phone keypad, for example to answer a phone menu.",
        Transfer => "Transfer the caller to another phone number. Only numbers the phone system allows can be reached.",
        Hold => "Put the caller on hold.",
        Unhold => "Take the caller off hold.",
        PlayPrompt => "Play a short recorded prompt to the caller.",
        _ => throw new ArgumentOutOfRangeException(nameof(name), name, "Not a gateway tool."),
    };

    private static string Schema(string name) => name switch
    {
        SendDtmf => """{"type":"object","properties":{"digits":{"type":"string","description":"Keys to press, from 0-9 * # A-D, for example \"1\" or \"123#\"."}},"required":["digits"],"additionalProperties":false}""",
        Transfer => """{"type":"object","properties":{"target":{"type":"string","description":"The phone number to transfer the caller to."}},"required":["target"],"additionalProperties":false}""",
        PlayPrompt => """{"type":"object","properties":{"name":{"type":"string","enum":["one-moment","goodbye"],"description":"Which prompt to play."}},"required":["name"],"additionalProperties":false}""",
        _ => NoArguments,
    };
}
