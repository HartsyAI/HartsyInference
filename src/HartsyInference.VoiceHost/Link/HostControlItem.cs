using HartsyInference.PhoneLink;
using HartsyInference.VoiceHost.Calls;

namespace HartsyInference.VoiceHost.Link;

/// <summary>A control frame queued for the sender thread, the only writer on the connection. <see cref="Arg"/> is the
/// turn, request id or end reason the frame carries, <see cref="Payload"/> its JSON body, and <see cref="Call"/> the call
/// whose sender-side state the frame changes (a flush raises its epoch, a call end stops its audio).</summary>
internal readonly record struct HostControlItem(LinkMessageType Type, uint CallId, uint Arg, ulong Timestamp, object? Payload, VoiceCall? Call);
