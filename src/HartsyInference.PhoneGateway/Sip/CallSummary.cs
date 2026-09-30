using HartsyInference.PhoneLink;

namespace HartsyInference.PhoneGateway.Sip;

/// <summary>A snapshot of the live call for <c>/health</c> and the per-call log line.</summary>
public sealed record CallSummary(uint CallId, string SipCallId, LinkCallDirection Direction, string Remote, DateTime StartedUtc, CallState State);
