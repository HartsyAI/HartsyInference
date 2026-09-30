using SIPSorcery.SIP;

namespace HartsyInference.PhoneGateway.Sip;

/// <summary>How the gateway refused one INVITE, as <see cref="InviteRejectionLedger"/> remembers it.</summary>
internal readonly record struct InviteRejection(SIPResponseStatusCodesEnum Status, string Reason, long RecordedAtMs);
