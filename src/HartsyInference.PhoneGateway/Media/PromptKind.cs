namespace HartsyInference.PhoneGateway.Media;

/// <summary>The prompts the gateway can play on its own, without the voice host.</summary>
public enum PromptKind
{
    /// <summary>"One moment": played when the host drops mid-call, repeated while the outage lasts.</summary>
    OneMoment,
    /// <summary>Played once before the gateway hangs up a call the host never came back for.</summary>
    Goodbye,
}
