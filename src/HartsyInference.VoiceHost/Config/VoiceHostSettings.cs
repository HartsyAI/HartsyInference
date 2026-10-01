using System.Text;
using HartsyInference.Voice;

namespace HartsyInference.VoiceHost.Config;

/// <summary>The loaded configuration, the session options built from it and the link token read from its secret file.
/// The token is held here and nowhere else, and <see cref="ToString"/> never prints it.</summary>
public sealed record VoiceHostSettings
{
    public required VoiceHostConfig Config { get; init; }

    /// <summary>PhoneLink token; empty when <c>link.tokenFile</c> is not set.</summary>
    public required string LinkToken { get; init; }

    /// <summary>Options of the model set and of every call's session.</summary>
    public required VoiceAgentOptions Agent { get; init; }

    /// <summary>Mode the socket file gets after bind.</summary>
    public required UnixFileMode SocketMode { get; init; }

    private bool PrintMembers(StringBuilder builder)
    {
        builder.Append("Config = ").Append(Config)
            .Append(", LinkToken = ").Append(LinkToken.Length == 0 ? "(none)" : "(set)")
            .Append(", Agent = ").Append(Agent)
            .Append(", SocketMode = ").Append(SocketMode);
        return true;
    }
}
