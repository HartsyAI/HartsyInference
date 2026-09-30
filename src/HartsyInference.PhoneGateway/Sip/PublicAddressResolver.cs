using System.Globalization;
using System.Net;
using HartsyInference.Core.Logging;
using SIPSorcery.Net;

namespace HartsyInference.PhoneGateway.Sip;

/// <summary>Resolves the address the gateway advertises in Contact and SDP: <c>none</c>, a literal IP, or
/// <c>stun:host[:port]</c>. STUN is asked again on every registration refresh so a changed WAN address is picked up;
/// a failed query keeps the last answer and logs.</summary>
public sealed class PublicAddressResolver
{
    private readonly IPAddress? _literal;
    private readonly string _stunHost;
    private readonly int _stunPort;
    private IPAddress? _last;

    private PublicAddressResolver(PublicAddressMode mode, IPAddress? literal, string stunHost, int stunPort)
    {
        Mode = mode;
        _literal = literal;
        _stunHost = stunHost;
        _stunPort = stunPort;
        _last = literal;
    }

    public PublicAddressMode Mode { get; }

    /// <summary>The most recent answer; null under <see cref="PublicAddressMode.None"/> or before a STUN query succeeded.</summary>
    public IPAddress? LastResolved => Volatile.Read(ref _last);

    /// <summary>Parses the <c>publicAddress</c> setting: <c>none</c> (default), <c>stun:host[:port]</c> or an IP literal.</summary>
    public static PublicAddressResolver Parse(string? setting)
    {
        string value = (setting ?? "").Trim();
        if (value.Length == 0 || value.Equals("none", StringComparison.OrdinalIgnoreCase))
        {
            return new PublicAddressResolver(PublicAddressMode.None, null, "", 0);
        }
        if (value.StartsWith("stun:", StringComparison.OrdinalIgnoreCase))
        {
            string hostPort = value.Substring(5);
            string host = hostPort;
            int port = STUNClient.DEFAULT_STUN_PORT;
            int colon = hostPort.LastIndexOf(':');
            if (colon > 0)
            {
                host = hostPort.Substring(0, colon);
                if (!int.TryParse(hostPort.AsSpan(colon + 1), NumberStyles.None, CultureInfo.InvariantCulture, out port) || port is < 1 or > 65535)
                {
                    throw new ArgumentException($"publicAddress '{value}': the STUN port is not valid.", nameof(setting));
                }
            }
            if (host.Length == 0)
            {
                throw new ArgumentException($"publicAddress '{value}': the STUN host is empty.", nameof(setting));
            }
            return new PublicAddressResolver(PublicAddressMode.Stun, null, host, port);
        }
        if (IPAddress.TryParse(value, out IPAddress? literal))
        {
            return new PublicAddressResolver(PublicAddressMode.Literal, literal, "", 0);
        }
        throw new ArgumentException($"publicAddress '{value}' is not 'none', 'stun:host[:port]' or an IP address.", nameof(setting));
    }

    /// <summary>Returns the address to advertise now; blocks on the STUN round trip in STUN mode.</summary>
    public IPAddress? Resolve()
    {
        switch (Mode)
        {
            case PublicAddressMode.None:
                return null;
            case PublicAddressMode.Literal:
                return _literal;
            default:
                IPAddress? answer = null;
                try
                {
                    answer = STUNClient.GetPublicIPAddress(_stunHost, _stunPort);
                }
                catch (Exception ex) when (ex is System.Net.Sockets.SocketException or TimeoutException or InvalidOperationException)
                {
                    Logs.Warning($"[PhoneGateway] STUN query to {_stunHost}:{_stunPort} failed: {ex.Message}");
                }
                if (answer is null)
                {
                    IPAddress? previous = Volatile.Read(ref _last);
                    Logs.Warning($"[PhoneGateway] STUN gave no public address; keeping {(previous is null ? "the local address" : previous.ToString())}.");
                    return previous;
                }
                if (!answer.Equals(Volatile.Read(ref _last)))
                {
                    Logs.Info($"[PhoneGateway] Public address from STUN: {answer}");
                }
                Volatile.Write(ref _last, answer);
                return answer;
        }
    }
}
