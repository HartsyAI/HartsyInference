using System.Net;
using HartsyInference.Core.Logging;
using SIPSorcery.SIP;
using SIPSorcery.SIP.App;

namespace HartsyInference.PhoneGateway.Sip;

/// <summary>The gateway's SIP identity: one <see cref="SIPTransport"/> with a UDP or TCP channel, the advertised
/// Contact host from <see cref="PublicAddressResolver"/>, and, when a registrar is configured, a
/// <see cref="SIPRegistrationUserAgent"/> that re-resolves the public address before every REGISTER.</summary>
public sealed class SipAccount : IDisposable
{
    private readonly SipAccountOptions _options;
    private readonly SIPTransport _transport = new();
    private SIPRegistrationUserAgent? _registration;
    private string _registrationState = "not configured";
    private int _started;
    private int _disposed;

    public SipAccount(SipAccountOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (!IPAddress.TryParse(options.ListenAddress, out _))
        {
            throw new ArgumentException($"sip.listenAddress '{options.ListenAddress}' is not an IP address.", nameof(options));
        }
        if (options.Port is < 0 or > 65535)
        {
            throw new ArgumentException($"sip.port {options.Port} is out of range (0 picks an ephemeral port).", nameof(options));
        }
        if (options.Transport is not ("udp" or "tcp"))
        {
            throw new ArgumentException($"sip.transport '{options.Transport}' must be udp or tcp.", nameof(options));
        }
        if (options.Registrar.Length > 0 && options.Username.Length == 0)
        {
            throw new ArgumentException("sip.username is required when a registrar is set.", nameof(options));
        }
        _options = options;
    }

    public SIPTransport Transport => _transport;

    /// <summary>The port the SIP channel actually listens on (the configured one, or the ephemeral pick for port 0).</summary>
    public int ListeningPort => _transport.GetSIPChannels().FirstOrDefault()?.ListeningEndPoint.Port ?? _options.Port;

    public PublicAddressResolver PublicAddress => _options.PublicAddress;

    /// <summary>Credentials for outbound calls through the registrar.</summary>
    public string Username => _options.Username;

    public string Password => _options.Password;

    public string Registrar => _options.Registrar;

    public bool IsRegistered => _registration?.IsRegistered ?? false;

    /// <summary>One line for <c>/health</c>: not configured, registering, registered, or the last failure.</summary>
    public string RegistrationState => Volatile.Read(ref _registrationState);

    /// <summary>The registration expiry actually used, clamped into the accepted range.</summary>
    public int ExpirySeconds => Math.Clamp(_options.RegistrationExpirySeconds, SipAccountOptions.MinExpirySeconds, SipAccountOptions.MaxExpirySeconds);

    public void Start()
    {
        if (Interlocked.Exchange(ref _started, 1) != 0)
        {
            throw new InvalidOperationException("SipAccount is already started.");
        }
        IPEndPoint listen = new(IPAddress.Parse(_options.ListenAddress), _options.Port);
        SIPChannel channel = _options.Transport == "tcp" ? new SIPTCPChannel(listen) : new SIPUDPChannel(listen);
        _transport.AddSIPChannel(channel);
        IPAddress? advertised = _options.PublicAddress.Resolve();
        if (advertised is not null)
        {
            _transport.ContactHost = advertised.ToString();
        }
        Logs.Info($"[PhoneGateway] SIP listening on {_options.Transport}:{listen}; contact host {(advertised is null ? "local" : advertised.ToString())}.");
        if (_options.Registrar.Length == 0)
        {
            return;
        }
        if (ExpirySeconds != _options.RegistrationExpirySeconds)
        {
            Logs.Warning($"[PhoneGateway] sip.registrationExpirySeconds {_options.RegistrationExpirySeconds} clamped to {ExpirySeconds}.");
        }
        SIPRegistrationUserAgent registration = new(_transport, _options.Username, _options.Password, _options.Registrar, ExpirySeconds);
        registration.RegistrationSuccessful += (uri, _) =>
        {
            Volatile.Write(ref _registrationState, "registered");
            Logs.Info($"[PhoneGateway] Registered {uri} with {_options.Registrar} (expiry {ExpirySeconds} s).");
        };
        registration.RegistrationTemporaryFailure += (uri, _, message) =>
        {
            Volatile.Write(ref _registrationState, "temporary failure: " + message);
            Logs.Warning($"[PhoneGateway] Registration of {uri} failed temporarily: {message}");
        };
        registration.RegistrationFailed += (uri, _, message) =>
        {
            Volatile.Write(ref _registrationState, "failed: " + message);
            Logs.Error($"[PhoneGateway] Registration of {uri} failed: {message}");
        };
        registration.RegistrationRemoved += (uri, _) =>
        {
            Volatile.Write(ref _registrationState, "removed");
            Logs.Warning($"[PhoneGateway] Registration of {uri} was removed by the registrar.");
        };
        registration.AdjustRegister = RefreshContactHost;
        Volatile.Write(ref _registrationState, "registering");
        _registration = registration;
        registration.Start();
    }

    public void Stop()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }
        SIPRegistrationUserAgent? registration = _registration;
        _registration = null;
        // Stop() sends a zero-expiry REGISTER so the registrar forgets us at once.
        registration?.Stop();
        _transport.Shutdown();
    }

    /// <summary>Runs before each REGISTER: re-resolves the public address and rewrites the Contact host to match.</summary>
    private SIPRequest RefreshContactHost(SIPRequest request)
    {
        IPAddress? advertised = _options.PublicAddress.Resolve();
        if (advertised is null)
        {
            return request;
        }
        string host = advertised.ToString();
        _transport.ContactHost = host;
        if (request.Header.Contact is not null)
        {
            foreach (SIPContactHeader contact in request.Header.Contact)
            {
                if (contact.ContactURI is not null)
                {
                    int port = contact.ContactURI.HostPort is null ? _options.Port : int.Parse(contact.ContactURI.HostPort, System.Globalization.CultureInfo.InvariantCulture);
                    contact.ContactURI.Host = $"{host}:{port}";
                }
            }
        }
        return request;
    }

    public void Dispose()
    {
        Stop();
        _transport.Dispose();
    }
}
