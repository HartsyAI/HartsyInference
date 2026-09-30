using System.Collections.Concurrent;
using System.Net;
using SIPSorcery.SIP;

namespace HartsyInference.PhoneGateway.Tests.Support;

/// <summary>A bare SIP transport with no transactions: sends one INVITE (fixed Call-ID and branch) as many times as asked,
/// the way a caller retransmits it, and records every response status it gets back.</summary>
internal sealed class RawSipClient : IDisposable
{
    private readonly SIPTransport _transport = new();
    private readonly ConcurrentQueue<int> _statuses = new();

    public RawSipClient()
    {
        _transport.AddSIPChannel(new SIPUDPChannel(new IPEndPoint(IPAddress.Loopback, 0)));
        _transport.SIPTransportResponseReceived += (_, _, response) =>
        {
            _statuses.Enqueue(response.StatusCode);
            return Task.CompletedTask;
        };
    }

    /// <summary>Every response status received, in arrival order.</summary>
    public int[] Statuses => _statuses.ToArray();

    /// <summary>Sends the same INVITE <paramref name="copies"/> times, <paramref name="gapMs"/> apart.</summary>
    public async Task SendInviteCopiesAsync(int gatewayPort, int copies, int gapMs)
    {
        SIPRequest invite = SIPRequest.GetRequest(SIPMethodsEnum.INVITE, SIPURI.ParseSIPURI($"sip:agent@127.0.0.1:{gatewayPort}"));
        SIPEndPoint gateway = new(SIPProtocolsEnum.udp, IPAddress.Loopback, gatewayPort);
        for (int i = 0; i < copies; i++)
        {
            await _transport.SendRequestAsync(gateway, invite);
            await Task.Delay(gapMs);
        }
    }

    public void Dispose()
    {
        _transport.Shutdown();
        _transport.Dispose();
    }
}
