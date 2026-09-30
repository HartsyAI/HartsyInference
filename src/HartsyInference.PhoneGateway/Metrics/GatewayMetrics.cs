namespace HartsyInference.PhoneGateway.Metrics;

/// <summary>Process-wide counters for <c>/metrics</c> and <c>/health</c>. Counters are plain <see cref="Interlocked"/>
/// longs; the live call's media numbers come through <see cref="MediaProbe"/>, which the call controller points at
/// the current call, and the link's through <see cref="LinkProbe"/>.</summary>
public sealed class GatewayMetrics
{
    private long _callsInbound;
    private long _callsOutbound;
    private long _callsActive;
    private long _callsRejectedBusy;
    private long _callsDeclined;
    private long _callsRejectedHostDown;
    private long _callsFailed;
    private long _callsMediaFault;
    private long _callSecondsTotal;
    private long _dtmfReceived;
    private long _toolRequests;
    private long _toolFailures;
    private long _outages;
    private long _outageHangups;

    /// <summary>Returns the live call's media counters, or null between calls.</summary>
    public Func<MediaSnapshot?>? MediaProbe { get; set; }

    /// <summary>Returns the link's counters.</summary>
    public Func<LinkSnapshot>? LinkProbe { get; set; }

    /// <summary>True when the SIP registration is current (or none is configured).</summary>
    public Func<bool>? RegisteredProbe { get; set; }

    public long CallsInbound => Volatile.Read(ref _callsInbound);
    public long CallsOutbound => Volatile.Read(ref _callsOutbound);
    public long CallsActive => Volatile.Read(ref _callsActive);
    public long CallsRejectedBusy => Volatile.Read(ref _callsRejectedBusy);
    public long CallsDeclined => Volatile.Read(ref _callsDeclined);
    public long CallsRejectedHostDown => Volatile.Read(ref _callsRejectedHostDown);
    public long CallsFailed => Volatile.Read(ref _callsFailed);

    /// <summary>Calls ended because their RTP tick thread faulted.</summary>
    public long CallsMediaFault => Volatile.Read(ref _callsMediaFault);
    public long CallSecondsTotal => Volatile.Read(ref _callSecondsTotal);
    public long DtmfReceived => Volatile.Read(ref _dtmfReceived);
    public long ToolRequests => Volatile.Read(ref _toolRequests);
    public long ToolFailures => Volatile.Read(ref _toolFailures);
    public long Outages => Volatile.Read(ref _outages);
    public long OutageHangups => Volatile.Read(ref _outageHangups);

    public void CallStarted(bool inbound)
    {
        Interlocked.Increment(ref inbound ? ref _callsInbound : ref _callsOutbound);
        Interlocked.Increment(ref _callsActive);
    }

    public void CallEnded(long seconds)
    {
        Interlocked.Decrement(ref _callsActive);
        Interlocked.Add(ref _callSecondsTotal, seconds);
    }

    public void RejectedBusy() => Interlocked.Increment(ref _callsRejectedBusy);

    public void Declined() => Interlocked.Increment(ref _callsDeclined);

    public void RejectedHostDown() => Interlocked.Increment(ref _callsRejectedHostDown);

    public void Failed() => Interlocked.Increment(ref _callsFailed);

    public void MediaFault() => Interlocked.Increment(ref _callsMediaFault);

    public void Dtmf() => Interlocked.Increment(ref _dtmfReceived);

    public void Tool(bool failed)
    {
        Interlocked.Increment(ref _toolRequests);
        if (failed)
        {
            Interlocked.Increment(ref _toolFailures);
        }
    }

    public void Outage() => Interlocked.Increment(ref _outages);

    public void OutageHangup() => Interlocked.Increment(ref _outageHangups);
}
