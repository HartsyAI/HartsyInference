using HartsyInference.Core.Logging;
using HartsyInference.PhoneGateway.Media;

namespace HartsyInference.PhoneGateway.Transport;

/// <summary>What a live call does while the voice host is unreachable: keeps the caller company with the "one moment"
/// prompt, re-attaches the call (<c>CallStart(resume)</c>) the moment the link is back, and after
/// <see cref="LinkOutageGuardOptions.OutageHangupMs"/> says goodbye and hangs up. The RTP clock keeps running the
/// whole time; this class only writes prompts and calls back into the controller.</summary>
/// <remarks>Driven by four notifications from the controller and link (<see cref="CallStarted"/>, <see cref="CallEnded"/>,
/// <see cref="LinkConnected"/>, <see cref="LinkDisconnected"/>) and three callbacks it invokes: <see cref="PlayPrompt"/>
/// returns the prompt's length in milliseconds, <see cref="ResumeCall"/> re-announces the call, <see cref="HangUp"/>
/// ends it. The wait runs on a thread-pool task with <c>Task.Delay</c>; it is not on the media path.</remarks>
public sealed class LinkOutageGuard : IDisposable
{
    private const int HangupGraceMs = 100;

    private readonly LinkOutageGuardOptions _options;
    private readonly object _lock = new();
    private CancellationTokenSource? _outage;
    private bool _inCall;
    private bool _linkUp;
    private long _outages;
    private long _outageHangups;

    public LinkOutageGuard(LinkOutageGuardOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentOutOfRangeException.ThrowIfLessThan(options.OutageHangupMs, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(options.PromptRepeatMs, 1);
        _options = options;
    }

    /// <summary>Plays a prompt to the caller and returns its duration in milliseconds.</summary>
    public Func<PromptKind, int>? PlayPrompt { get; set; }

    /// <summary>The link is back while a call is live: send <c>CallStart(resume: true)</c>.</summary>
    public Action? ResumeCall { get; set; }

    /// <summary>The outage outlasted the timeout and the goodbye has played: hang the call up.</summary>
    public Action? HangUp { get; set; }

    /// <summary>Outages that started during a live call.</summary>
    public long Outages => Volatile.Read(ref _outages);

    /// <summary>Calls hung up because the host never came back.</summary>
    public long OutageHangups => Volatile.Read(ref _outageHangups);

    /// <summary>True while an outage wait is running.</summary>
    public bool InOutage
    {
        get
        {
            lock (_lock)
            {
                return _outage is not null;
            }
        }
    }

    public void LinkConnected()
    {
        bool resume;
        lock (_lock)
        {
            _linkUp = true;
            resume = CancelOutageLocked() && _inCall;
        }
        if (resume)
        {
            Logs.Info("[PhoneGateway] Voice host link restored mid-call; resuming the call.");
            ResumeCall?.Invoke();
        }
    }

    public void LinkDisconnected()
    {
        lock (_lock)
        {
            _linkUp = false;
            if (_inCall)
            {
                StartOutageLocked();
            }
        }
    }

    public void CallStarted()
    {
        lock (_lock)
        {
            _inCall = true;
            if (!_linkUp)
            {
                StartOutageLocked();
            }
        }
    }

    public void CallEnded()
    {
        lock (_lock)
        {
            _inCall = false;
            CancelOutageLocked();
        }
    }

    private void StartOutageLocked()
    {
        if (_outage is not null)
        {
            return;
        }
        _outages++;
        CancellationTokenSource outage = new();
        _outage = outage;
        _ = RunOutageAsync(outage);
    }

    private bool CancelOutageLocked()
    {
        CancellationTokenSource? outage = _outage;
        _outage = null;
        if (outage is null)
        {
            return false;
        }
        // The task that owns the source disposes it once it has observed the cancellation.
        outage.Cancel();
        return true;
    }

    private async Task RunOutageAsync(CancellationTokenSource outage)
    {
        CancellationToken ct = outage.Token;
        long deadline = Environment.TickCount64 + _options.OutageHangupMs;
        try
        {
            Logs.Warning($"[PhoneGateway] Voice host unreachable during a call; holding the caller for up to {_options.OutageHangupMs} ms.");
            while (true)
            {
                PlayPrompt?.Invoke(PromptKind.OneMoment);
                long remaining = deadline - Environment.TickCount64;
                if (remaining <= 0)
                {
                    break;
                }
                await Task.Delay((int)Math.Min(remaining, _options.PromptRepeatMs), ct).ConfigureAwait(false);
                if (Environment.TickCount64 >= deadline)
                {
                    break;
                }
            }
            int goodbyeMs = PlayPrompt?.Invoke(PromptKind.Goodbye) ?? 0;
            await Task.Delay(goodbyeMs + HangupGraceMs, ct).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            Logs.Debug("[PhoneGateway] Host outage wait cancelled (link restored or call ended).");
            outage.Dispose();
            return;
        }
        lock (_lock)
        {
            bool current = ReferenceEquals(_outage, outage);
            if (current)
            {
                _outage = null;
                _outageHangups++;
            }
            outage.Dispose();
            if (!current)
            {
                return;
            }
        }
        Logs.Warning("[PhoneGateway] Voice host did not return; hanging the call up.");
        HangUp?.Invoke();
    }

    public void Dispose()
    {
        lock (_lock)
        {
            _inCall = false;
            CancelOutageLocked();
        }
    }
}
