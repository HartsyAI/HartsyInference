using HartsyInference.Core.Backends;

namespace HartsyInference.LLM.Transformer;

/// <summary>Keeps the host at most one ring's worth of steps ahead of the device during a stepped forward, so a forward
/// stopped between steps leaves that many steps queued instead of the rest of the work.</summary>
/// <remarks>Before issuing step k it waits on the fence recorded after step k minus the ring length, never on the step
/// just issued, so the device always has the next step queued behind the running one and an unstopped forward does not
/// stall. On a backend whose ops finish before they return there are no fences and it costs nothing. A local value:
/// call <see cref="Dispose"/> from the caller's own <c>finally</c> rather than through <c>using</c>, which would make the
/// local read-only and run every call on a copy.</remarks>
internal ref struct DeviceRunAhead
{
    private readonly IBackend _backend;
    private readonly Span<nint> _fences;
    private readonly bool _enabled;
    private int _issued;

    /// <summary>Bounds <paramref name="backend"/> to <paramref name="fences"/>.Length steps in flight when
    /// <paramref name="enabled"/>; the span is the ring and is cleared here.</summary>
    public DeviceRunAhead(IBackend backend, Span<nint> fences, bool enabled)
    {
        if (fences.IsEmpty)
        {
            throw new ArgumentException("The fence ring needs at least one slot.", nameof(fences));
        }
        _backend = backend;
        _fences = fences;
        _enabled = enabled;
        _issued = 0;
        fences.Clear();
    }

    /// <summary>Call before issuing a step: blocks until the step issued one ring length ago has finished.</summary>
    public void BeforeStep()
    {
        if (!_enabled)
        {
            return;
        }
        int slot = _issued % _fences.Length;
        nint fence = _fences[slot];
        if (fence == 0)
        {
            return;
        }
        _fences[slot] = 0;
        try
        {
            _backend.WaitFence(fence);
        }
        finally
        {
            _backend.ReleaseFence(fence);
        }
    }

    /// <summary>Call after issuing a step: records where it ends.</summary>
    public void AfterStep()
    {
        if (!_enabled)
        {
            return;
        }
        _fences[_issued % _fences.Length] = _backend.RecordFence();
        _issued++;
    }

    /// <summary>Gives back the fences still held without waiting on them.</summary>
    public void Dispose()
    {
        for (int i = 0; i < _fences.Length; i++)
        {
            nint fence = _fences[i];
            if (fence != 0)
            {
                _fences[i] = 0;
                _backend.ReleaseFence(fence);
            }
        }
    }
}
