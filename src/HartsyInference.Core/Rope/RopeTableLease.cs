using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Rope;

/// <summary>Borrowed RoPE tables, valid until disposed. Copies of a lease share one release: the first Dispose releases it,
/// and later Dispose calls on any copy do nothing.</summary>
public readonly struct RopeTableLease : IDisposable
{
    private readonly RopeTableCache? _owner;
    private readonly RopeLeaseSlot? _slot;
    private readonly ulong _ticket;
    private readonly RopeHostTable? _host;
    private readonly RopeDeviceTable? _device;

    internal RopeTableLease(RopeTableCache owner, RopeLeaseSlot slot, ulong ticket, RopeHostTable host, RopeDeviceTable? device)
    {
        _owner = owner;
        _slot = slot;
        _ticket = ticket;
        _host = host;
        _device = device;
    }

    /// <summary>False for a default lease, which holds no tables.</summary>
    public bool IsValid => _host is not null;

    /// <summary>Tokens in the signature this lease covers.</summary>
    public int Tokens => _host?.Tokens ?? 0;

    /// <summary>Host cos rows, compact to the rotated pairs of each token.</summary>
    public ReadOnlySpan<float> HostCos => _host is null ? ReadOnlySpan<float>.Empty : _host.Cos;

    /// <summary>Host sin rows, compact to the rotated pairs of each token.</summary>
    public ReadOnlySpan<float> HostSin => _host is null ? ReadOnlySpan<float>.Empty : _host.Sin;

    /// <summary>Device cos rows for the backend the lease was acquired on, or null when the cache does not preload.</summary>
    public Tensor? DeviceCos => _device?.Cos;

    /// <summary>Device sin rows for the backend the lease was acquired on, or null when the cache does not preload.</summary>
    public Tensor? DeviceSin => _device?.Sin;

    /// <summary>Releases the lease once. Copies and repeated calls are no-ops.</summary>
    public void Dispose()
    {
        if (_owner is not null && _slot is not null)
            _owner.ReleaseLease(_slot, _ticket);
    }
}
