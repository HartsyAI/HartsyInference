using System.Runtime.InteropServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Rope;

/// <summary>RoPE tables for one <see cref="RopeFrequencyPlan"/> and one settings snapshot. Host tables are shared per
/// position signature; device rows are built per backend. Leases keep tables alive while a backend call uses them.</summary>
/// <remarks>Retirement runs <c>beforeRetire</c> for the owning backend first, so the owner can invalidate its own step
/// graph before device memory is freed. The cache never resets a backend-wide graph itself. A retirement that throws stays
/// tracked and is retried on the next acquire, release or dispose.</remarks>
public sealed class RopeTableCache : IDisposable
{
    private readonly object _gate = new();
    private readonly RopeFrequencyPlan _plan;
    private readonly RopeAngleOrder _angleOrder;
    private readonly RopeTableLayout _layout;
    private readonly int _rotaryDim;
    private readonly int _activePairs;
    private readonly RopeTablePadding _padding;
    private readonly bool _preload;
    private readonly int _retainedSignatures;
    private readonly Action<IBackend>? _beforeRetire;
    private readonly Dictionary<int, List<RopeHostTable>> _slots = new();
    private readonly List<RopeDeviceTable> _pendingRetirement = new();
    private readonly Stack<RopeLeaseSlot> _freeLeaseSlots = new();
    private ulong _lastTicket;
    private bool _disposed;

    /// <summary>Creates a cache. Plan and settings are copied here and never read again.</summary>
    public RopeTableCache(RopeFrequencyPlan plan, RopeTableSettings settings, Action<IBackend>? beforeRetire = null)
    {
        ArgumentNullException.ThrowIfNull(plan);
        ArgumentNullException.ThrowIfNull(settings);
        if (settings.RetainedSignatures < 1)
            throw new ArgumentOutOfRangeException(nameof(settings), settings.RetainedSignatures, "At least one signature must be retained.");
        int effective = RopeTableExpander.EffectiveRotaryDim(plan.HeadDim, settings.RotaryDim, settings.Layout);

        _plan = plan;
        _angleOrder = settings.AngleOrder;
        _layout = settings.Layout;
        _rotaryDim = settings.RotaryDim;
        _activePairs = effective / 2;
        _padding = settings.Padding;
        _preload = settings.Preload;
        _retainedSignatures = settings.RetainedSignatures;
        _beforeRetire = beforeRetire;
    }

    /// <summary>Returns tables for the signature <paramref name="positions"/> on <paramref name="backend"/>. Dispose the
    /// lease once the backend call no longer needs the tables.</summary>
    public RopeTableLease Acquire(IBackend backend, int slot, int tokens, ReadOnlySpan<float> positions)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (slot < 0)
            throw new ArgumentOutOfRangeException(nameof(slot), slot, "Slot must be non-negative.");
        if (tokens < 0)
            throw new ArgumentOutOfRangeException(nameof(tokens), tokens, "Token count cannot be negative.");
        if (positions.Length != (long)tokens * _plan.Axes)
            throw new ArgumentException($"Positions must hold {tokens} tokens of {_plan.Axes} axes.", nameof(positions));

        lock (_gate)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(RopeTableCache));
            FlushPendingRetirement();
            RopeHostTable host = FindOrBuildHost(slot, tokens, positions);
            RopeDeviceTable? device = _preload ? FindOrBuildDevice(host, backend) : null;
            RopeLeaseSlot leaseSlot = _freeLeaseSlots.Count > 0 ? _freeLeaseSlots.Pop() : new RopeLeaseSlot();
            ulong ticket = ++_lastTicket;
            leaseSlot.Ticket = ticket;
            leaseSlot.Host = host;
            host.Outstanding++;
            return new RopeTableLease(this, leaseSlot, ticket, host, device);
        }
    }

    /// <summary>Marks the cache disposed and retires every table. Tables with outstanding leases are freed when their last
    /// lease is released. Later acquisitions throw.</summary>
    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed)
                return;
            _disposed = true;
            foreach (List<RopeHostTable> tables in _slots.Values)
            {
                foreach (RopeHostTable host in tables)
                    RetireHost(host);
            }
            _slots.Clear();
            FlushPendingRetirement();
        }
    }

    internal void ReleaseLease(RopeLeaseSlot slot, ulong ticket)
    {
        if (Interlocked.CompareExchange(ref slot.Ticket, 0UL, ticket) != ticket)
            return;
        lock (_gate)
        {
            RopeHostTable host = slot.Host!;
            slot.Host = null;
            _freeLeaseSlots.Push(slot);
            host.Outstanding--;
            if (host.Retired)
                RetireHost(host);
            FlushPendingRetirement();
        }
    }

    private RopeHostTable FindOrBuildHost(int slot, int tokens, ReadOnlySpan<float> positions)
    {
        ReadOnlySpan<byte> requested = MemoryMarshal.AsBytes(positions);
        if (_slots.TryGetValue(slot, out List<RopeHostTable>? tables))
        {
            for (int i = 0; i < tables.Count; i++)
            {
                RopeHostTable candidate = tables[i];
                if (candidate.Tokens != tokens || !requested.SequenceEqual(MemoryMarshal.AsBytes(candidate.Positions.AsSpan())))
                    continue;
                if (i > 0)
                {
                    tables.RemoveAt(i);
                    tables.Insert(0, candidate);
                }
                return candidate;
            }
        }

        (float[] cos, float[] sin) = MultiAxisRopeTables.ComputeActivePairs(_plan, positions, tokens, _angleOrder, _activePairs);
        RopeHostTable built = new(positions.ToArray(), tokens, cos, sin);
        if (!_slots.TryGetValue(slot, out tables))
        {
            tables = new List<RopeHostTable>();
            _slots[slot] = tables;
        }
        tables.Insert(0, built);
        bool evicted = false;
        while (tables.Count > _retainedSignatures)
        {
            RopeHostTable oldest = tables[^1];
            tables.RemoveAt(tables.Count - 1);
            RetireHost(oldest);
            evicted = true;
        }
        if (evicted)
            FlushPendingRetirement();
        return built;
    }

    private RopeDeviceTable FindOrBuildDevice(RopeHostTable host, IBackend backend)
    {
        if (host.Devices.TryGetValue(backend, out RopeDeviceTable? existing))
            return existing;

        int headDim = _plan.HeadDim;
        int count = checked(host.Tokens * headDim);
        Tensor cos = new(new TensorShape(host.Tokens, headDim), DType.F32);
        Tensor? sin = null;
        bool cosPreloaded = false;
        bool built = false;
        try
        {
            sin = new Tensor(new TensorShape(host.Tokens, headDim), DType.F32);
            unsafe
            {
                RopeTableExpander.Expand(host.Cos, host.Sin, host.Tokens, headDim, _layout, _rotaryDim, _padding,
                    new Span<float>(cos.DataPointer, count), new Span<float>(sin.DataPointer, count));
            }
            backend.PreloadWeights([cos]);
            cosPreloaded = true;
            backend.PreloadWeights([sin]);
            RopeDeviceTable device = new(backend, cos, sin);
            host.Devices[backend] = device;
            built = true;
            return device;
        }
        finally
        {
            if (!built)
            {
                if (cosPreloaded)
                    backend.FreeWeights([cos]);
                cos.Dispose();
                sin?.Dispose();
            }
        }
    }

    private void RetireHost(RopeHostTable host)
    {
        host.Retired = true;
        if (host.Outstanding > 0 || host.Devices.Count == 0)
            return;
        foreach (RopeDeviceTable device in host.Devices.Values)
            _pendingRetirement.Add(device);
        host.Devices.Clear();
    }

    private void FlushPendingRetirement()
    {
        if (_pendingRetirement.Count == 0)
            return;
        RopeDeviceTable[] pending = _pendingRetirement.ToArray();
        foreach (RopeDeviceTable device in pending)
        {
            _beforeRetire?.Invoke(device.Backend);
            device.Backend.FreeWeights([device.Cos, device.Sin]);
            device.Cos.Dispose();
            device.Sin.Dispose();
            _pendingRetirement.Remove(device);
        }
    }
}
