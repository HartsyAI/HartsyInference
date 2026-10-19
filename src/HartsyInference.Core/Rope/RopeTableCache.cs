using System.Runtime.InteropServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Rope;

/// <summary>RoPE tables for one <see cref="RopeFrequencyPlan"/> and one settings snapshot. Host pairs are shared per position
/// signature; device rows are built per backend. Leases keep device rows alive while a backend call uses them.</summary>
/// <remarks>Freeing a device table runs <c>beforeRetire</c> for its backend first, so the owner can invalidate its own step graph.
/// The cache never resets a backend-wide graph itself. A free that throws stays tracked and is retried on the next acquire,
/// release or dispose.</remarks>
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
    private readonly RopeHostTableWriter? _hostWriter;
    private readonly Dictionary<int, List<RopeHostTable>> _slots = new();
    private readonly List<RopeDeviceTable> _pendingRetirement = new();
    private readonly Stack<RopeLeaseSlot> _freeLeaseSlots = new();
    private ulong _lastTicket;
    private bool _disposed;

    /// <summary>Creates a cache. Plan and settings are copied here and never read again.</summary>
    /// <param name="hostWriter">Optional host-pair writer. When absent, pairs come from <see cref="MultiAxisRopeTables"/>.</param>
    public RopeTableCache(
        RopeFrequencyPlan plan,
        RopeTableSettings settings,
        Action<IBackend>? beforeRetire = null,
        RopeHostTableWriter? hostWriter = null)
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
        _hostWriter = hostWriter;
    }

    /// <summary>Returns tables for the signature <paramref name="positions"/> on <paramref name="backend"/>. Dispose the lease
    /// once the backend call no longer needs the tables.</summary>
    public RopeTableLease Acquire(IBackend backend, int slot, int tokens, ReadOnlySpan<float> positions)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (slot < 0)
            throw new ArgumentOutOfRangeException(nameof(slot), slot, "Slot must be non-negative.");
        if (tokens < 0)
            throw new ArgumentOutOfRangeException(nameof(tokens), tokens, "Token count cannot be negative.");
        if (positions.Length != (long)tokens * _plan.Axes)
            throw new ArgumentException($"Positions must hold {tokens} tokens of {_plan.Axes} axes.", nameof(positions));
        if (_preload && tokens == 0)
            throw new ArgumentOutOfRangeException(nameof(tokens), tokens, "Preloaded device tables need at least one token.");

        lock (_gate)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(RopeTableCache));
            FlushPendingRetirement();
            RopeHostTable host = FindOrBuildHost(slot, tokens, positions);
            RopeDeviceTable? device = null;
            if (_preload)
            {
                device = FindOrBuildDevice(host, backend);
                device.Outstanding++;
            }
            RopeLeaseSlot leaseSlot = _freeLeaseSlots.Count > 0 ? _freeLeaseSlots.Pop() : new RopeLeaseSlot();
            ulong ticket = ++_lastTicket;
            leaseSlot.Ticket = ticket;
            leaseSlot.Device = device;
            return new RopeTableLease(this, leaseSlot, ticket, host, device);
        }
    }

    /// <summary>Retires every table so the next acquisition rebuilds. Tables under a lease are freed by its last release.</summary>
    public void Clear()
    {
        lock (_gate)
        {
            if (_disposed)
                return;
            RetireAllLocked();
        }
    }

    /// <summary>Retires this backend's device tables and keeps the host pairs. An idle table is freed now. A table under a lease
    /// is unmapped at once and freed by its last release. A failed free leaves the table mapped, so the next call retries.</summary>
    public void ReleaseBackend(IBackend backend)
    {
        ArgumentNullException.ThrowIfNull(backend);
        lock (_gate)
        {
            foreach (List<RopeHostTable> tables in _slots.Values)
            {
                foreach (RopeHostTable host in tables)
                {
                    if (!host.Devices.TryGetValue(backend, out RopeDeviceTable? device))
                        continue;
                    if (device.Outstanding == 0)
                        FreeIdle(host, device);
                    else
                        Unmap(device);
                    host.Devices.Remove(backend);
                }
            }
            FlushPendingRetirement();
        }
    }

    /// <summary>Marks the cache disposed and retires every table. Tables under a lease are freed by their last release. Later
    /// acquisitions throw.</summary>
    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed)
            {
                // A failed retirement from the first call is retried here.
                FlushPendingRetirement();
                return;
            }
            _disposed = true;
            RetireAllLocked();
        }
    }

    internal void ReleaseLease(RopeLeaseSlot slot, ulong ticket)
    {
        if (Interlocked.CompareExchange(ref slot.Ticket, 0UL, ticket) != ticket)
            return;
        lock (_gate)
        {
            RopeDeviceTable? device = slot.Device;
            slot.Device = null;
            _freeLeaseSlots.Push(slot);
            if (device is not null)
            {
                device.Outstanding--;
                if (device.Outstanding == 0 && device.Retired)
                    _pendingRetirement.Add(device);
            }
            FlushPendingRetirement();
        }
    }

    private void RetireAllLocked()
    {
        foreach (List<RopeHostTable> tables in _slots.Values)
        {
            foreach (RopeHostTable host in tables)
                UnmapAll(host);
        }
        _slots.Clear();
        FlushPendingRetirement();
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

        (float[] cos, float[] sin) = BuildHostPairs(positions, tokens);
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
            UnmapAll(oldest);
            evicted = true;
        }
        if (evicted)
            FlushPendingRetirement();
        return built;
    }

    private (float[] Cos, float[] Sin) BuildHostPairs(ReadOnlySpan<float> positions, int tokens)
    {
        if (_hostWriter is null)
            return MultiAxisRopeTables.ComputeActivePairs(_plan, positions, tokens, _angleOrder, _activePairs);
        float[] cos = new float[checked(tokens * _activePairs)];
        float[] sin = new float[cos.Length];
        _hostWriter(positions, tokens, cos, sin);
        return (cos, sin);
    }

    private RopeDeviceTable FindOrBuildDevice(RopeHostTable host, IBackend backend)
    {
        if (host.Devices.TryGetValue(backend, out RopeDeviceTable? existing))
            return existing;

        int headDim = _plan.HeadDim;
        int count = checked(host.Tokens * headDim);
        Tensor cos = new(new TensorShape(host.Tokens, headDim), DType.F32);
        Tensor? sin = null;
        bool preloadAttempted = false;
        bool built = false;
        bool keepForRetry = false;
        try
        {
            sin = new Tensor(new TensorShape(host.Tokens, headDim), DType.F32);
            unsafe
            {
                RopeTableExpander.Expand(host.Cos, host.Sin, host.Tokens, headDim, _layout, _rotaryDim, _padding,
                    new Span<float>(cos.DataPointer, count), new Span<float>(sin.DataPointer, count));
            }
            // One batch, as the original per-backend tables uploaded it. A failed batch may have uploaded either tensor,
            // so rollback frees both.
            preloadAttempted = true;
            backend.PreloadWeights([cos, sin]);
            RopeDeviceTable device = new(backend, cos, sin);
            host.Devices[backend] = device;
            built = true;
            return device;
        }
        finally
        {
            if (!built)
            {
                if (preloadAttempted && sin is not null)
                {
                    try { backend.FreeWeights([cos, sin]); }
                    catch (Exception error)
                    {
                        // The failed batch may still hold device memory, so the tensors wait for the next flush.
                        Logs.Warning($"[RopeTableCache] Failed to roll back a partial preload; retrying on the next flush: {error.Message}");
                        _pendingRetirement.Add(new RopeDeviceTable(backend, cos, sin));
                        keepForRetry = true;
                    }
                }
                if (!keepForRetry)
                {
                    cos.Dispose();
                    sin?.Dispose();
                }
            }
        }
    }

    /// <summary>Unmaps every device of a host table, so none of them can be reused. Idle devices are queued for freeing.</summary>
    private void UnmapAll(RopeHostTable host)
    {
        foreach (RopeDeviceTable device in host.Devices.Values)
            Unmap(device);
        host.Devices.Clear();
    }

    /// <summary>Marks a device retired. An idle device is queued for freeing now; one under a lease waits for its last release.</summary>
    private void Unmap(RopeDeviceTable device)
    {
        device.Retired = true;
        if (device.Outstanding == 0)
            _pendingRetirement.Add(device);
    }

    /// <summary>Frees an idle device table now. It stays mapped if its free throws, so the next call retries it.</summary>
    private void FreeIdle(RopeHostTable host, RopeDeviceTable device)
    {
        _beforeRetire?.Invoke(device.Backend);
        device.Backend.FreeWeights([device.Cos, device.Sin]);
        device.Retired = true;
        host.Devices.Remove(device.Backend);
        device.Cos.Dispose();
        device.Sin.Dispose();
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
            _pendingRetirement.Remove(device);
            device.Cos.Dispose();
            device.Sin.Dispose();
        }
    }
}
