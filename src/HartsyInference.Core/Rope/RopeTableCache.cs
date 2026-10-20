using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Rope;

/// <summary>RoPE tables for one <see cref="RopeFrequencyPlan"/> and one settings snapshot. Host pairs are shared per position
/// signature; device rows are built per backend. Leases keep device rows alive while a backend call uses them.</summary>
/// <remarks>Freeing a device table runs <c>beforeRetire</c> for its backend first, so the owner can invalidate its own step graph.
/// The cache never resets a backend-wide graph itself. A free that throws stays tracked and is retried on the next acquire,
/// release or dispose. A cache that does not preload can also hand out staged rows through <see cref="AcquireStaged"/>: host
/// tensors built once per signature that any backend uploads on first use, for owners whose callers carry no backend.</remarks>
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
        ValidateRequest(slot, tokens, positions);
        if (_preload && tokens == 0)
            throw new ArgumentOutOfRangeException(nameof(tokens), tokens, "Preloaded device tables need at least one token.");

        lock (_gate)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(RopeTableCache));
            FlushPendingRetirement();
            RopeHostTable host = FindOrBuildHost(slot, tokens, positions);
            return Lease(host, _preload ? FindOrBuildDevice(host, backend) : null);
        }
    }

    /// <summary>Returns the host tables and the staged <c>[tokens, headDim]</c> rows for the signature <paramref name="positions"/>,
    /// with no backend: whichever backend reads the rows first uploads them. Only for a cache that does not preload. The rows
    /// are built once per signature and shared by every lease on it; dispose the lease once nothing reads them.</summary>
    public RopeTableLease AcquireStaged(int slot, int tokens, ReadOnlySpan<float> positions)
    {
        if (_preload)
            throw new InvalidOperationException("A preloading cache builds rows per backend; acquire them with a backend.");
        ValidateRequest(slot, tokens, positions);
        if (tokens == 0)
            throw new ArgumentOutOfRangeException(nameof(tokens), tokens, "Staged rows need at least one token.");

        lock (_gate)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(RopeTableCache));
            FlushPendingRetirement();
            RopeHostTable host = FindOrBuildHost(slot, tokens, positions);
            if (host.Staged is null)
            {
                (Tensor cos, Tensor sin) = BuildRows(host);
                host.Staged = new RopeDeviceTable(null, cos, sin);
            }
            return Lease(host, host.Staged);
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
    /// is unmapped at once and freed by its last release. A failed free leaves that table mapped, so the next call retries it,
    /// and the backend's other tables are still released; the failure is rethrown afterwards. It still works after
    /// <see cref="Dispose"/>, so an owner tearing down its backend frees whatever remains.</summary>
    public void ReleaseBackend(IBackend backend)
    {
        ArgumentNullException.ThrowIfNull(backend);
        lock (_gate)
        {
            List<Exception>? failures = null;
            foreach (List<RopeHostTable> tables in _slots.Values)
            {
                foreach (RopeHostTable host in tables)
                {
                    if (!host.Devices.TryGetValue(backend, out RopeDeviceTable? device))
                        continue;
                    if (device.Outstanding == 0)
                    {
                        try
                        {
                            FreeIdle(host, device);
                        }
                        catch (Exception error)
                        {
                            // Still mapped, so the next call retries it; the backend's other tables are released now.
                            Collect(ref failures, error);
                            continue;
                        }
                    }
                    else
                    {
                        Unmap(device);
                    }
                    host.Devices.Remove(backend);
                }
            }
            try
            {
                FlushPendingRetirement();
            }
            catch (Exception error)
            {
                Collect(ref failures, error);
            }
            Rethrow(failures);
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

        (Tensor cos, Tensor sin) = BuildRows(host);
        bool built = false;
        bool keepForRetry = false;
        try
        {
            // One batch, as the original per-backend tables uploaded it. A failed batch may have uploaded either tensor,
            // so rollback frees both.
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
                try { backend.FreeWeights([cos, sin]); }
                catch (Exception error)
                {
                    // The failed batch may still hold device memory, so the tensors wait for the next flush.
                    Logs.Warning($"[RopeTableCache] Failed to roll back a partial preload; retrying on the next flush: {error.Message}");
                    _pendingRetirement.Add(new RopeDeviceTable(backend, cos, sin));
                    keepForRetry = true;
                }
                if (!keepForRetry)
                {
                    cos.Dispose();
                    sin.Dispose();
                }
            }
        }
    }

    /// <summary>Allocates the <c>[tokens, headDim]</c> rows of <paramref name="host"/> and expands its pairs into them. Nothing
    /// leaks when a step throws.</summary>
    private (Tensor Cos, Tensor Sin) BuildRows(RopeHostTable host)
    {
        int headDim = _plan.HeadDim;
        int count = checked(host.Tokens * headDim);
        Tensor cos = new(new TensorShape(host.Tokens, headDim), DType.F32);
        Tensor? sin = null;
        bool built = false;
        try
        {
            sin = new Tensor(new TensorShape(host.Tokens, headDim), DType.F32);
            unsafe
            {
                RopeTableExpander.Expand(host.Cos, host.Sin, host.Tokens, headDim, _layout, _rotaryDim, _padding,
                    new Span<float>(cos.DataPointer, count), new Span<float>(sin.DataPointer, count));
            }
            built = true;
            return (cos, sin);
        }
        finally
        {
            if (!built)
            {
                cos.Dispose();
                sin?.Dispose();
            }
        }
    }

    private void ValidateRequest(int slot, int tokens, ReadOnlySpan<float> positions)
    {
        if (slot < 0)
            throw new ArgumentOutOfRangeException(nameof(slot), slot, "Slot must be non-negative.");
        if (tokens < 0)
            throw new ArgumentOutOfRangeException(nameof(tokens), tokens, "Token count cannot be negative.");
        if (positions.Length != (long)tokens * _plan.Axes)
            throw new ArgumentException($"Positions must hold {tokens} tokens of {_plan.Axes} axes.", nameof(positions));
    }

    /// <summary>Hands out a lease on <paramref name="host"/>, counting it against <paramref name="rows"/> when there are rows to
    /// keep alive. Runs under the gate.</summary>
    private RopeTableLease Lease(RopeHostTable host, RopeDeviceTable? rows)
    {
        if (rows is not null)
            rows.Outstanding++;
        RopeLeaseSlot leaseSlot = _freeLeaseSlots.Count > 0 ? _freeLeaseSlots.Pop() : new RopeLeaseSlot();
        ulong ticket = ++_lastTicket;
        leaseSlot.Ticket = ticket;
        leaseSlot.Device = rows;
        return new RopeTableLease(this, leaseSlot, ticket, host, rows);
    }

    /// <summary>Unmaps every device of a host table and its staged rows, so none of them can be reused. Idle tables are queued
    /// for freeing.</summary>
    private void UnmapAll(RopeHostTable host)
    {
        foreach (RopeDeviceTable device in host.Devices.Values)
            Unmap(device);
        host.Devices.Clear();
        if (host.Staged is not null)
        {
            Unmap(host.Staged);
            host.Staged = null;
        }
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
        IBackend backend = device.Backend!;
        _beforeRetire?.Invoke(backend);
        backend.FreeWeights([device.Cos, device.Sin]);
        device.Retired = true;
        host.Devices.Remove(backend);
        DisposeRows(device);
    }

    /// <summary>Frees every table waiting for retirement. One whose free throws stays pending for the next acquire, release or
    /// dispose, and the rest are still freed; the failure is rethrown afterwards, several as one AggregateException.</summary>
    private void FlushPendingRetirement()
    {
        if (_pendingRetirement.Count == 0)
            return;
        List<Exception>? failures = null;
        foreach (RopeDeviceTable device in _pendingRetirement.ToArray())
        {
            // Staged rows were never preloaded, so they only need disposing.
            if (device.Backend is not null)
            {
                try
                {
                    _beforeRetire?.Invoke(device.Backend);
                    device.Backend.FreeWeights([device.Cos, device.Sin]);
                }
                catch (Exception error)
                {
                    Collect(ref failures, error);
                    continue;
                }
            }
            _pendingRetirement.Remove(device);
            DisposeRows(device);
        }
        Rethrow(failures);
    }

    /// <summary>Adds a failure to the list, unwrapping one this cache already aggregated.</summary>
    private static void Collect(ref List<Exception>? failures, Exception error)
    {
        failures ??= [];
        if (error is AggregateException aggregate)
            failures.AddRange(aggregate.InnerExceptions);
        else
            failures.Add(error);
    }

    /// <summary>Rethrows collected retirement failures: one as itself, several as one AggregateException.</summary>
    private static void Rethrow(List<Exception>? failures)
    {
        if (failures is { Count: 1 })
            ExceptionDispatchInfo.Capture(failures[0]).Throw();
        if (failures is not null)
            throw new AggregateException("RoPE device tables failed to retire and stay pending.", failures);
    }

    /// <summary>Disposes both host tensors of a freed table, the second even when the first throws.</summary>
    private static void DisposeRows(RopeDeviceTable device)
    {
        try { device.Cos.Dispose(); }
        finally { device.Sin.Dispose(); }
    }
}
