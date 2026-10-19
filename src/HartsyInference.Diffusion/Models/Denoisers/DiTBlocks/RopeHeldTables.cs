using HartsyInference.Core.Backends;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Holds one device-table lease per backend for one slot of a <see cref="RopeTableCache"/>, so the raw tensors a
/// family returns stay valid until that backend builds a different signature. A block loop reads them after the call
/// returns, and two backends can build different signatures concurrently, so retiring them early would free tables in use.
/// A family whose callers carry no backend holds one staged lease instead (<see cref="GetOrBuildStaged"/>), shared by every
/// backend that uploads the rows.</summary>
/// <remarks>Build-then-swap: the new lease is acquired before the backend's previous lease is released, so a failed build
/// leaves the previous tables valid. Every call takes the holder's own gate, so concurrent callers (context-parallel ranks)
/// are serialized here. The held entry is installed before the previous lease is released, so a release that throws
/// (a failed free) leaves no consumed lease in the map.</remarks>
internal sealed class RopeHeldTables<TKey> : IDisposable where TKey : IEquatable<TKey>
{
    private readonly object _gate = new();
    private readonly RopeTableCache _cache;
    private readonly int _slot;
    private readonly int _axes;
    private readonly Dictionary<IBackend, Held> _held = new(ReferenceEqualityComparer.Instance);
    private Held? _staged;

    /// <summary>Creates a holder for <paramref name="slot"/> of <paramref name="cache"/>. Positions carry <paramref name="axes"/> values per token.</summary>
    public RopeHeldTables(RopeTableCache cache, int slot, int axes)
    {
        _cache = cache;
        _slot = slot;
        _axes = axes;
    }

    /// <summary>Returns the tensors <paramref name="backend"/> holds for <paramref name="key"/>, or false when it holds another
    /// signature or none. Lets a caller whose builder needs more than the key skip assembling that state on a hit.</summary>
    public bool TryGet(IBackend backend, TKey key, out (Tensor Cos, Tensor Sin) tables)
    {
        ArgumentNullException.ThrowIfNull(backend);
        lock (_gate)
        {
            if (_held.TryGetValue(backend, out Held? current) && current.Key.Equals(key))
            {
                tables = (current.Cos, current.Sin);
                return true;
            }
            tables = default;
            return false;
        }
    }

    /// <summary>Returns the tensors for <paramref name="key"/> on <paramref name="backend"/>. Builds from
    /// <paramref name="buildPositions"/> only when the backend's held signature differs; the builder receives the key, so a
    /// static lambda makes a hit allocation-free.</summary>
    public (Tensor Cos, Tensor Sin) GetOrBuild(IBackend backend, TKey key, Func<TKey, float[]> buildPositions)
    {
        ArgumentNullException.ThrowIfNull(backend);
        lock (_gate)
        {
            if (_held.TryGetValue(backend, out Held? current) && current.Key.Equals(key))
                return (current.Cos, current.Sin);

            float[] positions = buildPositions(key);
            RopeTableLease lease = _cache.Acquire(backend, _slot, positions.Length / _axes, positions);
            Tensor cos;
            Tensor sin;
            try
            {
                cos = lease.DeviceCos ?? throw new InvalidOperationException("RoPE device tables require a preloading cache.");
                sin = lease.DeviceSin ?? throw new InvalidOperationException("RoPE device tables require a preloading cache.");
            }
            catch
            {
                lease.Dispose();
                throw;
            }

            _held.TryGetValue(backend, out Held? previous);
            _held[backend] = new Held(key, lease, cos, sin);
            previous?.Lease.Dispose();
            return (cos, sin);
        }
    }

    /// <summary>Returns the staged rows for <paramref name="key"/> from a cache that does not preload. Builds from
    /// <paramref name="buildPositions"/> only when the held signature differs; the rows stay valid until the next build. The
    /// builder receives the key, so a static lambda makes a hit allocation-free.</summary>
    public (Tensor Cos, Tensor Sin) GetOrBuildStaged(TKey key, Func<TKey, float[]> buildPositions)
    {
        lock (_gate)
        {
            if (_staged is not null && _staged.Key.Equals(key))
                return (_staged.Cos, _staged.Sin);

            float[] positions = buildPositions(key);
            RopeTableLease lease = _cache.AcquireStaged(_slot, positions.Length / _axes, positions);
            Tensor cos;
            Tensor sin;
            try
            {
                cos = lease.StagedCos ?? throw new InvalidOperationException("RoPE staged rows require a cache that does not preload.");
                sin = lease.StagedSin ?? throw new InvalidOperationException("RoPE staged rows require a cache that does not preload.");
            }
            catch
            {
                lease.Dispose();
                throw;
            }

            Held? previous = _staged;
            _staged = new Held(key, lease, cos, sin);
            previous?.Lease.Dispose();
            return (cos, sin);
        }
    }

    /// <summary>Releases every held lease. The cache frees each table once its last lease is released.</summary>
    public void Dispose()
    {
        lock (_gate)
        {
            Held[] all = _held.Values.ToArray();
            _held.Clear();
            Held? staged = _staged;
            _staged = null;
            foreach (Held held in all)
                held.Lease.Dispose();
            staged?.Lease.Dispose();
        }
    }

    private sealed class Held
    {
        public Held(TKey key, RopeTableLease lease, Tensor cos, Tensor sin)
        {
            Key = key;
            Lease = lease;
            Cos = cos;
            Sin = sin;
        }

        public TKey Key { get; }

        public RopeTableLease Lease { get; }

        public Tensor Cos { get; }

        public Tensor Sin { get; }
    }
}
