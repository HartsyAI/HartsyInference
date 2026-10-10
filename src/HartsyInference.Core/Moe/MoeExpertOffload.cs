using HartsyInference.Core.Backends;
using HartsyInference.Core.Moe.Residency;

namespace HartsyInference.Core.Moe;

/// <summary>Running totals of where a model's routed experts ran.</summary>
/// <param name="ResidentRows">Routed (token, slot) pairs served by an expert resident in the device cache.</param>
/// <param name="StreamedRows">Pairs served on the device by an expert uploaded for that one call (a large batch, such as a prefill).</param>
/// <param name="HostRows">Pairs served on the CPU.</param>
/// <param name="Admitted">Experts queued for upload into the cache after a miss.</param>
public readonly record struct MoeOffloadStats(long ResidentRows, long StreamedRows, long HostRows, long Admitted)
{
    /// <summary>Share of routed pairs served from the device cache.</summary>
    public double ResidentShare => ResidentRows + StreamedRows + HostRows == 0 ? 0 : (double)ResidentRows / (ResidentRows + StreamedRows + HostRows);
}

/// <summary>
/// Expert offload for one model: a device cache holds as many routed experts as its budget allows, and the rest run elsewhere.
/// An expert the cache holds runs on the device; a miss runs on the CPU through <see cref="Host"/>, or, when it serves at least
/// <see cref="StreamRowThreshold"/> rows in one call (a prefill), on the device from a copy uploaded for that call, since moving its
/// weights once costs less than computing that many rows on the CPU.
/// </summary>
/// <remarks>
/// <para><b>Admission.</b> After a layer runs, its misses may be queued for upload so later steps find them resident. Misses fill
/// free room first, as many as fit without evicting anything (at most <see cref="MaxFillsPerLayer"/>). Beyond that a miss may
/// displace a resident expert only when its recent use (<see cref="DecayedLfuHysteresisPolicy"/>, the same scores that choose
/// eviction victims) reaches <see cref="AdmitMinScore"/>, and at most <see cref="MaxSwapsPerLayer"/> per layer per step, so a full
/// cache does not churn its upload ring on every token. Uploads run on the cache's upload stream; the
/// next step's plan waits only for the experts it uses.</para>
/// <para><b>Model contract.</b> A MoE block registers each layer's experts with <see cref="RegisterLayer"/>, plans each forward with
/// <see cref="ExpertScheduler.Plan"/> against <see cref="Cache"/> and <see cref="Policy"/>, and reports with
/// <see cref="Record"/> and <see cref="AfterLayer"/>. The weights a layer registers must be the tensors its device path projects
/// with, so a resident expert's device copy is the one the projection finds.</para>
/// </remarks>
public sealed class MoeExpertOffload : IDisposable
{
    private readonly DecayedLfuHysteresisPolicy _scores;
    private readonly IDisposable? _owned;
    private readonly List<ExpertKey> _admit = [];
    private readonly Dictionary<int, Func<ExpertKey, ExpertWeights>> _layers = [];
    private readonly Dictionary<int, long> _expertBytes = [];
    private long _residentRows, _streamedRows, _hostRows, _admitted;
    private int _disposed;

    /// <summary>Creates the offload over <paramref name="cache"/>, which must not have been used yet.</summary>
    /// <param name="cache">The device expert cache; its budget is the offload's expert budget.</param>
    /// <param name="host">Runs experts on the CPU.</param>
    /// <param name="scores">Recent-use scores: attached to <paramref name="cache"/> to choose eviction victims, and read for admission.</param>
    /// <param name="owned">Disposed with the offload after the cache, for whatever else the caller built for it.</param>
    public MoeExpertOffload(ExpertCacheBase cache, IExpertHostRunner host, DecayedLfuHysteresisPolicy scores, IDisposable? owned = null)
    {
        ArgumentNullException.ThrowIfNull(cache);
        ArgumentNullException.ThrowIfNull(host);
        ArgumentNullException.ThrowIfNull(scores);
        if (cache is not IResidencyAwareExpertCache residency)
            throw new ArgumentException("The offload needs a residency-aware cache.", nameof(cache));
        cache.AttachResidencyPolicy(scores);
        CacheBase = cache;
        Cache = residency;
        Host = host;
        _scores = scores;
        _owned = owned;
    }

    /// <summary>The device cache.</summary>
    public IResidencyAwareExpertCache Cache { get; }

    /// <summary>The device cache, for prefetching and stats.</summary>
    public ExpertCacheBase CacheBase { get; }

    /// <summary>Runs the experts the device does not hold.</summary>
    public IExpertHostRunner Host { get; }

    /// <summary>Resident experts on the device, misses elsewhere; the planner never uploads.</summary>
    public IMissExecutionPolicy Policy => ResidentFirstPolicy.Instance;

    /// <summary>Rows at or above which a miss runs on the device from a copy uploaded for that call.</summary>
    public int StreamRowThreshold { get; init; } = 16;

    /// <summary>Decayed accesses a miss needs before it may displace a resident expert.</summary>
    public double AdmitMinScore { get; init; } = 8.0;

    /// <summary>Most misses one layer may queue for upload in one step once the cache is full.</summary>
    public int MaxSwapsPerLayer { get; init; } = 1;

    /// <summary>Most misses one layer may queue for upload in one step while the cache has room.</summary>
    public int MaxFillsPerLayer { get; init; } = 16;

    /// <summary>Where routed pairs have run so far.</summary>
    public MoeOffloadStats Stats => new(Interlocked.Read(ref _residentRows), Interlocked.Read(ref _streamedRows),
        Interlocked.Read(ref _hostRows), Interlocked.Read(ref _admitted));

    /// <summary>Makes <paramref name="layer"/>'s experts resolvable; call once per MoE layer before its first forward.</summary>
    public void RegisterLayer(int layer, int expertCount, Func<ExpertKey, ExpertWeights> resolve)
    {
        ArgumentNullException.ThrowIfNull(resolve);
        CacheBase.RegisterBank(new ExpertBank(layer, expertCount, resolve));
        lock (_layers) _layers[layer] = resolve;
    }

    /// <summary>An expert's weights, from the layer that registered it: what the CPU runner reads.</summary>
    /// <exception cref="KeyNotFoundException">No layer registered <paramref name="key"/>'s layer.</exception>
    public ExpertWeights Resolve(ExpertKey key)
    {
        Func<ExpertKey, ExpertWeights> resolve;
        lock (_layers) resolve = _layers[key.Layer];
        return resolve(key);
    }

    /// <summary>Whether a miss serving <paramref name="rows"/> pairs runs on the device from an uploaded copy rather than on the CPU.</summary>
    public bool Streams(int rows) => rows >= StreamRowThreshold;

    /// <summary>Counts one expert's rows by where they ran.</summary>
    public void Record(ExpertPlacement placement, bool streamed, int rows)
    {
        if (placement == ExpertPlacement.Gpu) Interlocked.Add(ref _residentRows, rows);
        else if (streamed) Interlocked.Add(ref _streamedRows, rows);
        else Interlocked.Add(ref _hostRows, rows);
    }

    /// <summary>Queues some of a layer's misses for upload, by the admission rule in the type remarks. Call after the layer's lease is
    /// released.</summary>
    public void AfterLayer(List<ExpertKey> misses)
    {
        ArgumentNullException.ThrowIfNull(misses);
        if (misses.Count == 0) return;
        ExpertCacheStats stats = CacheBase.Stats;
        // Free room is whole experts that fit without evicting anything; past that, only a hot miss may displace a resident one.
        long expertBytes = ExpertBytes(misses[0]);
        long free = stats.BudgetBytes - stats.ResidentBytes;
        int fills = expertBytes > 0 ? (int)Math.Min(MaxFillsPerLayer, Math.Max(0, free / expertBytes)) : 0;
        int swaps = MaxSwapsPerLayer;
        _admit.Clear();
        foreach (ExpertKey key in misses)
        {
            if (fills > 0)
            {
                _admit.Add(key);
                fills--;
            }
            else if (swaps > 0 && _scores.ScoreOf(key) >= AdmitMinScore)
            {
                _admit.Add(key);
                swaps--;
            }
            if (fills == 0 && swaps == 0) break;
        }
        if (_admit.Count == 0) return;
        int started = CacheBase.Prefetch(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(_admit));
        Interlocked.Add(ref _admitted, started);
    }

    /// <summary>Bytes one expert of <paramref name="key"/>'s layer occupies; every expert of a layer has the same shape.</summary>
    private long ExpertBytes(ExpertKey key)
    {
        int layer = key.Layer;
        lock (_layers)
        {
            if (_expertBytes.TryGetValue(layer, out long cached)) return cached;
        }
        long bytes = Resolve(key).Bytes;
        lock (_layers) _expertBytes[layer] = bytes;
        return bytes;
    }

    /// <summary>Fills the empty cache at load: up to <paramref name="perLayer"/> experts of each layer, in expert order, until the budget
    /// is full. A seed only; admission reshapes it to what is actually routed.</summary>
    public void Seed(IReadOnlyList<(int Layer, int ExpertCount)> layers, int perLayer)
    {
        ArgumentNullException.ThrowIfNull(layers);
        if (perLayer <= 0) return;
        List<ExpertKey> keys = [];
        foreach ((int layer, int count) in layers)
        {
            keys.Clear();
            for (int e = 0; e < Math.Min(perLayer, count); e++) keys.Add(new ExpertKey(layer, e));
            if (CacheBase.Prefetch(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(keys)) < keys.Count) break;
        }
    }

    /// <summary>Disposes the cache, which drains its uploads and frees every resident expert, then whatever the caller handed over.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        CacheBase.Dispose();
        _owned?.Dispose();
    }
}
