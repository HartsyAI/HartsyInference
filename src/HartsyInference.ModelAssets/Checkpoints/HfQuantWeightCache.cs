using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf;

namespace HartsyInference.ModelAssets.Checkpoints;

/// <summary>Loads a Hugging Face checkpoint (plain, NF4, FP8, GPTQ, block-scaled) at the dtypes a model picks per weight. A weight used
/// as stored is borrowed from the mapped checkpoint; every converted one goes through a one-time GGUF cache
/// (<see cref="GgufQuantCache"/>) written one tensor at a time, so a 4-bit or FP8 checkpoint is decoded and re-quantized once, not on
/// every load. The cache records its source files and the caller's identity values and is rebuilt when either changes.</summary>
public static class HfQuantWeightCache
{
    /// <summary>The weights of <paramref name="checkpoint"/> that <paramref name="include"/> keeps, each at <paramref name="target"/>'s
    /// dtype for it. The returned owner holds whatever backs the tensors (the checkpoint mapping, the cache mapping or both); dispose
    /// it with the model. On success the checkpoint belongs to the owner; on failure the caller still owns it.</summary>
    /// <param name="cachePath">Where converted weights are cached; unused when nothing converts.</param>
    /// <param name="architecture">The cache file's <c>general.architecture</c>.</param>
    /// <param name="logTag">Log prefix, e.g. <c>[MOSS-VL]</c>.</param>
    /// <param name="identity">Values the cache must match to be reused (a version, the chosen precision); the source files are added.</param>
    /// <param name="produceAhead">Entries converted at once when the cache is written (see <see cref="GgufQuantCache.Open"/>); 1 is serial.</param>
    public static (IReadOnlyDictionary<string, Tensor> Weights, IDisposable Owner) Load(HfQuantCheckpoint checkpoint,
        Func<HfQuantWeight, DType> target, string cachePath, string architecture, string logTag,
        IReadOnlyDictionary<string, object>? identity = null, Func<string, bool>? include = null, int produceAhead = 1)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        ArgumentNullException.ThrowIfNull(target);
        include ??= static _ => true;
        List<HfQuantWeight> borrowed = [], converted = [];
        foreach (HfQuantWeight w in checkpoint.Weights.Values.Where(w => include(w.Name)).OrderBy(static w => w.Name, StringComparer.Ordinal))
            (w.Storage == HfWeightStorage.Plain && target(w) == w.StoredDType ? borrowed : converted).Add(w);

        Dictionary<string, Tensor> weights = new(borrowed.Count + converted.Count, StringComparer.Ordinal);
        foreach (HfQuantWeight w in borrowed) weights[w.Name] = checkpoint.Borrow(w.Name);
        if (converted.Count == 0) return (weights, checkpoint);

        GgufQuantError error = new();
        IEnumerable<GgufCacheEntry> Entries() => converted.Select(w =>
        {
            DType dtype = target(w);
            return new GgufCacheEntry(w.Name, w.Shape, dtype, () => checkpoint.Produce(w.Name, dtype, error));
        });
        Dictionary<string, object> metadata = new(StringComparer.Ordinal) { ["hartsy.cache.source"] = checkpoint.SourceIdentity };
        if (identity is not null)
            foreach (KeyValuePair<string, object> kv in identity) metadata[kv.Key] = kv.Value;

        Logs.Info($"{logTag} {converted.Count} weights convert ({string.Join(", ", converted.GroupBy(w => $"{w.Storage}→{target(w).Name}").Select(g => $"{g.Count()} {g.Key}"))}); "
            + $"{borrowed.Count} are used as stored.");
        GgufLoader loader = GgufQuantCache.Open(cachePath, Entries, architecture, logTag, metadata, produceAhead);
        try
        {
            if (error.Count > 0) Logs.Info($"{logTag} Re-quantization: {error}.");
            foreach (KeyValuePair<string, Tensor> kv in GgufQuantCache.ReadBack(loader, Entries())) weights[kv.Key] = kv.Value;
            return (weights, new Owners(loader, checkpoint));
        }
        catch
        {
            loader.Dispose();
            throw;
        }
    }

    /// <summary>Disposes the cache mapping and the checkpoint together.</summary>
    private sealed class Owners(IDisposable first, IDisposable second) : IDisposable
    {
        private int _disposed;

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
            try { first.Dispose(); }
            finally { second.Dispose(); }
        }
    }
}
