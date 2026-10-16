using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Gguf;

/// <summary>A one-time disk GGUF cache for a re-quantized or re-packed checkpoint. Converting on every load would hold the
/// source, a converted copy and F32 scratch at once; instead the cache is written once, streaming one tensor at a time
/// through <see cref="GgufWriter"/>'s lazy entries, and later loads map the file directly.</summary>
public static class GgufQuantCache
{
    /// <summary>Resolves a quant mode (q8_0 / q4_k / q5_k / q6_k, plus aliases) to a block dtype, or null for none.</summary>
    public static DType? ResolveQuant(string? mode) => mode?.Trim().ToLowerInvariant() switch
    {
        "q8_0" or "q8" => DType.Q8_0,
        "q4_k" or "q4" or "q4_k_m" or "q4_k_s" => DType.Q4_K,
        "q5_k" or "q5" or "q5_k_m" => DType.Q5_K,
        "q6_k" or "q6" => DType.Q6_K,
        _ => null,
    };

    /// <summary>Quantizes <paramref name="source"/> by <paramref name="policy"/> into the cache at <paramref name="cachePath"/>
    /// (written on the first call) and returns the cached weights under their source shapes, plus the loader that maps them —
    /// keep it alive while the weights are used.</summary>
    public static (Dictionary<string, Tensor> Weights, GgufLoader Loader) Load(IReadOnlyDictionary<string, Tensor> source,
        string cachePath, GgufQuantPolicy policy, string architecture, string logTag)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(policy);
        GgufLoader loader = Open(cachePath, () => Entries(source, policy), architecture, logTag);
        try
        {
            // Shapes come back from the source, not the file: the writer emits ggml order (see GgufQuantizer.ReadBack).
            return (GgufQuantizer.ReadBack(loader, source), loader);
        }
        catch
        {
            loader.Dispose();
            throw;
        }
    }

    /// <summary>Maps the cache at <paramref name="cachePath"/>, writing it from <paramref name="entries"/> first when it does not
    /// exist. The file is written beside its final name and moved into place, so an interrupted conversion never leaves a
    /// cache that looks complete.</summary>
    public static GgufLoader Open(string cachePath, Func<IEnumerable<GgufCacheEntry>> entries, string architecture, string logTag,
        IDictionary<string, object>? metadata = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(cachePath);
        ArgumentNullException.ThrowIfNull(entries);
        if (!File.Exists(cachePath))
        {
            Write(cachePath, entries(), architecture, logTag, metadata);
        }
        GgufLoader loader = new();
        try
        {
            loader.Load(cachePath);
        }
        catch
        {
            loader.Dispose();
            throw;
        }
        Logs.Info($"{logTag} Mapped the quant cache '{cachePath}' ({new FileInfo(cachePath).Length / (1024 * 1024)} MB).");
        return loader;
    }

    /// <summary>The cache's tensors under the engine-order shapes <paramref name="entries"/> declared (see <see cref="GgufQuantizer.ReadBack"/>).</summary>
    public static Dictionary<string, Tensor> ReadBack(GgufLoader loader, IEnumerable<GgufCacheEntry> entries)
    {
        ArgumentNullException.ThrowIfNull(entries);
        Dictionary<string, TensorShape> shapes = new(StringComparer.Ordinal);
        foreach (GgufCacheEntry entry in entries) shapes[entry.Name] = entry.Shape;
        return GgufQuantizer.ReadBack(loader, name => shapes.TryGetValue(name, out TensorShape shape) ? shape : null);
    }

    /// <summary>One lazy entry per source tensor: quantized or cast to the dtype <paramref name="policy"/> picks, or passed through.</summary>
    public static IEnumerable<GgufCacheEntry> Entries(IReadOnlyDictionary<string, Tensor> source, GgufQuantPolicy policy)
    {
        foreach (KeyValuePair<string, Tensor> kv in source)
        {
            Tensor src = kv.Value;
            DType target = policy.ResolveTargetDType(kv.Key, src);
            yield return new GgufCacheEntry(kv.Key, src.Shape, target, () => Convert(src, target));
        }
    }

    /// <summary>A tensor of <paramref name="target"/> dtype holding <paramref name="source"/>, which is left untouched: a fresh view
    /// when the dtype already matches, a cast for float targets, a quantization for block targets.</summary>
    public static Tensor Convert(Tensor source, DType target)
    {
        ArgumentNullException.ThrowIfNull(source);
        if (source.DType == target) return source.Reshape(source.Shape);
        if (!target.IsQuantized) return source.CastTo(target);
        return GgufQuantizer.Quantize(source, target);
    }

    private static void Write(string cachePath, IEnumerable<GgufCacheEntry> entries, string architecture, string logTag,
        IDictionary<string, object>? metadata)
    {
        Logs.Info($"{logTag} Writing the quant cache '{cachePath}' (one time, one tensor at a time)...");
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(cachePath))!);
        string temporary = cachePath + ".tmp";
        try
        {
            using (GgufWriter writer = new(temporary))
            {
                writer.SetMetadata("general.architecture", architecture);
                writer.SetMetadata("general.name", $"{architecture} (HartsyInference quant cache)");
                if (metadata is not null)
                {
                    foreach (KeyValuePair<string, object> kv in metadata) writer.SetMetadata(kv.Key, kv.Value);
                }
                foreach (GgufCacheEntry entry in entries)
                {
                    writer.AddTensor(entry.Name, entry.Shape, entry.DType, entry.Produce);
                }
                writer.Flush();
            }
            File.Move(temporary, cachePath, overwrite: true);
        }
        catch
        {
            try
            {
                if (File.Exists(temporary)) File.Delete(temporary);
            }
            catch (IOException cleanup)
            {
                Logs.Error($"{logTag} Could not remove the partial quant cache '{temporary}'.", cleanup);
            }
            throw;
        }
        Logs.Info($"{logTag} Wrote the quant cache ({new FileInfo(cachePath).Length / (1024 * 1024)} MB).");
    }
}
