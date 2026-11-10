using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Gguf;

/// <summary>A one-time disk GGUF cache for a re-quantized or re-packed checkpoint. Converting on every load would hold the
/// source, a converted copy and F32 scratch at once; instead the cache is written once, streaming one tensor at a time
/// through <see cref="GgufWriter"/>'s lazy entries, and later loads map the file directly.</summary>
public static class GgufQuantCache
{
    /// <summary>Resolves a quant mode (q8_0 / q4_0 / q4_k / q5_k / q6_k, plus aliases) to a block dtype, or null for none. A bare
    /// <c>q4</c> means Q4_K, the better 4-bit format; ask for <c>q4_0</c> by name.</summary>
    public static DType? ResolveQuant(string? mode) => mode?.Trim().ToLowerInvariant() switch
    {
        "q8_0" or "q8" => DType.Q8_0,
        "q4_0" => DType.Q4_0,
        "q4_k" or "q4" or "q4_k_m" or "q4_k_s" => DType.Q4_K,
        "q5_k" or "q5" or "q5_k_m" => DType.Q5_K,
        "q6_k" or "q6" => DType.Q6_K,
        _ => null,
    };

    /// <summary>A bounded <c>produceAhead</c> for entries that read a mapped checkpoint and quantize into their own buffers: several run
    /// at once while the writer still emits them in order, and a few large F32 intermediates are the most held at a time.</summary>
    public static int ParallelProduceAhead => Math.Clamp(Environment.ProcessorCount / 4, 1, 4);

    /// <summary>Where a model folder's converted-weight cache goes: <c>_gguf-cache</c> inside <paramref name="sourceDirectory"/> when
    /// that folder can be written (so the cache travels with the model), else a per-folder directory under
    /// <c>~/.cache/hartsyinference/quant-cache</c>. A Hugging Face cache snapshot (<c>snapshots/&lt;rev&gt;</c>) counts as read-only: its
    /// files are links into a content-addressed store, and a stray folder there is not tracked or cleaned with the repo.</summary>
    /// <param name="sourceDirectory">The model folder the cache converts.</param>
    /// <param name="fileName">The cache file's name, which carries whatever distinguishes one conversion from another.</param>
    public static string DefaultPath(string sourceDirectory, string fileName)
    {
        ArgumentException.ThrowIfNullOrEmpty(sourceDirectory);
        ArgumentException.ThrowIfNullOrEmpty(fileName);
        string full = Path.GetFullPath(sourceDirectory);
        string beside = Path.Combine(full, "_gguf-cache");
        if (!IsHuggingFaceSnapshot(full) && CanWrite(beside)) return Path.Combine(beside, fileName);
        string key = System.Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(full)))[..16];
        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        return Path.Combine(home, ".cache", "hartsyinference", "quant-cache", key, fileName);
    }

    private static bool IsHuggingFaceSnapshot(string fullPath) =>
        fullPath.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar).Contains("snapshots", StringComparer.Ordinal);

    private static bool CanWrite(string directory)
    {
        try
        {
            Directory.CreateDirectory(directory);
            string probe = Path.Combine(directory, $".write-probe-{Environment.ProcessId}-{Guid.NewGuid():N}");
            File.WriteAllBytes(probe, []);
            File.Delete(probe);
            return true;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return false;
        }
    }

    /// <summary>Quantizes <paramref name="source"/> by <paramref name="policy"/> into the cache at <paramref name="cachePath"/>
    /// (written on the first call) and returns the cached weights under their source shapes, plus the loader that maps them —
    /// keep it alive while the weights are used.</summary>
    public static (Dictionary<string, Tensor> Weights, GgufLoader Loader) Load(IReadOnlyDictionary<string, Tensor> source,
        string cachePath, GgufQuantPolicy policy, string architecture, string logTag, int produceAhead = 1)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(policy);
        GgufLoader loader = Open(cachePath, () => Entries(source, policy), architecture, logTag, produceAhead: produceAhead);
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
    /// exist or does not match them. The file is written beside its final name and moved into place, so an interrupted conversion
    /// never leaves a cache that looks complete. An existing file is reused only when every <paramref name="metadata"/> value and
    /// the tensor inventory (names, dtypes, element counts) match: a cache left by another version, precision or source checkpoint
    /// is rebuilt rather than read as this one, since quantization keeps the shapes a later check could catch.</summary>
    /// <param name="produceAhead">How many entries the writer produces at once when it writes the cache (see
    /// <see cref="GgufWriter.ProduceAhead"/>). 1 is serial; <see cref="ParallelProduceAhead"/> suits entries that read a mapped
    /// checkpoint and quantize into their own buffers. Each in-flight entry holds its own intermediates, so a caller opts in.</param>
    public static GgufLoader Open(string cachePath, Func<IEnumerable<GgufCacheEntry>> entries, string architecture, string logTag,
        IDictionary<string, object>? metadata = null, int produceAhead = 1)
    {
        ArgumentException.ThrowIfNullOrEmpty(cachePath);
        ArgumentNullException.ThrowIfNull(entries);
        if (File.Exists(cachePath))
        {
            GgufLoader existing = Map(cachePath);
            string? mismatch = Mismatch(existing, entries(), metadata);
            if (mismatch is null)
            {
                Logs.Info($"{logTag} Mapped the quant cache '{cachePath}' ({new FileInfo(cachePath).Length / (1024 * 1024)} MB).");
                return existing;
            }
            existing.Dispose();
            Logs.Warning($"{logTag} The quant cache '{cachePath}' does not match this load ({mismatch}); rebuilding it.");
        }
        Write(cachePath, entries(), architecture, logTag, metadata, produceAhead);
        GgufLoader loader = Map(cachePath);
        Logs.Info($"{logTag} Mapped the quant cache '{cachePath}' ({new FileInfo(cachePath).Length / (1024 * 1024)} MB).");
        return loader;
    }

    /// <summary>Why the mapped cache differs from what <paramref name="entries"/> and <paramref name="metadata"/> would write, or null
    /// when it matches.</summary>
    public static string? Mismatch(GgufLoader cache, IEnumerable<GgufCacheEntry> entries, IDictionary<string, object>? metadata)
    {
        ArgumentNullException.ThrowIfNull(cache);
        ArgumentNullException.ThrowIfNull(entries);
        if (metadata is not null)
        {
            foreach (KeyValuePair<string, object> kv in metadata)
            {
                string expected = MetadataText(kv.Value);
                string found = cache.Metadata.TryGetValue(kv.Key, out object? value) ? MetadataText(value) : "missing";
                if (found != expected) return $"{kv.Key} is {found}, expected {expected}";
            }
        }
        int count = 0;
        foreach (GgufCacheEntry entry in entries)
        {
            count++;
            if (!cache.Descriptors.TryGetValue(entry.Name, out GgufTensorDescriptor? stored)) return $"'{entry.Name}' is missing";
            if (stored.DType != entry.DType) return $"'{entry.Name}' is {stored.DType.Name}, expected {entry.DType.Name}";
            if (stored.Shape.ElementCount != entry.Shape.ElementCount)
                return $"'{entry.Name}' has {stored.Shape.ElementCount} elements, expected {entry.Shape.ElementCount}";
        }
        return cache.Descriptors.Count == count ? null : $"it holds {cache.Descriptors.Count} tensors, expected {count}";
    }

    /// <summary>Deletes temporary files a writer of <paramref name="cachePath"/> left behind when its process died mid-write, so an
    /// interrupted conversion does not keep its partial file on disk forever. A temporary whose process is still running is left
    /// alone: it may be writing.</summary>
    private static void RemoveAbandonedTemporaries(string cachePath, string logTag)
    {
        string full = Path.GetFullPath(cachePath);
        string prefix = Path.GetFileName(full) + ".";
        foreach (string file in Directory.EnumerateFiles(Path.GetDirectoryName(full)!, prefix + "*.tmp"))
        {
            string[] parts = Path.GetFileName(file)[prefix.Length..].Split('.');
            if (parts.Length != 3 || !int.TryParse(parts[0], out int processId) || IsRunning(processId)) continue;
            try
            {
                File.Delete(file);
                Logs.Info($"{logTag} Removed the abandoned partial quant cache '{file}'.");
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                Logs.Warning($"{logTag} Could not remove the abandoned partial quant cache '{file}': {ex.Message}");
            }
        }
    }

    private static bool IsRunning(int processId)
    {
        try
        {
            using System.Diagnostics.Process process = System.Diagnostics.Process.GetProcessById(processId);
            return !process.HasExited;
        }
        catch (ArgumentException)
        {
            return false;
        }
    }

    private static string MetadataText(object? value) => System.Convert.ToString(value, System.Globalization.CultureInfo.InvariantCulture) ?? "";

    private static GgufLoader Map(string cachePath)
    {
        GgufLoader loader = new();
        try
        {
            loader.Load(cachePath);
            return loader;
        }
        catch
        {
            loader.Dispose();
            throw;
        }
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
        IDictionary<string, object>? metadata, int produceAhead)
    {
        Logs.Info($"{logTag} Writing the quant cache '{cachePath}' (one time, one tensor at a time)...");
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(cachePath))!);
        RemoveAbandonedTemporaries(cachePath, logTag);
        // Each writer has its own file, so two processes first-loading the same cache never write into one another.
        string temporary = $"{cachePath}.{Environment.ProcessId}.{Guid.NewGuid():N}.tmp";
        try
        {
            using (GgufWriter writer = new(temporary))
            {
                writer.ProduceAhead = produceAhead;
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
            try
            {
                File.Move(temporary, cachePath, overwrite: true);
            }
            catch (Exception ex) when ((ex is IOException or UnauthorizedAccessException) && File.Exists(cachePath))
            {
                // Another writer finished first and its file is in use (Windows refuses to replace a mapped file): use that one.
                File.Delete(temporary);
                Logs.Info($"{logTag} Another load finished the quant cache first; using its file.");
                return;
            }
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
