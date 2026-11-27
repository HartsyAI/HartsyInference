using System.Text.Json;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.Duo;

/// <summary>The offline conversions that make the Duo Realization weights loadable by the engine without torch: the released distributed-checkpoint LoRA to safetensors, the LoRA merged into the original-layout backbone shards
/// (row order and BF16 rounding as upstream), and the exact AdaLN tables for a fixed time list. Every output is written through a staging file, one tensor at a time.</summary>
public static class DuoRepacker
{
    public const string BlockAdalnTableSuffix = ".adaln_proj.table";
    public const string AdalnTimesKey = "adaln_times";
    private const string IndexFileName = "model.safetensors.index.json";

    /// <summary>Writes the LoRA as an F32 safetensors file with the module names it has upstream, sorted, so a consumer needs neither torch nor a pickle reader.</summary>
    public static void WriteLoraSafetensors(DuoLoraSource lora, string outputPath)
    {
        ArgumentNullException.ThrowIfNull(lora);
        ArgumentException.ThrowIfNullOrEmpty(outputPath);
        List<SafeTensorsStreamWriter.Entry> entries = [];
        foreach ((string key, DType dtype, long[] shape) in lora.Describe())
        {
            string module = DuoLoraKeyMap.ModuleOf(key, out bool isA)!;
            entries.Add(new SafeTensorsStreamWriter.Entry(key, DType.F32, new TensorShape(shape), () =>
            {
                (Tensor a, Tensor b) = lora.Read(module);
                if (isA)
                {
                    b.Dispose();
                    return (a, true);
                }
                a.Dispose();
                return (b, true);
            }));
        }
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outputPath))!);
        SafeTensorsStreamWriter.Write(outputPath, entries, new Dictionary<string, string> { ["format"] = "pt", ["producer"] = "DuoRepacker" });
    }

    /// <summary>Merges <paramref name="lora"/> into every shard of <paramref name="backboneDir"/> and writes the shards, with the same file names, and a fresh index to <paramref name="outputDir"/>. The fused qkv comes out as <c>[Q; K; V]</c> blocks
    /// (the engine's order) whatever <see cref="DuoMergeOptions.InputQkv"/> was; <c>mlp.fc1</c> stays <c>[gate; value]</c>. Every LoRA module must land on a tensor; one that does not fails the repack. Returns the written file names relative to <paramref name="outputDir"/>.</summary>
    public static IReadOnlyList<string> MergeBackbone(string backboneDir, DuoLoraSource lora, string outputDir, DuoMergeOptions options, Action<string>? log = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(backboneDir);
        ArgumentNullException.ThrowIfNull(lora);
        ArgumentException.ThrowIfNullOrEmpty(outputDir);
        ArgumentNullException.ThrowIfNull(options);
        string[] shards = ShardFiles(backboneDir);
        if (string.Equals(Path.GetFullPath(backboneDir).TrimEnd('\\', '/'), Path.GetFullPath(outputDir).TrimEnd('\\', '/'), StringComparison.OrdinalIgnoreCase))
            throw new ArgumentException("The output folder must differ from the backbone folder; sources are never overwritten.", nameof(outputDir));

        Dictionary<string, List<(string Module, DuoLoraTarget Target)>> byKey = ModulesByKey(lora);
        Dictionary<string, TensorShape> shapes = new(StringComparer.Ordinal);
        foreach (string shard in shards)
        {
            using SafeTensorsLoader header = new();
            header.Load(shard);
            foreach (KeyValuePair<string, SafeTensorDescriptor> tensor in header.Descriptors)
            {
                if (!shapes.TryAdd(tensor.Key, tensor.Value.Shape))
                    throw new InvalidDataException($"Tensor '{tensor.Key}' is in more than one shard; the repack would be ambiguous. Nothing was written.");
            }
        }
        CheckTargets(shapes, byKey, lora, options);
        Directory.CreateDirectory(outputDir);
        HashSet<string> merged = new(StringComparer.Ordinal);
        Dictionary<string, string> weightMap = new(StringComparer.Ordinal);
        List<string> written = [];
        long totalBytes = 0;
        foreach (string shard in shards)
        {
            using SafeTensorsLoader loader = new();
            loader.Load(shard);
            List<SafeTensorsStreamWriter.Entry> entries = [];
            foreach (KeyValuePair<string, SafeTensorDescriptor> tensor in loader.Descriptors)
            {
                string name = tensor.Key;
                if (options.DropBlockAdalnProj && IsBlockAdalnProj(name)) continue;
                List<(string Module, DuoLoraTarget Target)>? modules = byKey.GetValueOrDefault(name);
                if (modules is not null) foreach ((string module, _) in modules) merged.Add(module);
                bool qkv = IsQkvWeight(name);
                entries.Add(new SafeTensorsStreamWriter.Entry(name, tensor.Value.DType, tensor.Value.Shape,
                    () => Materialize(loader.GetTensor(name), qkv, modules, lora, options)));
                weightMap[name] = Path.GetFileName(shard);
                totalBytes += tensor.Value.DType.ComputeByteCount(tensor.Value.Shape.ElementCount);
            }

            string target = Path.Combine(outputDir, Path.GetFileName(shard));
            log?.Invoke($"merging {Path.GetFileName(shard)} ({entries.Count} tensors)");
            SafeTensorsStreamWriter.Write(target, entries, new Dictionary<string, string> { ["format"] = "pt" });
            written.Add(Path.GetFileName(shard));
        }

        if (merged.Count != lora.Modules.Count)
        {
            string missing = lora.Modules.First(m => !merged.Contains(m));
            throw new InvalidDataException(
                $"{lora.Modules.Count - merged.Count} LoRA module(s) matched no backbone tensor (for example '{missing}'); the shards are not the Duo backbone.");
        }

        WriteIndex(Path.Combine(outputDir, IndexFileName), weightMap, totalBytes);
        written.Add(IndexFileName);
        return written;
    }

    /// <summary>The merge of <see cref="MergeBackbone"/> on loaded weights, with the same rounding and, for per-head input, the same qkv
    /// reorder. Every check runs before any entry changes. Each rewritten entry is replaced by a new tensor, and the caller disposes the returned
    /// tensors once the weights are no longer used; the replaced tensors stay the caller's.</summary>
    /// <exception cref="InvalidDataException">A module has no target or does not fit it, or the qkv layout does not match the options.</exception>
    /// <exception cref="NotSupportedException">A tensor that would be rewritten is neither BF16 nor F32.</exception>
    public static IReadOnlyList<Tensor> MergeInto(IDictionary<string, Tensor> weights, DuoLoraSource lora, DuoMergeOptions options,
        Action<string>? log = null)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(lora);
        ArgumentNullException.ThrowIfNull(options);
        Dictionary<string, List<(string Module, DuoLoraTarget Target)>> byKey = ModulesByKey(lora);
        Dictionary<string, TensorShape> shapes = weights.ToDictionary(w => w.Key, w => w.Value.Shape, StringComparer.Ordinal);
        CheckTargets(shapes, byKey, lora, options);
        bool reorder = options.InputQkv == DuoQkvInput.HeadInterleaved;
        string[] rewritten = [.. weights.Keys.Where(k => byKey.ContainsKey(k) || (reorder && IsQkvWeight(k))).Order(StringComparer.Ordinal)];
        foreach (string key in rewritten)
        {
            if (weights[key].DType != DType.BF16 && weights[key].DType != DType.F32)
                throw new NotSupportedException($"The Duo LoRA merges into BF16 or F32 weights; '{key}' is {weights[key].DType}. Nothing was changed.");
        }

        List<(string Key, Tensor Tensor)> staged = [];
        try
        {
            foreach (string key in rewritten)
            {
                (Tensor tensor, _) = Materialize(weights[key], IsQkvWeight(key), byKey.GetValueOrDefault(key), lora, options);
                staged.Add((key, tensor));
            }
        }
        catch
        {
            foreach ((_, Tensor tensor) in staged) tensor.Dispose();
            throw;
        }
        foreach ((string key, Tensor tensor) in staged) weights[key] = tensor;
        log?.Invoke($"merged {lora.Modules.Count} LoRA modules into {byKey.Count} weights; rewrote {staged.Count} tensors");
        return [.. staged.Select(s => s.Tensor)];
    }

    /// <summary>Computes the AdaLN tables of <paramref name="times"/> for every block of the backbone in <paramref name="backboneDir"/>, and writes them to <paramref name="outputPath"/> as <c>blocks.N.adaln_proj.table</c> <c>[times * 3, 6 * hidden]</c> F32
    /// plus <c>adaln_times</c> F32 <c>[times]</c>. The shards are read in place; only one block's weight is touched at a time.</summary>
    public static void WriteAdalnTables(string backboneDir, IReadOnlyList<float> times, string outputPath, DuoAdalnPrecision precision, Action<string>? log = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(backboneDir);
        ArgumentNullException.ThrowIfNull(times);
        ArgumentException.ThrowIfNullOrEmpty(outputPath);
        if (times.Count == 0) throw new ArgumentException("At least one time is needed.", nameof(times));

        List<SafeTensorsLoader> loaders = [];
        try
        {
            Dictionary<string, SafeTensorsLoader> owner = new(StringComparer.Ordinal);
            foreach (string shard in ShardFiles(backboneDir))
            {
                SafeTensorsLoader loader = new();
                loader.Load(shard);
                loaders.Add(loader);
                foreach (string name in loader.Descriptors.Keys) owner[name] = loader;
            }

            Tensor Get(string name) => owner.TryGetValue(name, out SafeTensorsLoader? l) ? l.GetTensor(name) : throw new KeyNotFoundException($"The backbone has no tensor '{name}'.");
            Tensor? GetOptional(string name) => owner.TryGetValue(name, out SafeTensorsLoader? l) ? l.GetTensor(name) : null;

            float[] timeArray = [.. times];
            using Tensor w1 = Get("time_embedder.proj_in.weight"), w2 = Get("time_embedder.proj_out.weight");
            using Tensor? b1 = GetOptional("time_embedder.proj_in.bias"), b2 = GetOptional("time_embedder.proj_out.bias");
            float[] embeddings = DuoAdalnTables.TimeEmbeddings(timeArray, w1, b1, w2, b2);
            int embedDim = (int)w2.Shape[0];

            List<int> blocks = [];
            for (int i = 0; owner.ContainsKey($"blocks.{i}.adaln_proj.linear.weight"); i++) blocks.Add(i);
            if (blocks.Count == 0) throw new InvalidDataException("The backbone has no blocks.N.adaln_proj.linear.weight tensors.");

            List<SafeTensorsStreamWriter.Entry> entries = [];
            foreach (int block in blocks)
            {
                string weightKey = $"blocks.{block}.adaln_proj.linear.weight";
                SafeTensorDescriptor descriptor = owner[weightKey].Descriptors[weightKey];
                long outDim = descriptor.Shape[0];
                if (outDim % 3 != 0) throw new InvalidDataException($"'{weightKey}' has {outDim} outputs, not a multiple of the 3 modalities.");
                entries.Add(new SafeTensorsStreamWriter.Entry($"blocks.{block}{BlockAdalnTableSuffix}", DType.F32,
                    new TensorShape(times.Count * 3L, outDim / 3), () =>
                    {
                        using Tensor weight = Get(weightKey);
                        using Tensor? bias = GetOptional($"blocks.{block}.adaln_proj.linear.bias");
                        log?.Invoke($"adaln table block {block}");
                        Tensor table = DuoAdalnTables.Project(embeddings, embedDim, weight, bias, precision,
                            new TensorShape(times.Count * 3L, outDim / 3));
                        return (table, true);
                    }));
            }
            entries.Add(new SafeTensorsStreamWriter.Entry(AdalnTimesKey, DType.F32, new TensorShape(times.Count), () =>
            {
                Tensor t = new(new TensorShape(times.Count), DType.F32);
                timeArray.CopyTo(t.AsSpan<float>());
                return (t, true);
            }));

            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outputPath))!);
            SafeTensorsStreamWriter.Write(outputPath, entries, new Dictionary<string, string>
            {
                ["format"] = "pt",
                ["precision"] = precision.ToString(),
                ["modalities"] = "3",
                ["producer"] = "DuoRepacker",
            });
        }
        finally
        {
            foreach (SafeTensorsLoader loader in loaders) loader.Dispose();
        }
    }

    /// <summary>The LoRA modules grouped by the backbone weight each merges into; a module with no target fails.</summary>
    private static Dictionary<string, List<(string Module, DuoLoraTarget Target)>> ModulesByKey(DuoLoraSource lora)
    {
        Dictionary<string, List<(string Module, DuoLoraTarget Target)>> byKey = new(StringComparer.Ordinal);
        List<string> unmapped = [];
        foreach (string module in lora.Modules)
        {
            if (!DuoLoraKeyMap.TryMap(module, out DuoLoraTarget target)) { unmapped.Add(module); continue; }
            if (!byKey.TryGetValue(target.BackboneKey, out List<(string, DuoLoraTarget)>? list)) byKey[target.BackboneKey] = list = [];
            list.Add((module, target));
        }
        if (unmapped.Count > 0)
            throw new InvalidDataException($"{unmapped.Count} LoRA module(s) have no backbone weight to merge into, for example '{unmapped[0]}'.");
        return byKey;
    }

    /// <summary>Shape checks of every target and of the qkv layout, from the backbone's shapes alone.</summary>
    private static void CheckTargets(Dictionary<string, TensorShape> shapes,
        Dictionary<string, List<(string Module, DuoLoraTarget Target)>> byKey, DuoLoraSource lora, DuoMergeOptions options)
    {
        bool interleaved = options.InputQkv == DuoQkvInput.HeadInterleaved;
        string? bias = interleaved ? shapes.Keys.FirstOrDefault(k => k.EndsWith(".attn.qkv_proj.bias", StringComparison.Ordinal)) : null;
        if (bias is not null)
            throw new InvalidDataException(
                $"'{bias}' is a fused qkv bias in per-head order, which this repack does not reorder; refusing rather than write a mismatched bias. Nothing was written.");

        Dictionary<string, long[]> loraShapes = lora.Describe().ToDictionary(t => t.Key, t => t.Shape, StringComparer.Ordinal);
        foreach ((string key, List<(string Module, DuoLoraTarget Target)> modules) in byKey)
        {
            if (!shapes.TryGetValue(key, out TensorShape weight))
                throw new InvalidDataException(
                    $"LoRA module '{modules[0].Module}' targets '{key}', which no backbone shard holds; the shards are not the Duo backbone. Nothing was written.");
            if (weight.Rank != 2)
                throw new InvalidDataException($"'{key}' must be a matrix, got {weight}. Nothing was written.");
            foreach ((string module, DuoLoraTarget target) in modules)
            {
                long[] a = loraShapes[module + DuoLoraKeyMap.LoraASuffix], b = loraShapes[module + DuoLoraKeyMap.LoraBSuffix];
                bool fits = a.Length == 2 && b.Length == 2 && a[0] == b[1] && a[1] == weight[1]
                    && weight[0] % target.Slots == 0 && b[0] == weight[0] / target.Slots;
                if (!fits)
                    throw new InvalidDataException(
                        $"LoRA module '{module}' (A [{string.Join(", ", a)}], B [{string.Join(", ", b)}]) does not fit '{key}' {weight}; "
                        + $"expected B with {weight[0] / target.Slots} rows and A with {weight[1]} columns. Nothing was written.");
            }
        }

        if (!interleaved)
        {
            return;
        }
        foreach (string key in shapes.Keys.Where(IsQkvWeight))
        {
            TensorShape weight = shapes[key];
            if (weight.Rank != 2 || weight[0] != 3L * options.Heads * options.HeadDim)
                throw new InvalidDataException(
                    $"'{key}' is {weight}, not 3 * {options.Heads} heads * {options.HeadDim} rows; check --heads and --head-dim. Nothing was written.");
            string norm = key[..^"qkv_proj.weight".Length] + "q_norm.weight";
            if (shapes.TryGetValue(norm, out TensorShape qNorm) && (qNorm.Rank != 1 || qNorm[0] != options.HeadDim))
                throw new InvalidDataException(
                    $"'{norm}' has {qNorm} elements, not the head dim {options.HeadDim}; check --heads and --head-dim. Nothing was written.");
        }
    }

    private static bool IsQkvWeight(string name) => name.EndsWith(".attn.qkv_proj.weight", StringComparison.Ordinal);

    private static unsafe (Tensor Tensor, bool Owned) Materialize(Tensor source, bool qkv, List<(string Module, DuoLoraTarget Target)>? modules,
        DuoLoraSource lora, DuoMergeOptions options)
    {
        Tensor work;
        if (qkv && options.InputQkv == DuoQkvInput.HeadInterleaved)
        {
            work = DuoQkvLayout.ToBlockOrder(source, options.Heads, options.HeadDim);
        }
        else if (modules is not null)
        {
            work = new Tensor(source.Shape, source.DType);
            Buffer.MemoryCopy((void*)source.DataPointer, (void*)work.DataPointer, work.ElementCount * work.DType.SizeInBytes, work.ElementCount * work.DType.SizeInBytes);
        }
        else
        {
            return (source, false);
        }

        try
        {
            if (modules is not null)
            {
                foreach ((string module, DuoLoraTarget target) in modules)
                {
                    (Tensor a, Tensor b) = lora.Read(module);
                    try
                    {
                        long slotRows = work.Shape[0] / target.Slots;
                        if (work.Shape[0] % target.Slots != 0 || b.Shape[0] != slotRows)
                            throw new InvalidDataException($"LoRA module '{module}' has {b.Shape[0]} rows but its slot of '{target.BackboneKey}' has {slotRows}.");
                        float scale = (float)((options.Alpha ?? a.Shape[0]) / a.Shape[0]);
                        DuoLoraMerge.MergeRows(work, target.Slot * slotRows, a, b, scale, target.SwapHalves);
                    }
                    finally
                    {
                        a.Dispose();
                        b.Dispose();
                    }
                }
            }
            return (work, true);
        }
        catch
        {
            work.Dispose();
            throw;
        }
    }

    private static bool IsBlockAdalnProj(string name) => name.StartsWith("blocks.", StringComparison.Ordinal) && name.Contains(".adaln_proj.", StringComparison.Ordinal);

    private static string[] ShardFiles(string dir)
    {
        if (!Directory.Exists(dir)) throw new DirectoryNotFoundException($"Backbone folder not found: {dir}");
        string[] shards = Directory.GetFiles(dir, "*.safetensors", SearchOption.TopDirectoryOnly);
        if (shards.Length == 0) throw new FileNotFoundException($"No safetensors shards in {dir}.");
        Array.Sort(shards, StringComparer.Ordinal);
        return shards;
    }

    private static void WriteIndex(string path, Dictionary<string, string> weightMap, long totalSize)
    {
        using FileStream stream = File.Create(path);
        using Utf8JsonWriter json = new(stream, new JsonWriterOptions { Indented = true });
        json.WriteStartObject();
        json.WriteStartObject("metadata");
        json.WriteNumber("total_size", totalSize);
        json.WriteEndObject();
        json.WriteStartObject("weight_map");
        foreach (KeyValuePair<string, string> entry in weightMap.OrderBy(static e => e.Key, StringComparer.Ordinal)) json.WriteString(entry.Key, entry.Value);
        json.WriteEndObject();
        json.WriteEndObject();
    }
}
