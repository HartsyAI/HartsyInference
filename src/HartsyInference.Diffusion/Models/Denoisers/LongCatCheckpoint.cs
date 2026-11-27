using HartsyInference.Core.Configuration;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>A LongCat-Video DiT checkpoint folder: a sharded safetensors index with reference key names, either dense (bf16 or F32, the
/// diffusers-named index) or the per-row INT8 release (<c>quantized_model.safetensors.index.json</c>, <c>.weight_int8</c> plus
/// <c>.weight_scale</c>), or, alone in a folder of its own, one GGUF repack in the ComfyUI Wan layout
/// (<see cref="LongCatGgufConverter"/>); and its <c>config.json</c>, derived from the shapes when absent. Tensors are borrowed views of the
/// file mappings, or copies this owns, and live until this is disposed. INT8 weights come back under the dense key names as I8 tensors
/// carrying their row scale (<see cref="LongCatInt8Converter"/>); GGUF block quants stay packed.</summary>
public sealed class LongCatCheckpoint : IDisposable
{
    /// <summary>The index file the dense base and avatar repositories ship.</summary>
    public const string IndexFileName = "diffusion_pytorch_model.safetensors.index.json";

    /// <summary>The index file of the per-row INT8 release.</summary>
    public const string Int8IndexFileName = "quantized_model.safetensors.index.json";

    private readonly IDisposable _files;

    private LongCatCheckpoint(IDisposable files, LongCatConfig config, Dictionary<string, Tensor> weights, bool isInt8,
        List<Tensor> converted)
    {
        _files = files;
        Config = config;
        Weights = weights;
        IsInt8 = isInt8;
        _converted = converted;
    }

    private readonly List<Tensor> _converted;

    /// <summary>True for the per-row INT8 release: its linears are I8 tensors carrying <see cref="Tensor.QuantInfo"/>. Those need
    /// <c>QuantizedWeightPolicy.PrepareForBackends</c> (or a backend with a resident int8 path) before a forward.</summary>
    public bool IsInt8 { get; }

    /// <summary>The geometry read from <c>config.json</c>, or derived from the tensor shapes when the folder has none.</summary>
    public LongCatConfig Config { get; }

    /// <summary>Every tensor by checkpoint key.</summary>
    public IReadOnlyDictionary<string, Tensor> Weights { get; }

    /// <summary>An F32 tensor as BF16, each value rounded to nearest, ties to even (<see cref="BFloat16Bits"/>), as the reference's bf16 load rounds it.
    /// <c>Tensor.CastTo</c> truncates, which is not what the reference does.</summary>
    internal static Tensor ToBf16RoundToNearest(Tensor source)
    {
        ArgumentNullException.ThrowIfNull(source);
        Tensor bf16 = new(source.Shape, DType.BF16);
        // A tensor's host buffer is allocated lazily by its first AsSpan/DataPointer (Tensor's contract), so touch both serially here: if
        // the workers made that first call, two of them could each allocate a buffer and the conversion would land in a discarded one.
        _ = source.AsReadOnlySpan<float>();
        _ = bf16.AsSpan<ushort>();
        // Each range writes only its own elements, so the result is the same at any worker count (CpuParallel.ForRanges' contract).
        // The conversion is elementwise, so the ranges can be any size; 2^18 elements keeps each range a few hundred microseconds.
        CpuParallel.ForRanges(source.ElementCount, RangeElements, workPerElement: 4, (source, bf16), static (start, length, state) =>
        {
            ReadOnlySpan<float> values = state.source.AsReadOnlySpan<float>().Slice((int)start, (int)length);
            Span<ushort> bits = state.bf16.AsSpan<ushort>().Slice((int)start, (int)length);
            for (int i = 0; i < values.Length; i++)
                bits[i] = BFloat16Bits.FromFloat(values[i]);
        });
        return bf16;
    }

    /// <summary>Elements per parallel range of the BF16 conversion.</summary>
    internal const int RangeElements = 1 << 18;

    /// <summary>The index <see cref="Open"/> reads in <paramref name="directory"/>: the dense one when both are present, the INT8 one only in a
    /// folder that has no dense one.</summary>
    public static string IndexNameIn(string directory) =>
        File.Exists(Path.Combine(directory, IndexFileName)) || !File.Exists(Path.Combine(directory, Int8IndexFileName))
            ? IndexFileName : Int8IndexFileName;

    /// <summary>Opens <paramref name="directory"/>, checks every expected key and shape and every dtype, and maps the tensors, from the index
    /// <see cref="IndexNameIn"/> names.</summary>
    public static LongCatCheckpoint Open(string directory)
    {
        if (GgufIn(directory) is { } gguf)
            return OpenGguf(directory, gguf);
        ShardedSafeTensorSet shards = ShardedSafeTensorSet.OpenIndex(directory, null, IndexNameIn(directory));
        try
        {
            bool int8 = LongCatInt8Converter.IsInt8Layout(shards.Inventory.Keys);
            Dictionary<string, TensorShape> shapes = new(shards.Inventory.Count, StringComparer.Ordinal);
            foreach ((string name, TensorLocation location) in shards.Inventory)
            {
                if (int8 && name.EndsWith(LongCatInt8Converter.ScaleSuffix, StringComparison.Ordinal))
                {
                    string quantized = name[..^LongCatInt8Converter.ScaleSuffix.Length] + LongCatInt8Converter.QuantizedSuffix;
                    if (!shards.Inventory.ContainsKey(quantized))
                        throw new InvalidDataException($"'{name}' has no '{quantized}'; an orphan dequant scale is refused.");
                    continue;
                }
                shapes[int8 && name.EndsWith(LongCatInt8Converter.QuantizedSuffix, StringComparison.Ordinal)
                    ? name[..^LongCatInt8Converter.QuantizedSuffix.Length] + ".weight" : name] = location.Shape;
            }

            string configPath = Path.Combine(directory, "config.json");
            LongCatConfig config = File.Exists(configPath)
                ? LongCatConfig.FromJson(File.ReadAllText(configPath))
                : LongCatConfig.FromShapes(name => shapes.TryGetValue(name, out TensorShape shape) ? shape : null, CountBlocks(shapes),
                    (1, 2, 2), true);
            Validate(config, shapes);

            Dictionary<string, Tensor> weights = new(shards.Inventory.Count, StringComparer.Ordinal);
            foreach ((string name, TensorLocation location) in shards.Inventory)
            {
                bool quantized = int8 && name.EndsWith(LongCatInt8Converter.QuantizedSuffix, StringComparison.Ordinal);
                if (location.DType != DType.BF16 && location.DType != DType.F16 && location.DType != DType.F32
                    && !(quantized && location.DType == DType.I8))
                    throw new NotSupportedException($"'{name}' is {location.DType}; the LongCat loader reads BF16, F16 and F32 tensors"
                        + (int8 ? " and I8 .weight_int8 weights." : "."));
                weights[name] = shards.GetTensor(name);
            }
            List<Tensor> converted = [];
            if (!int8)
                HoldDenseAsBf16(weights, converted);
            return new LongCatCheckpoint(shards, config, int8 ? LongCatInt8Converter.Convert(weights) : weights, int8, converted);
        }
        catch
        {
            shards.Dispose();
            throw;
        }
    }

    /// <summary>The GGUF file <see cref="Open"/> reads from <paramref name="directory"/>: its one <c>.gguf</c>, or null for a folder without one.
    /// A folder with several, or with one beside a safetensors index, is refused: a caller that named a file reaches here with its folder, so
    /// picking would load weights other than the named ones.</summary>
    public static string? GgufIn(string directory)
    {
        ArgumentNullException.ThrowIfNull(directory);
        if (!Directory.Exists(directory))
            return null;
        string[] files = Directory.GetFiles(directory, "*.gguf");
        if (files.Length > 0 && (File.Exists(Path.Combine(directory, IndexFileName)) || File.Exists(Path.Combine(directory, Int8IndexFileName))))
            throw new InvalidDataException($"'{directory}' holds both a safetensors index and {Path.GetFileName(files[0])}; put the GGUF in a "
                + "folder of its own.");
        if (files.Length > 1)
            throw new InvalidDataException($"'{directory}' holds {files.Length} .gguf files "
                + $"({string.Join(", ", files.Select(Path.GetFileName).Order(StringComparer.Ordinal))}); put the one to load in a "
                + "folder of its own.");
        return files.Length == 1 ? files[0] : null;
    }

    private static LongCatCheckpoint OpenGguf(string directory, string path)
    {
        CheckpointSource source = CheckpointSource.Open(path);
        List<Tensor> converted = [];
        try
        {
            if (!LongCatGgufConverter.IsComfyLayout(source.Weights.Keys))
                throw new InvalidDataException($"'{Path.GetFileName(path)}' is not a LongCat DiT in the ComfyUI Wan layout "
                    + "(blocks.0.self_attn.q.weight beside blocks.0.ffn.w1.weight).");
            Dictionary<string, Tensor> weights = LongCatGgufConverter.Convert(source.Weights, converted, out int widened);
            foreach ((string name, Tensor tensor) in weights)
                if (tensor.DType != DType.BF16 && tensor.DType != DType.F16 && tensor.DType != DType.F32 && !tensor.DType.IsQuantized)
                    throw new NotSupportedException($"'{name}' is {tensor.DType}; the LongCat GGUF loader reads block quants, BF16, F16 and F32.");
            if (widened > 0)
                Logs.Info($"[LongCatCheckpoint] {Path.GetFileName(path)}: {widened} projections join parts of different quant types, so they are "
                    + "held as BF16 instead of packed.");
            Dictionary<string, TensorShape> shapes = weights.ToDictionary(entry => entry.Key, entry => entry.Value.Shape, StringComparer.Ordinal);
            string configPath = Path.Combine(directory, "config.json");
            LongCatConfig config = File.Exists(configPath)
                ? LongCatConfig.FromJson(File.ReadAllText(configPath))
                : LongCatConfig.FromShapes(name => shapes.TryGetValue(name, out TensorShape shape) ? shape : null, CountBlocks(shapes),
                    (1, 2, 2), true);
            Validate(config, shapes);
            HoldDenseAsBf16(weights, converted);
            return new LongCatCheckpoint(source, config, weights, false, converted);
        }
        catch
        {
            foreach (Tensor tensor in converted) tensor.Dispose();
            source.Dispose();
            throw;
        }
    }

    /// <summary>The reference loads the DiT in BF16 (torch_dtype=bfloat16). A dense F32 tensor is converted here, so a 54.3 GB checkpoint does not
    /// sit in host memory at F32 and stream at twice the bytes; the converted tensors join <paramref name="converted"/>, which the checkpoint
    /// owns.</summary>
    private static void HoldDenseAsBf16(Dictionary<string, Tensor> weights, List<Tensor> converted)
    {
        if (!EngineKnobs.LongCatDitBf16.Value)
            return;
        foreach (string name in weights.Keys.ToList())
        {
            Tensor source = weights[name];
            if (source.DType != DType.F32)
                continue;
            Tensor bf16 = ToBf16RoundToNearest(source);
            converted.Add(bf16);
            weights[name] = bf16;
        }
    }

    /// <summary>Every tensor the DiT reads, with its shape, for <paramref name="config"/>.</summary>
    public static Dictionary<string, TensorShape> ExpectedShapes(LongCatConfig config)
    {
        int h = config.HiddenSize, f = config.FfnHiddenDim, a = config.AdalnTembedDim, headDim = config.HeadDim;
        (int pt, int ph, int pw) = config.PatchSize;
        Dictionary<string, TensorShape> shapes = new(StringComparer.Ordinal)
        {
            ["x_embedder.proj.weight"] = new TensorShape([h, config.InChannels, pt, ph, pw]),
            ["x_embedder.proj.bias"] = new TensorShape(h),
            ["t_embedder.mlp.0.weight"] = new TensorShape(a, config.FrequencyEmbeddingSize),
            ["t_embedder.mlp.0.bias"] = new TensorShape(a),
            ["t_embedder.mlp.2.weight"] = new TensorShape(a, a),
            ["t_embedder.mlp.2.bias"] = new TensorShape(a),
            ["y_embedder.y_proj.0.weight"] = new TensorShape(h, config.CaptionChannels),
            ["y_embedder.y_proj.0.bias"] = new TensorShape(h),
            ["y_embedder.y_proj.2.weight"] = new TensorShape(h, h),
            ["y_embedder.y_proj.2.bias"] = new TensorShape(h),
            ["final_layer.adaLN_modulation.1.weight"] = new TensorShape(2 * h, a),
            ["final_layer.adaLN_modulation.1.bias"] = new TensorShape(2 * h),
            ["final_layer.linear.weight"] = new TensorShape(pt * ph * pw * config.OutChannels, h),
            ["final_layer.linear.bias"] = new TensorShape(pt * ph * pw * config.OutChannels),
        };
        for (int i = 0; i < config.Depth; i++)
        {
            string p = $"blocks.{i}.";
            shapes[p + "adaLN_modulation.1.weight"] = new TensorShape(6 * h, a);
            shapes[p + "adaLN_modulation.1.bias"] = new TensorShape(6 * h);
            shapes[p + "attn.qkv.weight"] = new TensorShape(3 * h, h);
            shapes[p + "attn.qkv.bias"] = new TensorShape(3 * h);
            shapes[p + "attn.q_norm.weight"] = new TensorShape(headDim);
            shapes[p + "attn.k_norm.weight"] = new TensorShape(headDim);
            shapes[p + "attn.proj.weight"] = new TensorShape(h, h);
            shapes[p + "attn.proj.bias"] = new TensorShape(h);
            shapes[p + "pre_crs_attn_norm.weight"] = new TensorShape(h);
            shapes[p + "pre_crs_attn_norm.bias"] = new TensorShape(h);
            shapes[p + "cross_attn.q_linear.weight"] = new TensorShape(h, h);
            shapes[p + "cross_attn.q_linear.bias"] = new TensorShape(h);
            shapes[p + "cross_attn.kv_linear.weight"] = new TensorShape(2 * h, h);
            shapes[p + "cross_attn.kv_linear.bias"] = new TensorShape(2 * h);
            shapes[p + "cross_attn.q_norm.weight"] = new TensorShape(headDim);
            shapes[p + "cross_attn.k_norm.weight"] = new TensorShape(headDim);
            shapes[p + "cross_attn.proj.weight"] = new TensorShape(h, h);
            shapes[p + "cross_attn.proj.bias"] = new TensorShape(h);
            shapes[p + "ffn.w1.weight"] = new TensorShape(f, h);
            shapes[p + "ffn.w2.weight"] = new TensorShape(h, f);
            shapes[p + "ffn.w3.weight"] = new TensorShape(f, h);
        }
        return shapes;
    }

    /// <summary>Throws, naming the first offenders, unless every expected tensor is present with its shape. Extra tensors (the
    /// avatar's audio keys) are allowed.</summary>
    public static void Validate(LongCatConfig config, IReadOnlyDictionary<string, TensorShape> actual)
    {
        List<string> problems = [];
        foreach ((string key, TensorShape expected) in ExpectedShapes(config))
        {
            if (!actual.TryGetValue(key, out TensorShape shape))
                problems.Add($"missing {key}");
            else if (!shape.Equals(expected))
                problems.Add($"{key} is {shape}, expected {expected}");
            if (problems.Count >= 8)
                break;
        }
        if (problems.Count > 0)
            throw new InvalidDataException(
                $"The checkpoint does not match the LongCat DiT geometry ({config.Depth} blocks, hidden {config.HiddenSize}): "
                + $"{string.Join("; ", problems)}.");
    }

    /// <summary>Blocks present: the highest <c>blocks.N</c> index plus one.</summary>
    private static int CountBlocks(IReadOnlyDictionary<string, TensorShape> shapes)
    {
        int count = 0;
        foreach (string key in shapes.Keys)
        {
            if (!key.StartsWith("blocks.", StringComparison.Ordinal))
                continue;
            int end = key.IndexOf('.', 7);
            if (end > 7 && int.TryParse(key.AsSpan(7, end - 7), out int index))
                count = Math.Max(count, index + 1);
        }
        return count;
    }

    /// <summary>Unmaps the files; tensors from <see cref="Weights"/> are invalid afterwards.</summary>
    public void Dispose()
    {
        foreach (Tensor tensor in _converted) tensor.Dispose();
        _files.Dispose();
    }
}
