using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>A LongCat-Video DiT checkpoint folder: the diffusers-named sharded safetensors index (bf16 or F32 shards, reference key names) and its
/// <c>config.json</c>. Tensors are borrowed views of the shard mappings and live until this is disposed. The INT8 layout
/// (<c>.weight_int8</c> keys) is a
/// separate loader.</summary>
public sealed class LongCatCheckpoint : IDisposable
{
    /// <summary>The index file the base and avatar repositories ship.</summary>
    public const string IndexFileName = "diffusion_pytorch_model.safetensors.index.json";

    private readonly ShardedSafeTensorSet _shards;

    private LongCatCheckpoint(ShardedSafeTensorSet shards, LongCatConfig config, Dictionary<string, Tensor> weights)
    {
        _shards = shards;
        Config = config;
        Weights = weights;
    }

    /// <summary>The geometry read from <c>config.json</c>, or derived from the tensor shapes when the folder has none.</summary>
    public LongCatConfig Config { get; }

    /// <summary>Every tensor by checkpoint key.</summary>
    public IReadOnlyDictionary<string, Tensor> Weights { get; }

    /// <summary>Opens <paramref name="directory"/>, checks every expected key and shape and every dtype, and maps the tensors.</summary>
    public static LongCatCheckpoint Open(string directory)
    {
        ShardedSafeTensorSet shards = ShardedSafeTensorSet.OpenIndex(directory, null, IndexFileName);
        try
        {
            Dictionary<string, TensorShape> shapes = new(shards.Inventory.Count, StringComparer.Ordinal);
            foreach ((string name, TensorLocation location) in shards.Inventory)
                shapes[name] = location.Shape;
            if (shapes.Keys.Any(static key => key.EndsWith(".weight_int8", StringComparison.Ordinal)))
                throw new NotSupportedException(
                    $"'{directory}' is the per-row INT8 layout, which its own loader reads; point this at the bf16 or F32 folder.");

            string configPath = Path.Combine(directory, "config.json");
            LongCatConfig config = File.Exists(configPath)
                ? LongCatConfig.FromJson(File.ReadAllText(configPath))
                : LongCatConfig.FromShapes(name => shapes.TryGetValue(name, out TensorShape shape) ? shape : null, CountBlocks(shapes),
                    (1, 2, 2), true);
            Validate(config, shapes);

            Dictionary<string, Tensor> weights = new(shards.Inventory.Count, StringComparer.Ordinal);
            foreach ((string name, TensorLocation location) in shards.Inventory)
            {
                if (location.DType != DType.BF16 && location.DType != DType.F16 && location.DType != DType.F32)
                    throw new NotSupportedException($"'{name}' is {location.DType}; the LongCat loader reads BF16, F16 and F32 tensors.");
                weights[name] = shards.GetTensor(name);
            }
            return new LongCatCheckpoint(shards, config, weights);
        }
        catch
        {
            shards.Dispose();
            throw;
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

    /// <summary>Unmaps the shards; tensors from <see cref="Weights"/> are invalid afterwards.</summary>
    public void Dispose() => _shards.Dispose();
}
