using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Gguf;

namespace HartsyInference.Audio.Models.Haru;

/// <summary>Turns the Haru checkpoint into the weights <see cref="HaruPipeline"/> loads at a chosen <see cref="HaruPrecision"/>. Weights
/// used as stored are borrowed from the mapped checkpoint; the language model's matrices that change go through a one-time disk cache
/// (<see cref="GgufQuantCache"/>), and the reported re-quantization error says what the conversion cost.</summary>
public static class HaruWeights
{
    /// <summary>Bumped whenever the cached layout or the policy changes, so an older cache is never read as a newer one.</summary>
    public const int CacheVersion = 1;

    private static readonly string[] AttentionProjections =
    [
        ".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight", ".self_attn.o_proj.weight",
        ".linear_attn.in_proj_qkv.weight", ".linear_attn.in_proj_z.weight", ".linear_attn.out_proj.weight",
    ];

    private static readonly string[] FeedForwardProjections = [".mlp.gate_proj.weight", ".mlp.up_proj.weight", ".mlp.down_proj.weight"];

    /// <summary>Parses a precision name: native (or empty), q8 / q8_0, q6 / q6_k, q5 / q5_k, q4 / q4_k.</summary>
    public static bool TryParsePrecision(string? name, out HaruPrecision precision)
    {
        HaruPrecision? parsed = name?.Trim().ToLowerInvariant() switch
        {
            null or "" or "native" or "default" or "bf16" => HaruPrecision.Native,
            "q8" or "q8_0" => HaruPrecision.Q8_0,
            "q6" or "q6_k" => HaruPrecision.Q6_K,
            "q5" or "q5_k" or "q5_k_m" => HaruPrecision.Q5_K,
            "q4" or "q4_k" or "q4_k_m" => HaruPrecision.Q4_K,
            _ => null,
        };
        precision = parsed ?? HaruPrecision.Native;
        return parsed is not null;
    }

    /// <summary>The dtype <paramref name="weight"/> is loaded as at <paramref name="precision"/>.</summary>
    public static DType TargetDType(HfQuantWeight weight, HaruPrecision precision)
    {
        ArgumentNullException.ThrowIfNull(weight);
        if (weight.Storage != HfWeightStorage.Plain)
        {
            throw new NotSupportedException($"Haru '{weight.Name}' is stored as {weight.Storage}; the checkpoint is plain BF16.");
        }
        bool layer = weight.Name.StartsWith("lm.layers.", StringComparison.Ordinal);
        bool attention = layer && Array.Exists(AttentionProjections, suffix => weight.Name.EndsWith(suffix, StringComparison.Ordinal));
        bool feedForward = layer && Array.Exists(FeedForwardProjections, suffix => weight.Name.EndsWith(suffix, StringComparison.Ordinal));
        if (precision == HaruPrecision.Native || !(attention || feedForward))
        {
            return weight.StoredDType;
        }
        DType target = precision switch
        {
            HaruPrecision.Q8_0 => DType.Q8_0,
            HaruPrecision.Q6_K => DType.Q6_K,
            HaruPrecision.Q5_K => attention ? DType.Q6_K : DType.Q5_K,
            HaruPrecision.Q4_K => attention ? DType.Q6_K : DType.Q4_K,
            _ => throw new ArgumentOutOfRangeException(nameof(precision)),
        };
        // A block quant never spans rows: a row that is not a whole number of blocks falls back to Q8_0, then to the stored dtype.
        long inDim = weight.Shape[weight.Shape.Rank - 1];
        if (target.IsQuantized && inDim % target.BlockElementCount != 0)
        {
            target = inDim % DType.Q8_0.BlockElementCount == 0 ? DType.Q8_0 : weight.StoredDType;
        }
        return target;
    }

    /// <summary>The weights at <paramref name="precision"/>: borrowed from <paramref name="checkpoint"/> where nothing changes, the rest
    /// mapped from the cache at <paramref name="cachePath"/> (written first when missing). The returned owner keeps whatever backs the
    /// weights alive; dispose it with the model. On success the checkpoint belongs to the owner.</summary>
    public static (IReadOnlyDictionary<string, Tensor> Weights, IDisposable Owner) Load(HfQuantCheckpoint checkpoint, HaruPrecision precision, string cachePath)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        Dictionary<string, object> identity = new(StringComparer.Ordinal)
        {
            ["hartsy.haru.cache_version"] = CacheVersion,
            ["hartsy.haru.precision"] = precision.ToString(),
        };
        return HfQuantWeightCache.Load(checkpoint, w => TargetDType(w, precision), cachePath, "haru", "[Audio][Haru]", identity,
            produceAhead: GgufQuantCache.ParallelProduceAhead);
    }
}
