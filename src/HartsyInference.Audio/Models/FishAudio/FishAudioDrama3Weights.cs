using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Gguf;

namespace HartsyInference.Audio.Models.FishAudio;

/// <summary>Turns a Drama 3 checkpoint — the unquantized original, the block-FP8 release or the GPTQ int4 one — into the weights
/// <see cref="FishAudioDrama3Lm"/> loads at a chosen <see cref="FishAudioDrama3Precision"/>. When every weight is used as stored
/// they are borrowed from the mapped checkpoint; otherwise they go through a one-time disk cache (<see cref="GgufQuantCache"/>)
/// written one tensor at a time, and the reported re-quantization error says what the conversion cost.</summary>
public static class FishAudioDrama3Weights
{
    /// <summary>Bumped whenever the cached layout or policy changes, so an older cache is never read as a newer one.</summary>
    public const int CacheVersion = 1;

    private static readonly string[] DenseProjections =
    [
        ".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight", ".self_attn.o_proj.weight",
        ".linear_attn.in_proj_qkv.weight", ".linear_attn.in_proj_z.weight", ".linear_attn.out_proj.weight",
        ".mlp.shared_expert.gate_proj.weight", ".mlp.shared_expert.up_proj.weight", ".mlp.shared_expert.down_proj.weight",
    ];

    /// <summary>Parses a precision name: native (or empty), bf16, q8 / q8_0, q6 / q6_k, q5 / q5_k, q4 / q4_k.</summary>
    /// <exception cref="ArgumentException">Any other name.</exception>
    public static FishAudioDrama3Precision ParsePrecision(string? name) => TryParsePrecision(name, out FishAudioDrama3Precision precision)
        ? precision
        : throw new ArgumentException($"Unknown Drama 3 precision '{name}'; expected native, bf16, q8, q6, q5 or q4.", nameof(name));

    /// <summary>Parses a precision name as <see cref="ParsePrecision"/> does; false for any other name.</summary>
    public static bool TryParsePrecision(string? name, out FishAudioDrama3Precision precision)
    {
        FishAudioDrama3Precision? parsed = name?.Trim().ToLowerInvariant() switch
        {
            null or "" or "native" or "default" => FishAudioDrama3Precision.Native,
            "bf16" or "off" or "none" => FishAudioDrama3Precision.Bf16,
            "q8" or "q8_0" => FishAudioDrama3Precision.Q8_0,
            "q6" or "q6_k" => FishAudioDrama3Precision.Q6_K,
            "q5" or "q5_k" or "q5_k_m" => FishAudioDrama3Precision.Q5_K,
            "q4" or "q4_k" or "q4_k_m" => FishAudioDrama3Precision.Q4_K,
            _ => null,
        };
        precision = parsed ?? FishAudioDrama3Precision.Native;
        return parsed is not null;
    }

    /// <summary>The dtype <paramref name="weight"/> is loaded as at <paramref name="precision"/>.</summary>
    public static DType TargetDType(HfQuantWeight weight, FishAudioDrama3Precision precision)
    {
        ArgumentNullException.ThrowIfNull(weight);
        bool expert = weight.Name.Contains(".mlp.experts.", StringComparison.Ordinal);
        bool dense = Array.Exists(DenseProjections, suffix => weight.Name.EndsWith(suffix, StringComparison.Ordinal));
        bool attention = dense && !weight.Name.Contains(".shared_expert.", StringComparison.Ordinal);
        if (!expert && !dense)
        {
            return weight.Storage == HfWeightStorage.Plain ? weight.StoredDType
                : throw new NotSupportedException($"Drama 3 '{weight.Name}' is stored as {weight.Storage}, which only expert and projection weights may be.");
        }
        DType target = precision switch
        {
            FishAudioDrama3Precision.Native => weight.Storage switch
            {
                HfWeightStorage.Fp8Block => DType.Q8_0,
                HfWeightStorage.GptqInt4 => DType.Q4_0,
                _ => weight.StoredDType,
            },
            FishAudioDrama3Precision.Bf16 => DType.BF16,
            FishAudioDrama3Precision.Q8_0 => DType.Q8_0,
            FishAudioDrama3Precision.Q6_K => DType.Q6_K,
            FishAudioDrama3Precision.Q5_K => attention ? DType.Q6_K : DType.Q5_K,
            FishAudioDrama3Precision.Q4_K => attention ? DType.Q6_K : DType.Q4_K,
            _ => throw new ArgumentOutOfRangeException(nameof(precision)),
        };
        // A block quant never spans rows: a row that is not a whole number of blocks falls back to Q8_0, then to the stored float
        // dtype (BF16 for a weight that has none).
        long inDim = weight.Shape[weight.Shape.Rank - 1];
        if (target.IsQuantized && inDim % target.BlockElementCount != 0)
        {
            target = inDim % DType.Q8_0.BlockElementCount == 0 ? DType.Q8_0
                : weight.Storage == HfWeightStorage.Plain ? weight.StoredDType : DType.BF16;
        }
        return target;
    }

    /// <summary>True when every weight is used exactly as stored, so nothing needs converting.</summary>
    public static bool UsesStoredWeights(HfQuantCheckpoint checkpoint, FishAudioDrama3Precision precision)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        foreach (HfQuantWeight weight in checkpoint.Weights.Values)
        {
            if (weight.Storage != HfWeightStorage.Plain || TargetDType(weight, precision) != weight.StoredDType) return false;
        }
        return true;
    }

    /// <summary>The checkpoint's weights as stored, borrowed from its mapping (which must outlive them); every weight must be plain.</summary>
    public static Dictionary<string, Tensor> Borrow(HfQuantCheckpoint checkpoint)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        Dictionary<string, Tensor> weights = new(checkpoint.Weights.Count, StringComparer.Ordinal);
        foreach (string name in checkpoint.Weights.Keys) weights[name] = checkpoint.Borrow(name);
        return weights;
    }

    /// <summary>One lazy cache entry per weight at <paramref name="precision"/>, quantizations folding their error into <paramref name="error"/>.</summary>
    public static IEnumerable<GgufCacheEntry> CacheEntries(HfQuantCheckpoint checkpoint, FishAudioDrama3Precision precision, GgufQuantError error)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        ArgumentNullException.ThrowIfNull(error);
        foreach ((string name, HfQuantWeight weight) in checkpoint.Weights.OrderBy(static w => w.Key, StringComparer.Ordinal))
        {
            DType target = TargetDType(weight, precision);
            yield return new GgufCacheEntry(name, weight.Shape, target, () => checkpoint.Produce(name, target, error));
        }
    }

    /// <summary>The weights at <paramref name="precision"/>: borrowed from <paramref name="checkpoint"/> when nothing changes, else mapped
    /// from the cache at <paramref name="cachePath"/>, written first when missing. The returned owner keeps whichever mapping backs
    /// the weights alive; dispose it with the model.</summary>
    public static (IReadOnlyDictionary<string, Tensor> Weights, IDisposable Owner) Load(HfQuantCheckpoint checkpoint,
        FishAudioDrama3Precision precision, string cachePath)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        if (UsesStoredWeights(checkpoint, precision))
        {
            return (Borrow(checkpoint), checkpoint);
        }
        GgufQuantError error = new();
        Dictionary<string, object> identity = new(StringComparer.Ordinal)
        {
            ["hartsy.drama3.cache_version"] = CacheVersion,
            ["hartsy.drama3.precision"] = precision.ToString(),
            ["hartsy.drama3.source"] = checkpoint.SourceIdentity,
        };
        GgufLoader loader = GgufQuantCache.Open(cachePath, () => CacheEntries(checkpoint, precision, error), "fish-drama3",
            "[Audio][Drama3]", identity);
        if (error.Count > 0)
        {
            Logs.Info($"[Audio][Drama3] Re-quantization to {precision}: {error}.");
        }
        try
        {
            Dictionary<string, Tensor> weights = GgufQuantCache.ReadBack(loader, CacheEntries(checkpoint, precision, error));
            checkpoint.Dispose();
            return (weights, loader);
        }
        catch
        {
            loader.Dispose();
            throw;
        }
    }
}
