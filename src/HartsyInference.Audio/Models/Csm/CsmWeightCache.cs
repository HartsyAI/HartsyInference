using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.Gguf;

namespace HartsyInference.Audio.Models.Csm;

/// <summary>Disk-cached quantization for the HeartMuLa / Sesame CSM language model. The AR decode is
/// memory-bandwidth-bound (per-token weight streaming dominates), so quantizing the projection + head matrices lets
/// the engine's fused GEMV read 2–4× fewer bytes/token (Q8 ≈ 2×, Q4 ≈ 3.6×) — and it shrinks the on-disk + VRAM
/// footprint. Quantization is done ONCE to a GGUF cache and loaded from there forever after: converting on every
/// load would transiently hold the BF16 model + a quantized copy + F32 conversion buffers (~12 GB for the 3B) and
/// OOM the host. The conversion streams tensor-by-tensor from the source mmap, so its peak is one tensor's F32
/// buffer; the read-back mmaps the (small) quantized GGUF directly, never materializing the BF16 weights.</summary>
public static class CsmWeightCache
{
    /// <summary>Resolves a quant mode string (q8_0 / q4_k / q5_k / q6_k, plus aliases) to a backbone DType, or null
    /// (unset / unrecognized → no quantization).</summary>
    public static DType? ResolveQuant(string? mode)
    {
        return mode?.Trim().ToLowerInvariant() switch
        {
            "q8_0" or "q8" => DType.Q8_0,
            "q4_k" or "q4" or "q4_k_m" or "q4_k_s" => DType.Q4_K,
            "q5_k" or "q5" or "q5_k_m" => DType.Q5_K,
            "q6_k" or "q6" => DType.Q6_K,
            _ => null,
        };
    }

    /// <summary>Loads the CSM/HeartMuLa weights quantized from a disk GGUF cache, converting once (streaming) from
    /// <paramref name="source"/> (a mmap-backed weight dict) if the cache is missing. Returns the (mmap-backed)
    /// quantized weight dict plus the <see cref="GgufLoader"/> that owns the mapping — the caller MUST keep it alive
    /// while the weights are used and dispose it when done. When <paramref name="quant"/> is unset/unrecognized,
    /// returns <paramref name="source"/> unchanged with a null loader.</summary>
    public static (IReadOnlyDictionary<string, Tensor> weights, GgufLoader? cacheLoader) LoadQuantized(
        IReadOnlyDictionary<string, Tensor> source, string cacheGgufPath, string? quant)
    {
        DType? backbone = ResolveQuant(quant);
        if (backbone is null) return (source, null);

        if (!File.Exists(cacheGgufPath))
        {
            Logs.Info($"[HeartMuLa] Quantizing LM to {backbone.Value.Name} → {cacheGgufPath} (one-time, streaming — no OOM)...");
            Directory.CreateDirectory(Path.GetDirectoryName(cacheGgufPath)!);
            string tmp = cacheGgufPath + ".tmp";
            try
            {
                GgufQuantizer.ConvertDictionaryToGguf(source, tmp, PolicyFor(backbone.Value), "heartmula");
                File.Move(tmp, cacheGgufPath, overwrite: true);
            }
            catch
            {
                try { if (File.Exists(tmp)) File.Delete(tmp); } catch { /* best-effort cleanup */ }
                throw;
            }
            Logs.Info($"[HeartMuLa] Quantized cache written ({new FileInfo(cacheGgufPath).Length / (1024 * 1024)} MB).");
        }

        GgufLoader loader = new();
        loader.Load(cacheGgufPath);
        Dictionary<string, Tensor> weights = new(loader.Descriptors.Count);
        foreach (string name in loader.Descriptors.Keys) weights[name] = loader.GetTensor(name);
        Logs.Info($"[HeartMuLa] Loaded {backbone.Value.Name} quantized weights from cache ({weights.Count} tensors, {new FileInfo(cacheGgufPath).Length / (1024 * 1024)} MB).");
        return (weights, loader);
    }

    /// <summary>HeartMuLa/CSM quant policy: quantize the 2-D projection + head matrices to the chosen dtype; keep the
    /// embed tables (host-gathered per row), all norms, biases, and the tiny conditioning vectors (muq_linear,
    /// unconditional_text_embedding) at F16. Names differ from the llama.cpp defaults — CSM uses
    /// <c>text_embeddings</c>/<c>audio_embeddings</c> (not <c>embed_tokens</c>) — so a custom keep-predicate is needed.</summary>
    private static GgufQuantPolicy PolicyFor(DType backbone) => new()
    {
        BackboneDType = backbone,
        // V/output projections get a higher-fidelity quant on the K-quant mixes (matches llama.cpp _M policy).
        HighFidelityDType = backbone == DType.Q4_K ? DType.Q6_K : (backbone == DType.Q5_K ? DType.Q6_K : null),
        IsHighFidelity = name =>
        {
            string l = name.ToLowerInvariant();
            return l.Contains("v_proj", StringComparison.Ordinal) || l.Contains("o_proj", StringComparison.Ordinal);
        },
        ShouldKeepF16 = (name, _) =>
        {
            string l = name.ToLowerInvariant();
            return l.Contains("embed", StringComparison.Ordinal)          // text_embeddings / audio_embeddings
                || l.Contains("norm", StringComparison.Ordinal)           // input/post-attn/final norms
                || l.EndsWith(".bias", StringComparison.Ordinal)
                || l.Contains("muq", StringComparison.Ordinal)            // muq_linear (tiny, once/gen)
                || l.Contains("unconditional", StringComparison.Ordinal); // unconditional_text_embedding
        },
    };
}
