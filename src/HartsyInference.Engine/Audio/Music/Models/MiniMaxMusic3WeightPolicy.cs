using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.Csm;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.SafeTensors;
using System.Security.Cryptography;
using System.Text;
using System.Linq;

namespace HartsyInference.Engine.Audio;

/// <summary>Precision policy for MiniMax Music 3's two large components.
///
/// <para>The released checkpoint is 17.2 GB of BF16 language model plus 9.6 GB of F32 transformer — together more
/// than a 24 GB card holds even before the KV caches, and far more than a 12 GB one. The <c>:q8</c>/<c>:q4</c>
/// variants quantize the language model's projections into a disk-cached GGUF and cast the transformer to BF16,
/// which is what makes the model reachable on smaller cards. The bare variant stays at checkpoint precision and is
/// the parity baseline.</para></summary>
internal static class MiniMaxMusic3WeightPolicy
{
    /// <summary>Applies the language model's quantization, returning the weights to load and (when quantizing) the GGUF mapping that owns them — keep it alive for the model's lifetime.</summary>
    internal static IReadOnlyDictionary<string, Tensor> PrepareLanguageModel(
        IReadOnlyDictionary<string, Tensor> weights, string repo, string? quant, out IDisposable? cache)
    {
        cache = null;
        DType? backbone = CsmWeightCache.ResolveQuant(quant);
        if (backbone is null)
        {
            return weights;
        }
        // Only the decoder layers are quantized. The embedding table and the output head are read row-wise on the
        // host (the head is sliced to 16385 rows at load), so quantizing them would buy nothing and would break the
        // row reader; they stay mmapped from the source checkpoint.
        return QuantizeToCache(weights, key => key.StartsWith("model.layers.", StringComparison.Ordinal),
            PolicyFor(backbone.Value), CachePath(repo, quant!, "lm"), "language model", backbone.Value, out cache);
    }

    /// <summary>Applies the depth decoder's quantization, returning the weights to load and (when quantizing) the
    /// GGUF mapping that owns them — keep it alive for the model's lifetime.
    ///
    /// <para>The depth decoder runs eight forwards per audio frame, so its 1.3 GB of BF16 is re-read far more often
    /// than the language model's weights are; leaving it unquantized under <c>:q8</c>/<c>:q4</c> made it the second
    /// biggest term in the autoregressive stage.</para></summary>
    internal static IReadOnlyDictionary<string, Tensor> PrepareDepthDecoder(
        IReadOnlyDictionary<string, Tensor> weights, string repo, string? quant, out IDisposable? cache)
    {
        cache = null;
        DType? backbone = CsmWeightCache.ResolveQuant(quant);
        if (backbone is null)
        {
            return weights;
        }
        // Only the projections go through a backend op. audio_embeddings, pos_embedding and every norm are read as
        // RAW FLOATS on the host (MiniMaxMusic3DepthDecoder widens them at load), so a block-quantized layout would
        // be read as garbage rather than rejected — they stay mmapped from the source checkpoint.
        return QuantizeToCache(weights, IsDepthProjection, DepthPolicyFor(backbone.Value),
            CachePath(repo, quant!, "depth"), "depth decoder", backbone.Value, out cache);
    }

    /// <summary>True for the depth decoder's matmul weights — everything else is read host-side as raw floats.</summary>
    private static bool IsDepthProjection(string key)
        => key.Contains(".attn.to_", StringComparison.Ordinal)
            || key.EndsWith("gate_proj.weight", StringComparison.Ordinal)
            || key.EndsWith("up_proj.weight", StringComparison.Ordinal)
            || key.EndsWith("down_proj.weight", StringComparison.Ordinal)
            || key.StartsWith("audio_heads.", StringComparison.Ordinal)
            || key.Equals("projection.weight", StringComparison.Ordinal);

    // Beside the weights these are derived from, via the SAME root AudioModelCache resolved them through
    // (paths.modelCacheRoot, else paths.modelsRoot/audio, else the user cache dir) — resolving against
    // paths.modelsRoot alone would drop multi-gigabyte quant caches on <repo>/Models for an install that
    // configured only the cache root, which may be read-only.
    private static string CachePath(string repo, string quant, string component) => Path.Combine(
        AudioModelCache.CacheRoot, "music", "_gguf-cache",
        component == "lm" ? $"{repo.Replace('/', '_')}-{quant.ToLowerInvariant()}.gguf"
            : $"{repo.Replace('/', '_')}-{component}-{quant.ToLowerInvariant()}.gguf");

    /// <summary>Quantizes the <paramref name="quantizable"/> subset of <paramref name="weights"/> into a disk-cached GGUF (written once, on the first run) and returns it merged back over the untouched remainder.</summary>
    private static IReadOnlyDictionary<string, Tensor> QuantizeToCache(
        IReadOnlyDictionary<string, Tensor> weights, Func<string, bool> quantizable, GgufQuantPolicy policy,
        string cachePath, string label, DType backbone, out IDisposable? cache)
    {
        Dictionary<string, Tensor> selected = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        Dictionary<string, Tensor> passthrough = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        foreach (KeyValuePair<string, Tensor> entry in weights)
        {
            (quantizable(entry.Key) ? selected : passthrough)[entry.Key] = entry.Value;
        }

        (Dictionary<string, Tensor> quantized, GgufLoader loader) =
            GgufQuantCache.Load(selected, cachePath, policy, "minimax-music3", "[Audio][MiniMaxMusic3]");
        cache = loader;
        Dictionary<string, Tensor> merged = new Dictionary<string, Tensor>(passthrough, StringComparer.Ordinal);
        foreach (KeyValuePair<string, Tensor> entry in quantized)
        {
            merged[entry.Key] = entry.Value;
        }
        Logs.Info($"[Audio][MiniMaxMusic3] Loaded the {backbone.Name} {label} from cache "
            + $"({new FileInfo(cachePath).Length / (1024 * 1024)} MB).");
        return merged;
    }

    /// <summary>The flow transformer on the quantized variants, with its projections stored as BF16 in a disk cache that is
    /// written once and then mapped. The loader borrows the mapping, so no BF16 copy of the 9.6 GB F32 source is held in host
    /// memory; the device upload reads the mapped pages. A 9.6 GB F32 transformer does not fit beside the vocoder on a 12 GB card,
    /// which is why the narrowing exists. Returns the source untouched when nothing is quantized.</summary>
    internal static IReadOnlyDictionary<string, Tensor> PrepareTransformer(
        IReadOnlyDictionary<string, Tensor> weights, string repo, string? quant, out IDisposable? owned)
    {
        owned = null;
        if (CsmWeightCache.ResolveQuant(quant) is null)
        {
            return weights;
        }
        string path = Path.Combine(AudioModelCache.CacheRoot, "music", "_gguf-cache",
            $"{repo.Replace('/', '_')}-transformer-bf16.safetensors");
        string signature = TransformerSignature(weights);
        string sidecar = path + ".sig";
        if (!File.Exists(path) || !File.Exists(sidecar) || File.ReadAllText(sidecar) != signature)
        {
            List<SafeTensorsStreamWriter.Entry> entries = new List<SafeTensorsStreamWriter.Entry>(weights.Count);
            foreach (KeyValuePair<string, Tensor> entry in weights)
            {
                Tensor source = entry.Value;
                if (NarrowsToBf16(entry.Key, source))
                {
                    entries.Add(new SafeTensorsStreamWriter.Entry(entry.Key, DType.BF16, source.Shape,
                        () => (source.CastTo(DType.BF16), true)));
                }
                else
                {
                    entries.Add(new SafeTensorsStreamWriter.Entry(entry.Key, source.DType, source.Shape, () => (source, false)));
                }
            }
            Logs.Info($"[Audio][MiniMaxMusic3] Writing the BF16 flow transformer cache '{path}' (once; later runs map it).");
            SafeTensorsStreamWriter.Write(path, entries, new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["hartsy.policy"] = "transformer-bf16-v1",
                ["hartsy.signature"] = signature,
            });
            File.WriteAllText(sidecar, signature);
        }
        SafeTensorsLoader loader = new SafeTensorsLoader();
        try
        {
            loader.Load(path);
        }
        catch
        {
            loader.Dispose();
            throw;
        }
        owned = loader;
        return loader.GetAllTensors();
    }

    /// <summary>Only the F32 projections are narrowed. Norms and biases are tiny and precision-sensitive, and time_proj is read
    /// host-side by the Fourier embedding rather than through a backend op.</summary>
    private static bool NarrowsToBf16(string name, Tensor tensor) =>
        tensor.DType == DType.F32 && tensor.Shape.Rank >= 2 && !name.StartsWith("time_proj", StringComparison.Ordinal);

    /// <summary>Identifies the cache's source: the tensor names, shapes, dtypes and narrowing decisions. A release that changes
    /// any of them rebuilds the cache.</summary>
    private static string TransformerSignature(IReadOnlyDictionary<string, Tensor> weights)
    {
        StringBuilder text = new StringBuilder("transformer-bf16-v1\n");
        foreach (KeyValuePair<string, Tensor> entry in weights.OrderBy(e => e.Key, StringComparer.Ordinal))
        {
            Tensor tensor = entry.Value;
            text.Append(entry.Key).Append('|').Append(tensor.DType.Name).Append('|');
            for (int d = 0; d < tensor.Shape.Rank; d++)
            {
                text.Append(tensor.Shape[d]).Append('x');
            }
            text.Append('|').Append(NarrowsToBf16(entry.Key, tensor) ? "bf16" : "keep").Append('\n');
        }
        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(text.ToString())));
    }

    /// <summary>Quantizes the projection matrices; keeps norms and biases at F16.</summary>
    private static GgufQuantPolicy PolicyFor(DType backbone) => new()
    {
        BackboneDType = backbone,
        HighFidelityDType = backbone == DType.Q4_K || backbone == DType.Q5_K ? DType.Q6_K : null,
        IsHighFidelity = name =>
            name.Contains("v_proj", StringComparison.Ordinal) || name.Contains("o_proj", StringComparison.Ordinal),
        ShouldKeepF16 = (name, _) =>
            name.Contains("norm", StringComparison.OrdinalIgnoreCase) || name.EndsWith(".bias", StringComparison.Ordinal),
    };

    /// <summary>Same mix for the depth decoder, whose attention weights are named <c>to_v</c>/<c>to_out</c> rather than <c>v_proj</c>/<c>o_proj</c>. Nothing precision-sensitive reaches this policy — the caller only offers it the projections — so there is no keep-at-F16 set.</summary>
    private static GgufQuantPolicy DepthPolicyFor(DType backbone) => new()
    {
        BackboneDType = backbone,
        HighFidelityDType = backbone == DType.Q4_K || backbone == DType.Q5_K ? DType.Q6_K : null,
        IsHighFidelity = name =>
            name.Contains(".to_v.", StringComparison.Ordinal) || name.Contains(".to_out.", StringComparison.Ordinal),
    };

    private sealed class OwnedTensors : IDisposable
    {
        private readonly List<Tensor> _tensors = [];

        internal Tensor Track(Tensor tensor)
        {
            _tensors.Add(tensor);
            return tensor;
        }

        public void Dispose()
        {
            foreach (Tensor tensor in _tensors)
            {
                tensor.Dispose();
            }
            _tensors.Clear();
        }
    }
}
