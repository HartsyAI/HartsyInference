using System.Text.Json;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Converter for Lumina-Image-2.0 (Alpha-VLLM) safetensors checkpoints. The diffusers and ComfyUI single-file distributions both use the same key naming as the upstream <c>Lumina2Transformer2DModel</c>: separate <c>attn.to_q/to_k/to_v.weight</c>, <c>attn.to_out.0.weight</c>, <c>attn.norm_q/k.weight</c>, <c>norm1.norm.weight</c> + <c>norm1.linear.{weight,bias}</c> (or just <c>norm1.weight</c> on context_refiner), <c>norm2.weight</c>, <c>ffn_norm{1,2}.weight</c>, <c>feed_forward.linear_{1,2,3}.weight</c>, <c>x_embedder.{weight,bias}</c>, <c>time_caption_embed.{caption_embedder.{0,1},timestep_embedder.linear_{1,2}}.*</c>, and <c>norm_out.linear_{1,2}.{weight,bias}</c>. This converter is mostly passthrough — its job is to partition transformer/VAE/text-encoder buckets.</summary>
public sealed class Lumina2CheckpointConverter
{
    /// <summary>Result of partitioning a Lumina-Image-2.0 single-file safetensors checkpoint.</summary>
    public sealed class ConvertedWeights
    {
        /// <summary>Lumina-Image-2.0 transformer weights — pass directly to <c>Lumina2Transformer.LoadWeights</c>.</summary>
        public required Dictionary<string, Tensor> Transformer { get; init; }

        /// <summary>VAE weights (Flux VAE; usually empty when the transformer ships standalone).</summary>
        public required Dictionary<string, Tensor> Vae { get; init; }

        /// <summary>Gemma 2 2B text encoder weights (usually empty — text encoder ships separately).</summary>
        public required Dictionary<string, Tensor> TextEncoder { get; init; }

        /// <summary>True if any transformer linear weight is FP8 — pipeline should preload via the FP8 path.</summary>
        public required bool IsFp8Mix { get; init; }
    }

    /// <summary>Every file that makes up this checkpoint: the shard set a sibling <c>*.safetensors.index.json</c> lists <paramref name="checkpointPath"/> as a member of (the real <c>Alpha-VLLM/Lumina-Image-2.0</c> diffusers weights ship as 2 shards), or the one path otherwise.</summary>
    /// <remarks><para>The shards are opened as one <see cref="Checkpoints.CheckpointSource"/> rather than merged raw:
    /// safetensors sharding makes no promise that a weight and its <c>.weight_scale</c> land in the same file, so
    /// folding each shard alone splits pairs that belong together.</para>
    /// <para>Membership is what decides it, not the index's mere presence: a GGUF repack parked beside the original
    /// sharded release would otherwise be discarded for the full-precision checkpoint the index names, which loads a
    /// different model — or nothing, out of memory — with no sign that the selection was ignored.</para></remarks>
    public static IReadOnlyList<string> ResolveShardPaths(string checkpointPath)
    {
        ArgumentException.ThrowIfNullOrEmpty(checkpointPath);
        string? dir = Path.GetDirectoryName(checkpointPath);
        if (string.IsNullOrEmpty(dir))
            return [checkpointPath];
        string selected = Path.GetFileName(checkpointPath);
        foreach (string indexPath in Directory.GetFiles(dir, "*.safetensors.index.json"))
        {
            SortedSet<string> members = ReadIndexMembers(indexPath);
            if (!members.Contains(selected))
                continue;
            List<string> shards = new List<string>(members.Count);
            foreach (string member in members)
            {
                string shard = Path.Combine(dir, member);
                if (!File.Exists(shard))
                    throw new FileNotFoundException($"Lumina-2 shard '{member}' listed in '{indexPath}' is missing.", shard);
                shards.Add(shard);
            }
            return shards;
        }
        return [checkpointPath];
    }

    /// <summary>Partitions a flat dict of Lumina-Image-2.0 safetensors keys.</summary>
    /// <remarks>Quantization companions are expected to be folded already — <see cref="Checkpoints.CheckpointSource"/>
    /// does it before any converter runs, because this converter strips key prefixes its <c>.weight_scale</c>
    /// companion does not share, and folding after that pairs nothing and drops the scale silently.</remarks>
    public static ConvertedWeights Convert(IReadOnlyDictionary<string, Tensor> allWeights)
    {
        CheckpointConvertUtils.RequireFoldedCompanions(allWeights, nameof(Lumina2CheckpointConverter));

        Dictionary<string, Tensor> transformer = new(allWeights.Count);
        Dictionary<string, Tensor> vae = new();
        Dictionary<string, Tensor> textEncoder = new();

        foreach (KeyValuePair<string, Tensor> kvp in allWeights)
        {
            string key = kvp.Key;
            Tensor tensor = kvp.Value;

            string transformerKey = key;
            if (key.StartsWith("model.diffusion_model.", StringComparison.Ordinal))
                transformerKey = key["model.diffusion_model.".Length..];
            else if (key.StartsWith("transformer.", StringComparison.Ordinal))
                transformerKey = key["transformer.".Length..];

            if (IsTransformerKey(transformerKey))
            {
                transformer[transformerKey] = tensor;
                continue;
            }

            if (key.StartsWith("vae.", StringComparison.Ordinal) ||
                key.StartsWith("first_stage_model.", StringComparison.Ordinal))
            {
                vae[key] = tensor;
                continue;
            }

            if (key.StartsWith("text_encoder.", StringComparison.Ordinal) ||
                key.StartsWith("text_encoders.", StringComparison.Ordinal))
            {
                textEncoder[key] = tensor;
            }
        }

        bool isFp8Mix = DetectFp8Mix(transformer);

        return new ConvertedWeights
        {
            Transformer = transformer,
            Vae = vae,
            TextEncoder = textEncoder,
            IsFp8Mix = isFp8Mix,
        };
    }

    /// <summary>Auto-detects layer counts and FP8 status from a transformer dict.</summary>
    public static (int numLayers, int numRefinerLayers, int hidden, int qOutDim, int kvOutDim, int ffnDim, bool isFp8Mix) DetectArchitecture(
        IReadOnlyDictionary<string, Tensor> transformerWeights)
    {
        int maxLayer = -1;
        int maxNoiseRefiner = -1;
        int maxContextRefiner = -1;

        foreach (string key in transformerWeights.Keys)
        {
            if (key.StartsWith("layers.", StringComparison.Ordinal))
            {
                int dot = key.IndexOf('.', 7);
                if (dot > 0 && int.TryParse(key.AsSpan(7, dot - 7), out int idx) && idx > maxLayer)
                    maxLayer = idx;
            }
            else if (key.StartsWith("noise_refiner.", StringComparison.Ordinal))
            {
                int dot = key.IndexOf('.', 14);
                if (dot > 0 && int.TryParse(key.AsSpan(14, dot - 14), out int idx) && idx > maxNoiseRefiner)
                    maxNoiseRefiner = idx;
            }
            else if (key.StartsWith("context_refiner.", StringComparison.Ordinal))
            {
                int dot = key.IndexOf('.', 16);
                if (dot > 0 && int.TryParse(key.AsSpan(16, dot - 16), out int idx) && idx > maxContextRefiner)
                    maxContextRefiner = idx;
            }
        }

        int hidden = 2304;
        int qOutDim = 2304;
        int kvOutDim = 768;
        int ffnDim = 6144;

        if (transformerWeights.TryGetValue("layers.0.attn.to_q.weight", out Tensor? toQ))
        {
            hidden = (int)toQ.Shape[1];
            qOutDim = (int)toQ.Shape[0];
        }
        if (transformerWeights.TryGetValue("layers.0.attn.to_k.weight", out Tensor? toK))
            kvOutDim = (int)toK.Shape[0];
        if (transformerWeights.TryGetValue("layers.0.feed_forward.linear_1.weight", out Tensor? linear1))
            ffnDim = (int)linear1.Shape[0];

        int numRefiner = Math.Min(maxNoiseRefiner, maxContextRefiner) + 1;
        if (numRefiner <= 0) numRefiner = 2;

        return (maxLayer + 1, numRefiner, hidden, qOutDim, kvOutDim, ffnDim, DetectFp8Mix(transformerWeights));
    }

    private static bool IsTransformerKey(string key)
    {
        return key.StartsWith("layers.", StringComparison.Ordinal)
            || key.StartsWith("noise_refiner.", StringComparison.Ordinal)
            || key.StartsWith("context_refiner.", StringComparison.Ordinal)
            || key.StartsWith("time_caption_embed.", StringComparison.Ordinal)
            || key.StartsWith("x_embedder.", StringComparison.Ordinal)
            || key.StartsWith("norm_out.", StringComparison.Ordinal);
    }

    private static bool DetectFp8Mix(IReadOnlyDictionary<string, Tensor> transformer)
    {
        if (transformer.TryGetValue("layers.0.attn.to_q.weight", out Tensor? probe))
            return probe.DType == DType.F8E4M3 || probe.DType == DType.F8E5M2;

        foreach (Tensor t in transformer.Values)
        {
            if (t.DType == DType.F8E4M3 || t.DType == DType.F8E5M2)
                return true;
        }
        return false;
    }

    /// <summary>The distinct shard file names a diffusers <c>*.safetensors.index.json</c> weight map points at, in ordinal order.</summary>
    private static SortedSet<string> ReadIndexMembers(string indexPath)
    {
        SortedSet<string> members = new SortedSet<string>(StringComparer.Ordinal);
        using FileStream stream = File.OpenRead(indexPath);
        using JsonDocument document = JsonDocument.Parse(stream);
        if (!document.RootElement.TryGetProperty("weight_map", out JsonElement weightMap)
            || weightMap.ValueKind != JsonValueKind.Object)
        {
            return members;
        }
        foreach (JsonProperty entry in weightMap.EnumerateObject())
        {
            string? file = entry.Value.ValueKind == JsonValueKind.String ? entry.Value.GetString() : null;
            if (!string.IsNullOrEmpty(file))
                members.Add(file);
        }
        return members;
    }
}
