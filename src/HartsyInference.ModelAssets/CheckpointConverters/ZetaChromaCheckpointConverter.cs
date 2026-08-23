using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Converter for Zeta-Chroma (<c>lodestones/Zeta-Chroma</c>) pixel-proto single-file safetensors. Zeta-Chroma is the Z-Image S3-DiT retrained for pixel space, so this is a thin wrapper over <see cref="ZImageCheckpointConverter"/>: the shared partitioner already strips wrappers (incl. the <c>_orig_mod.</c> torch.compile prefix), folds <c>fp8_scaled</c> companions, and buckets the Zeta-only <c>dec_net.*</c> decoder-head keys into the transformer dict. The Diffusion-side <c>ZetaChromaTransformer.LoadWeights</c> validates the decoder layout defensively.</summary>
public sealed class ZetaChromaCheckpointConverter
{
    /// <summary>Partitions a flat dict of Zeta-Chroma safetensors keys (delegates to the Z-Image partitioner), then normalizes split diffusers-style attention (<c>to_q/to_k/to_v</c>, <c>to_out.0</c>, <c>norm_q/norm_k</c> — the layout newer Zeta releases ship) to the fused Z-Image naming (<c>qkv</c>, <c>out</c>, <c>q_norm/k_norm</c>).</summary>
    public static ZImageCheckpointConverter.ConvertedWeights Convert(Dictionary<string, Tensor> allWeights)
    {
        ZImageCheckpointConverter.ConvertedWeights converted = ZImageCheckpointConverter.Convert(allWeights);
        FuseSplitAttention(converted.Transformer);
        return converted;
    }

    /// <summary>Rewrites split-attention keys in place: row-concats <c>{p}.to_q/to_k/to_v.weight</c> into <c>{p}.qkv.weight</c> and renames the out-projection and QK-norm keys.</summary>
    private static unsafe void FuseSplitAttention(Dictionary<string, Tensor> weights)
    {
        List<string> qKeys = new();
        foreach (string key in weights.Keys)
            if (key.EndsWith(".attention.to_q.weight", StringComparison.Ordinal))
                qKeys.Add(key);

        foreach (string qKey in qKeys)
        {
            string prefix = qKey[..^"to_q.weight".Length];
            Tensor q = weights[qKey];
            Tensor k = weights[prefix + "to_k.weight"];
            Tensor v = weights[prefix + "to_v.weight"];
            long qBytes = q.DType.ComputeByteCount(q.ElementCount);
            long kBytes = k.DType.ComputeByteCount(k.ElementCount);
            long vBytes = v.DType.ComputeByteCount(v.ElementCount);
            Tensor fused = new Tensor(new TensorShape(q.Shape[0] + k.Shape[0] + v.Shape[0], q.Shape[1]), q.DType);
            byte* dst = (byte*)fused.DataPointer;
            Buffer.MemoryCopy((void*)q.DataPointer, dst, qBytes, qBytes);
            Buffer.MemoryCopy((void*)k.DataPointer, dst + qBytes, kBytes, kBytes);
            Buffer.MemoryCopy((void*)v.DataPointer, dst + qBytes + kBytes, vBytes, vBytes);
            weights[prefix + "qkv.weight"] = fused;
            weights.Remove(qKey);
            weights.Remove(prefix + "to_k.weight");
            weights.Remove(prefix + "to_v.weight");

            if (weights.Remove(prefix + "to_out.0.weight", out Tensor? outW)) weights[prefix + "out.weight"] = outW;
            if (weights.Remove(prefix + "norm_q.weight", out Tensor? nq)) weights[prefix + "q_norm.weight"] = nq;
            if (weights.Remove(prefix + "norm_k.weight", out Tensor? nk)) weights[prefix + "k_norm.weight"] = nk;
        }
    }

    /// <summary>Loads and partitions a Zeta-Chroma single-file checkpoint.</summary>
    public static (ZImageCheckpointConverter.ConvertedWeights weights, SafeTensorsLoader loader) LoadAndConvert(
        string checkpointPath)
    {
        SafeTensorsLoader loader = new();
        loader.Load(checkpointPath);
        ZImageCheckpointConverter.ConvertedWeights converted = Convert(loader.GetAllTensors());
        return (converted, loader);
    }

    /// <summary>True when a partitioned Z-Image-family transformer dict is a Zeta-Chroma pixel checkpoint (the <c>dec_net.*</c> decoder head replaces <c>final_layer.*</c>).</summary>
    public static bool IsZetaChroma(IReadOnlyDictionary<string, Tensor> transformerWeights)
    {
        foreach (string key in transformerWeights.Keys)
        {
            if (key.StartsWith("dec_net.", StringComparison.Ordinal))
                return true;
        }
        return false;
    }
}
