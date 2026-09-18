using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Converter for Zeta-Chroma (<c>lodestones/Zeta-Chroma</c>) pixel-proto single-file safetensors. Zeta-Chroma is the Z-Image S3-DiT retrained for pixel space, so this is a thin wrapper over <see cref="ZImageCheckpointConverter"/>: the shared partitioner already strips wrappers (incl. the <c>_orig_mod.</c> torch.compile prefix), folds <c>fp8_scaled</c> companions, and buckets the Zeta-only <c>dec_net.*</c> decoder-head keys into the transformer dict. The Diffusion-side <c>ZetaChromaTransformer.LoadWeights</c> validates the decoder layout defensively.</summary>
public sealed class ZetaChromaCheckpointConverter
{
    /// <summary>Partitions a flat dict of Zeta-Chroma safetensors keys (delegates to the Z-Image partitioner), then normalizes split diffusers-style attention (<c>to_q/to_k/to_v</c>, <c>to_out.0</c>, <c>norm_q/norm_k</c> — the layout newer Zeta releases ship) to the fused Z-Image naming (<c>qkv</c>, <c>out</c>, <c>q_norm/k_norm</c>).</summary>
    public static ZImageCheckpointConverter.ConvertedWeights Convert(IReadOnlyDictionary<string, Tensor> allWeights)
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
            // Refused before anything is allocated: an int8_tensorwise weight's scales are indexed by output row, so
            // a concatenation along dim 0 would need its scales concatenated too, and keeping only Q's would run K
            // and V at the wrong magnitude with no symptom until the image comes out as noise.
            if (q.QuantInfo is not null || k.QuantInfo is not null || v.QuantInfo is not null)
            {
                throw new NotSupportedException(
                    $"Zeta-Chroma '{prefix}' ships split attention in {q.QuantInfo?.Format ?? "a per-row quantized"} "
                    + "format, whose per-row scales cannot be fused here. Use a build with fused qkv weights.");
            }
            if (q.Fp8ScaleFactor != k.Fp8ScaleFactor || q.Fp8ScaleFactor != v.Fp8ScaleFactor)
            {
                throw new NotSupportedException(
                    $"Zeta-Chroma '{prefix}' has per-tensor fp8 scales that differ across Q/K/V "
                    + $"({q.Fp8ScaleFactor}/{k.Fp8ScaleFactor}/{v.Fp8ScaleFactor}); one fused weight carries only one.");
            }
            if (q.DType != k.DType || q.DType != v.DType)
            {
                throw new NotSupportedException(
                    $"Zeta-Chroma '{prefix}' stores Q/K/V in three different dtypes "
                    + $"({q.DType.Name}/{k.DType.Name}/{v.DType.Name}); one fused weight declares one encoding, so "
                    + "concatenating their bytes would produce a tensor nothing can decode.");
            }
            // Block quants pack fixed-size blocks that never span a row, so the concatenation is byte-exact only
            // while each row is a whole number of blocks — which SliceByteCount validates.
            long qBytes = CheckpointConvertUtils.SliceByteCount(q, q.ElementCount);
            long kBytes = CheckpointConvertUtils.SliceByteCount(k, k.ElementCount);
            long vBytes = CheckpointConvertUtils.SliceByteCount(v, v.ElementCount);
            Tensor fused = new Tensor(new TensorShape(q.Shape[0] + k.Shape[0] + v.Shape[0], q.Shape[1]), q.DType);
            fused.Fp8ScaleFactor = q.Fp8ScaleFactor;
            fused.Fp8InputScaleFactor = q.Fp8InputScaleFactor;
            long fusedBytes = CheckpointConvertUtils.SliceByteCount(fused, fused.ElementCount);
            byte* dst = (byte*)fused.DataPointer;
            Buffer.MemoryCopy((void*)q.DataPointer, dst, fusedBytes, qBytes);
            Buffer.MemoryCopy((void*)k.DataPointer, dst + qBytes, fusedBytes - qBytes, kBytes);
            Buffer.MemoryCopy((void*)v.DataPointer, dst + qBytes + kBytes, fusedBytes - qBytes - kBytes, vBytes);
            weights[prefix + "qkv.weight"] = fused;
            weights.Remove(qKey);
            weights.Remove(prefix + "to_k.weight");
            weights.Remove(prefix + "to_v.weight");

            if (weights.Remove(prefix + "to_out.0.weight", out Tensor? outW)) weights[prefix + "out.weight"] = outW;
            if (weights.Remove(prefix + "norm_q.weight", out Tensor? nq)) weights[prefix + "q_norm.weight"] = nq;
            if (weights.Remove(prefix + "norm_k.weight", out Tensor? nk)) weights[prefix + "k_norm.weight"] = nk;
        }
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
