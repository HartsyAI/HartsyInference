using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Normalizes MiniMax-H3 Fun ControlNet-Union checkpoints from VideoX-Fun or Comfy layouts into the fused
/// H3 block names consumed by the native control branch.</summary>
public static unsafe class MiniMaxH3ControlNetCheckpointConverter
{
    private const string ToQ = ".attn.to_q.weight";
    private const string ToK = ".attn.to_k.weight";
    private const string ToV = ".attn.to_v.weight";

    /// <summary>Converts every tensor without silently dropping unknown branch state.</summary>
    public static Dictionary<string, Tensor> Convert(Dictionary<string, Tensor> source)
    {
        ArgumentNullException.ThrowIfNull(source);
        Dictionary<string, Tensor> normalized = CheckpointConvertUtils.ApplyFp8ScaledDequant(source);
        Dictionary<string, Tensor> output = new Dictionary<string, Tensor>(normalized.Count, StringComparer.Ordinal);
        HashSet<string> consumed = new HashSet<string>(StringComparer.Ordinal);

        foreach (KeyValuePair<string, Tensor> pair in normalized)
        {
            string key = StripWrapper(pair.Key);
            if (!key.EndsWith(ToQ, StringComparison.Ordinal))
            {
                continue;
            }

            string root = key[..^ToQ.Length];
            string sourcePrefix = pair.Key[..^ToQ.Length];
            Tensor q = pair.Value;
            Tensor k = Require(normalized, sourcePrefix + ToK);
            Tensor v = Require(normalized, sourcePrefix + ToV);
            Tensor fused = ConcatRows(q, k, v, sourcePrefix);
            AddUnique(output, root + ".attn.qkv_proj.weight", fused);
            consumed.Add(pair.Key);
            consumed.Add(sourcePrefix + ToK);
            consumed.Add(sourcePrefix + ToV);
        }

        foreach (KeyValuePair<string, Tensor> pair in normalized)
        {
            if (consumed.Contains(pair.Key))
            {
                continue;
            }
            string key = StripWrapper(pair.Key);
            if (key.EndsWith(ToK, StringComparison.Ordinal) || key.EndsWith(ToV, StringComparison.Ordinal))
            {
                throw new HartsyInferenceException(
                    $"MiniMax-H3 control checkpoint carries '{pair.Key}' without a matching split Q projection.");
            }

            bool diffusersFc1 = key.EndsWith(".ff.net.0.proj.weight", StringComparison.Ordinal);
            string mapped = MapKey(key);
            Tensor tensor = diffusersFc1 ? SwapSwiGluHalves(pair.Value, pair.Key) : pair.Value;
            AddUnique(output, mapped, tensor);
        }

        return output;
    }

    /// <summary>Loads one safetensors branch and returns converted tensors backed by the returned loader.</summary>
    public static (Dictionary<string, Tensor> Weights, SafeTensorsLoader Loader) LoadAndConvert(string path)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException("MiniMax-H3 Fun ControlNet checkpoint not found.", path);
        }
        SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(path);
        try
        {
            return (Convert(loader.GetAllTensors()), loader);
        }
        catch
        {
            loader.Dispose();
            throw;
        }
    }

    private static string StripWrapper(string key)
    {
        string[] prefixes = ["controlnet.", "control_model.", "model.controlnet."];
        foreach (string prefix in prefixes)
        {
            if (key.StartsWith(prefix, StringComparison.Ordinal))
            {
                return key[prefix.Length..];
            }
        }
        return key;
    }

    private static string MapKey(string key)
    {
        return key.Replace(".attn.norm_q.", ".attn.q_norm.", StringComparison.Ordinal)
            .Replace(".attn.norm_k.", ".attn.k_norm.", StringComparison.Ordinal)
            .Replace(".attn.to_out.0.", ".attn.out_proj.", StringComparison.Ordinal)
            .Replace(".ff.net.0.proj.", ".mlp.fc1.", StringComparison.Ordinal)
            .Replace(".ff.net.2.", ".mlp.fc2.", StringComparison.Ordinal);
    }

    private static Tensor ConcatRows(Tensor q, Tensor k, Tensor v, string prefix)
    {
        if (q.Shape.Rank != 2 || k.Shape.Rank != 2 || v.Shape.Rank != 2 || q.Shape[1] != k.Shape[1]
            || q.Shape[1] != v.Shape[1] || q.DType != k.DType || q.DType != v.DType)
        {
            throw new HartsyInferenceException(
                $"Split Q/K/V tensors at '{prefix}' are incompatible: q={q.Shape}/{q.DType}, "
                + $"k={k.Shape}/{k.DType}, v={v.Shape}/{v.DType}.");
        }
        if (q.QuantInfo is not null || k.QuantInfo is not null || v.QuantInfo is not null
            || q.Fp8ScaleFactor != k.Fp8ScaleFactor || q.Fp8ScaleFactor != v.Fp8ScaleFactor
            || q.Fp8InputScaleFactor != k.Fp8InputScaleFactor || q.Fp8InputScaleFactor != v.Fp8InputScaleFactor)
        {
            throw new NotSupportedException(
                $"Split quantized Q/K/V tensors at '{prefix}' cannot be fused losslessly. Use the published BF16 "
                + "branch or a Comfy-layout checkpoint that already carries qkv_proj.");
        }

        Tensor fused = new Tensor(new TensorShape(q.Shape[0] + k.Shape[0] + v.Shape[0], q.Shape[1]), q.DType)
        {
            Fp8ScaleFactor = q.Fp8ScaleFactor,
            Fp8InputScaleFactor = q.Fp8InputScaleFactor,
        };
        long qBytes = q.DType.ComputeByteCount(q.ElementCount);
        long kBytes = k.DType.ComputeByteCount(k.ElementCount);
        long vBytes = v.DType.ComputeByteCount(v.ElementCount);
        byte* destination = (byte*)fused.DataPointer;
        Buffer.MemoryCopy((void*)q.DataPointer, destination, qBytes, qBytes);
        Buffer.MemoryCopy((void*)k.DataPointer, destination + qBytes, kBytes, kBytes);
        Buffer.MemoryCopy((void*)v.DataPointer, destination + qBytes + kBytes, vBytes, vBytes);
        return fused;
    }

    private static Tensor SwapSwiGluHalves(Tensor source, string key)
    {
        if (source.Shape.Rank != 2 || source.Shape[0] % 2 != 0)
        {
            throw new HartsyInferenceException(
                $"VideoX-Fun SwiGLU tensor '{key}' must be rank-2 with an even row count; got {source.Shape}.");
        }
        if (source.QuantInfo is not null)
        {
            throw new NotSupportedException(
                $"VideoX-Fun split-layout quantized SwiGLU tensor '{key}' cannot be row-swapped without rewriting "
                + "its quantization companions. Use a Comfy-layout converted branch.");
        }

        long halfRows = source.Shape[0] / 2;
        long rowBytes = source.DType.ComputeByteCount(source.Shape[1]);
        long halfBytes = checked(halfRows * rowBytes);
        Tensor swapped = new Tensor(source.Shape, source.DType)
        {
            Fp8ScaleFactor = source.Fp8ScaleFactor,
            Fp8InputScaleFactor = source.Fp8InputScaleFactor,
        };
        byte* input = (byte*)source.DataPointer;
        byte* output = (byte*)swapped.DataPointer;
        Buffer.MemoryCopy(input + halfBytes, output, halfBytes, halfBytes);
        Buffer.MemoryCopy(input, output + halfBytes, halfBytes, halfBytes);
        return swapped;
    }

    private static Tensor Require(IReadOnlyDictionary<string, Tensor> tensors, string key)
    {
        return tensors.TryGetValue(key, out Tensor? tensor) ? tensor
            : throw new HartsyInferenceException($"MiniMax-H3 control checkpoint is missing '{key}'.");
    }

    private static void AddUnique(Dictionary<string, Tensor> output, string key, Tensor tensor)
    {
        if (!output.TryAdd(key, tensor))
        {
            if (!ReferenceEquals(output[key], tensor))
            {
                tensor.Dispose();
            }
            throw new HartsyInferenceException($"MiniMax-H3 control conversion produced duplicate key '{key}'.");
        }
    }
}
