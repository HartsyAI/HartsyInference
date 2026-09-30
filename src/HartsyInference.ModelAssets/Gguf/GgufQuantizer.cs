using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf.Codecs;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.Gguf;

/// <summary>Orchestrator: safetensors → GGUF conversion. Wraps <see cref="SafeTensorsLoader"/> + <see cref="GgufQuantPolicy"/> + <see cref="GgufWriter"/>. Per-tensor quantization decision comes from the policy; tensors that <see cref="GgufQuantPolicy.ResolveTargetDType"/> returns F16 for stay at F16, others get quantized via <see cref="GgufCodecRegistry"/>.</summary>
public static class GgufQuantizer
{
    /// <summary>Converts a safetensors file to a GGUF file using the given <paramref name="policy"/>. Sets <c>general.architecture = </c><paramref name="architecture"/> in the output metadata so <see cref="GgufModelLoader"/> picks the right key mapper on read-back.
    ///
    /// <para>Tensor data is mmap-read from the input safetensors and quantized in-memory before writing. For very large checkpoints (10+ GB), this means the working set during conversion is ~2× the input size (mmap + quantized output buffer). Acceptable on 32 GB host RAM for most diffusion models.</para></summary>
    public static GgufQuantizationReport ConvertSafetensorsToGguf(
        string safetensorsPath,
        string outputGgufPath,
        GgufQuantPolicy policy,
        string architecture,
        IDictionary<string, object>? extraMetadata = null)
    {
        if (!File.Exists(safetensorsPath))
            throw new FileNotFoundException("Input safetensors not found.", safetensorsPath);

        SafeTensorsLoader loader = new();
        loader.Load(safetensorsPath);
        try
        {
            GgufQuantizationReport report = ConvertInternal(loader, outputGgufPath, policy, architecture, extraMetadata);
            return report;
        }
        finally
        {
            loader.Dispose();
        }
    }

    /// <summary>Converts a pre-loaded weight dictionary to a GGUF file. Same as <see cref="ConvertSafetensorsToGguf"/> but starts from an already-built <c>Dictionary&lt;string, Tensor&gt;</c> (e.g. from a checkpoint converter's intermediate output, or from another GGUF file we want to re-quantize).</summary>
    public static GgufQuantizationReport ConvertDictionaryToGguf(
        IReadOnlyDictionary<string, Tensor> tensors,
        string outputGgufPath,
        GgufQuantPolicy policy,
        string architecture,
        IDictionary<string, object>? extraMetadata = null)
    {
        GgufQuantizationReport report = new();
        using GgufWriter writer = new(outputGgufPath);
        writer.SetMetadata("general.architecture", architecture);
        writer.SetMetadata("general.name", $"{architecture} (HartsyInference quantized)");
        if (extraMetadata is not null)
        {
            foreach (KeyValuePair<string, object> kv in extraMetadata) writer.SetMetadata(kv.Key, kv.Value);
        }

        List<Tensor> ownedQuantTensors = new();
        try
        {
            foreach (KeyValuePair<string, Tensor> kv in tensors)
            {
                Tensor src = kv.Value;
                DType target = policy.ResolveTargetDType(kv.Key, src);
                Tensor toWrite;

                if (target == src.DType)
                {
                    toWrite = src;
                    report.PassthroughCount++;
                }
                else if (target == DType.F16 || target == DType.F32 || target == DType.BF16)
                {
                    Tensor cast = src.DType == target ? src : src.CastTo(target);
                    if (cast != src) ownedQuantTensors.Add(cast);
                    toWrite = cast;
                    report.CastCount++;
                }
                else
                {
                    toWrite = QuantizeTensor(src, target);
                    ownedQuantTensors.Add(toWrite);
                    report.QuantizedCount++;
                    if (!report.QuantTotals.ContainsKey(target.Name)) report.QuantTotals[target.Name] = 0;
                    report.QuantTotals[target.Name]++;
                }

                writer.AddTensor(kv.Key, toWrite);
            }

            writer.Flush();
            report.OutputBytes = new FileInfo(outputGgufPath).Length;
            Logs.Info($"GgufQuantizer: wrote '{outputGgufPath}' — passthrough={report.PassthroughCount}, cast={report.CastCount}, quantized={report.QuantizedCount}, output={report.OutputBytes / (1024 * 1024)} MB.");
            return report;
        }
        finally
        {
            foreach (Tensor t in ownedQuantTensors) t.Dispose();
        }
    }

    /// <summary>Reads every tensor of a GGUF that <see cref="ConvertDictionaryToGguf"/> wrote back under the shape its
    /// source tensor carried, so a disk-cached quantization comes back exactly as the dictionary it was made from.</summary>
    /// <remarks>
    /// <para>A raw <see cref="GgufLoader"/> hands a tensor back in the file's own order, which since the writer started
    /// emitting ggml <c>ne</c> is the reverse of the engine's: a <c>[N, K]</c> projection reads back as <c>[K, N]</c>.
    /// <see cref="GgufModelLoader"/> relabels that for checkpoints; the two quant caches (HeartMuLa's, MiniMax Music
    /// 3's) read the raw loader and never did, so every projection they fed the backend was transposed, its GEMV
    /// derived <c>M = 0</c> and the first CUDA launch failed with an invalid grid. Both readers hold the source
    /// dictionary, so the shape comes from there rather than from a guess about which writer made the file: a cache
    /// written before the writer changed already matches its source and is left alone, one written after is
    /// swapped back. A tensor without a source (a cache the caller no longer feeds the same dictionary) falls back
    /// to reversing the file's axes, which is right for anything the current writer produced.</para>
    /// <para>Shapes only. The data is row-major in the engine's order either way, so this is
    /// <see cref="Tensor.Reshape"/> over the mmap, valid for the quantized dtypes it never reads. The returned
    /// tensors borrow the loader's mapping; the caller keeps the loader alive for as long as they are used.</para>
    /// </remarks>
    public static Dictionary<string, Tensor> ReadBack(GgufLoader loader, IReadOnlyDictionary<string, Tensor> source)
    {
        ArgumentNullException.ThrowIfNull(loader);
        ArgumentNullException.ThrowIfNull(source);
        Dictionary<string, Tensor> weights = new(loader.Descriptors.Count, StringComparer.Ordinal);
        foreach (string name in loader.Descriptors.Keys)
        {
            Tensor stored = loader.GetTensor(name);
            weights[name] = stored;
            if (source.TryGetValue(name, out Tensor? original))
            {
                if (original.Shape.ElementCount != stored.Shape.ElementCount)
                {
                    throw new HartsyInference.Core.Exceptions.HartsyInferenceException(
                        $"GGUF cache tensor '{name}' holds {stored.Shape.ElementCount} elements but its source holds "
                        + $"{original.Shape.ElementCount}; the cache was not written from this dictionary.");
                }
                if (!SameShape(original.Shape, stored.Shape))
                {
                    weights[name] = stored.Reshape(original.Shape);
                }
                continue;
            }
            if (stored.Shape.Rank >= 2)
            {
                Logs.Warning($"GgufQuantizer: cache tensor '{name}' has no source tensor to take its shape from; "
                    + "assuming the file is in ggml order and reversing its axes.");
                weights[name] = stored.Reshape(Reversed(stored.Shape));
            }
        }
        return weights;
    }

    private static bool SameShape(TensorShape a, TensorShape b)
    {
        if (a.Rank != b.Rank) return false;
        for (int i = 0; i < a.Rank; i++)
        {
            if (a[i] != b[i]) return false;
        }
        return true;
    }

    private static TensorShape Reversed(TensorShape shape)
    {
        long[] dims = new long[shape.Rank];
        for (int i = 0; i < shape.Rank; i++) dims[i] = shape[shape.Rank - 1 - i];
        return new TensorShape(dims);
    }

    /// <summary>Quantizes a single tensor (F32/F16/BF16 source) to a target quant dtype — Q8_0, Q4_K, Q5_K, or Q6_K. For in-memory quantization of decode-hot weights (projections/heads): the fused GEMV reads the quant bytes directly, so a quantized weight streams 2–4× fewer bytes/token. Keep 1-D norms and host-gathered embed tables unquantized. Returns a new tensor; the source is unchanged.</summary>
    public static Tensor Quantize(Tensor src, DType targetDtype) => QuantizeTensor(src, targetDtype);

    private static unsafe Tensor QuantizeTensor(Tensor src, DType targetDtype)
    {
        IGgufCodec codec = GgufCodecRegistry.Get(targetDtype);
        if (!codec.SupportsQuantize)
            throw new HartsyInference.Core.Exceptions.HartsyInferenceException(
                $"Codec for {targetDtype} does not support quantize. Pick a different target DType (Q8_0, Q4_K, Q5_K, Q6_K are currently supported as quantize targets).");

        Tensor srcF32 = src.DType == DType.F32 ? src : src.CastTo(DType.F32);
        bool ownsSrcF32 = !ReferenceEquals(srcF32, src);
        try
        {
            Tensor result = new Tensor(src.Shape, targetDtype);
            try
            {
                codec.QuantizeFromF32((float*)srcF32.DataPointer, (byte*)result.DataPointer, src.Shape.ElementCount);
                return result;
            }
            catch
            {
                result.Dispose();
                throw;
            }
        }
        finally
        {
            if (ownsSrcF32) srcF32.Dispose();
        }
    }

    private static GgufQuantizationReport ConvertInternal(
        SafeTensorsLoader loader, string outputPath, GgufQuantPolicy policy, string architecture,
        IDictionary<string, object>? extraMetadata)
    {
        Dictionary<string, Tensor> tensors = loader.GetAllTensors();
        return ConvertDictionaryToGguf(tensors, outputPath, policy, architecture, extraMetadata);
    }
}

/// <summary>Statistics from a quantization run: how many tensors took which path, total output size.</summary>
public sealed class GgufQuantizationReport
{
    public int PassthroughCount { get; set; }
    public int CastCount { get; set; }
    public int QuantizedCount { get; set; }
    public long OutputBytes { get; set; }
    public Dictionary<string, int> QuantTotals { get; } = new();
}
