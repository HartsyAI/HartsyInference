using SharpInference.Core.Logging;
using SharpInference.Core.Tensors;
using SharpInference.ModelHandler.Gguf.Codecs;
using SharpInference.ModelHandler.SafeTensors;

namespace SharpInference.ModelHandler.Gguf;

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
        writer.SetMetadata("general.name", $"{architecture} (SharpInference quantized)");
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

    private static unsafe Tensor QuantizeTensor(Tensor src, DType targetDtype)
    {
        IGgufCodec codec = GgufCodecRegistry.Get(targetDtype);
        if (!codec.SupportsQuantize)
            throw new SharpInference.Core.Exceptions.SharpInferenceException(
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
