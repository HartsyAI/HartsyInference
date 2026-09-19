using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Gguf;

namespace HartsyInference.ModelAssets.Quant;

/// <summary>Which container and precision a quantization job writes.</summary>
public enum QuantizationTargetKind
{
    /// <summary>A GGUF file, per-tensor precision chosen by a <see cref="GgufQuantPolicy"/>.</summary>
    Gguf,
}

/// <summary>The output format for a <see cref="QuantizationJob"/>.</summary>
/// <param name="Kind">Container to write.</param>
/// <param name="Policy">Per-tensor precision rules; required for <see cref="QuantizationTargetKind.Gguf"/>.</param>
public sealed record QuantizationTarget(QuantizationTargetKind Kind, GgufQuantPolicy? Policy = null);

/// <summary>One offline quantization request. Precision is an argument here rather than a knob on purpose: knob
/// domains are Numerics/Vram/Diagnostics/Paths, and "what precision is this file" is none of those.</summary>
public sealed record QuantizationJob
{
    /// <summary>Checkpoint to read. Any container <see cref="CheckpointSource"/> opens, including a GGUF being
    /// re-quantized to a smaller one.</summary>
    public required string SourcePath { get; init; }

    /// <summary>File to write. Refused if it already exists unless <see cref="Overwrite"/> is set.</summary>
    public required string OutputPath { get; init; }

    /// <summary>Container and precision.</summary>
    public required QuantizationTarget Target { get; init; }

    /// <summary>Value for the output's <c>general.architecture</c>, which is how a reader picks a key mapper.
    /// Defaults to the source's own when it has one.</summary>
    public string? Architecture { get; init; }

    /// <summary>Whether an existing output may be replaced.</summary>
    public bool Overwrite { get; init; }
}

/// <summary>What a finished job produced.</summary>
public sealed record QuantizationReport
{
    /// <summary>Tensors written.</summary>
    public required int TensorCount { get; init; }

    /// <summary>Tensors that were quantized, as opposed to passed through wide.</summary>
    public required int QuantizedCount { get; init; }

    /// <summary>Input file size in bytes.</summary>
    public required long SourceBytes { get; init; }

    /// <summary>Output file size in bytes.</summary>
    public required long OutputBytes { get; init; }

    /// <summary>Output size as a fraction of input.</summary>
    public double Ratio => SourceBytes > 0 ? (double)OutputBytes / SourceBytes : 0;
}

/// <summary>Writes a quantized copy of a checkpoint, offline. This is deliberately NOT a load-time path: the engine
/// never quantizes on the way in, so what a generation runs is the file on disk and nothing else.</summary>
public static class CheckpointQuantizer
{
    /// <summary>Quantizes <paramref name="job"/>'s source into its output and reports what was written.</summary>
    public static QuantizationReport Quantize(QuantizationJob job, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(job);
        if (!File.Exists(job.SourcePath))
            throw new FileNotFoundException("Checkpoint to quantize not found.", job.SourcePath);
        if (File.Exists(job.OutputPath) && !job.Overwrite)
            throw new HartsyInferenceException(
                $"'{job.OutputPath}' already exists. Pass --overwrite to replace it.");
        if (Path.GetFullPath(job.SourcePath) == Path.GetFullPath(job.OutputPath))
            throw new HartsyInferenceException("Source and output are the same file.");
        if (job.Target.Kind != QuantizationTargetKind.Gguf)
            throw new NotSupportedException($"Quantization target '{job.Target.Kind}' is not implemented yet.");
        GgufQuantPolicy policy = job.Target.Policy
            ?? throw new HartsyInferenceException("A GGUF target needs a quantization policy.");

        // Through the container, not SafeTensorsLoader: a fp8_scaled or int8 checkpoint carries its scales in
        // companion tensors, and quantizing the raw values without folding them first produces a file that is
        // wrong by whatever those scales were. Opening this way also lets a GGUF be re-quantized.
        using CheckpointSource source = CheckpointSource.Open(job.SourcePath);
        string architecture = job.Architecture ?? "unknown";
        Logs.Info($"[Quantize] {Path.GetFileName(job.SourcePath)} ({source.Format}, {source.Weights.Count} tensors) "
            + $"→ {policy.BackboneDType.Name} GGUF, architecture '{architecture}'.");

        Dictionary<string, Tensor> dense = new(source.Weights.Count, StringComparer.Ordinal);
        List<Tensor> owned = new();
        try
        {
            foreach (KeyValuePair<string, Tensor> kv in source.Weights)
            {
                cancel.ThrowIfCancellationRequested();
                Tensor wide = MaterializeF32(kv.Value, kv.Key, owned);
                dense[kv.Key] = wide;
            }
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(job.OutputPath))!);
            GgufQuantizationReport inner =
                GgufQuantizer.ConvertDictionaryToGguf(dense, job.OutputPath, policy, architecture);
            return new QuantizationReport
            {
                TensorCount = inner.QuantizedCount + inner.PassthroughCount + inner.CastCount,
                QuantizedCount = inner.QuantizedCount,
                SourceBytes = new FileInfo(job.SourcePath).Length,
                OutputBytes = new FileInfo(job.OutputPath).Length,
            };
        }
        finally
        {
            foreach (Tensor t in owned) t.Dispose();
        }
    }

    /// <summary>Gets a tensor to plain F32 whatever form it arrived in, so the quantizer only ever sees real values.
    /// <para>The three forms need three different routes and picking the wrong one is silent: <c>CastTo</c> folds an
    /// <see cref="Tensor.Fp8ScaleFactor"/> into the values, but REFUSES a block-quantized source by design, because
    /// decoding a block layout is the dequantizer's job rather than a dtype conversion's.</para></summary>
    private static Tensor MaterializeF32(Tensor weight, string key, List<Tensor> owned)
    {
        if (weight.DType == DType.F32 && weight.Fp8ScaleFactor == 1.0f && weight.QuantInfo is null)
        {
            return weight;
        }
        if (weight.QuantInfo is { RowScale: not null } info)
        {
            using Tensor bf16 = Core.Tensors.Int8ConvRotCodec.DequantToBf16(weight, info.RowScale!, info.ConvRotGroupSize);
            Tensor f32 = bf16.CastTo(DType.F32);
            owned.Add(f32);
            return f32;
        }
        if (weight.DType.IsQuantized)
        {
            Tensor f32 = GgufDequantizer.Dequantize(weight, DType.F32);
            owned.Add(f32);
            return f32;
        }
        if (weight.DType == DType.I8)
        {
            throw new HartsyInferenceException(
                $"'{key}' is int8 with no quantization descriptor, so its dequant scale is unknowable. "
                + "Quantize from a BF16/fp8_scaled build instead.");
        }
        Tensor cast = weight.CastTo(DType.F32);
        owned.Add(cast);
        return cast;
    }
}
