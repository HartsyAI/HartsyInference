using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.Quant;

/// <summary>Which container and precision a quantization job writes.</summary>
public enum QuantizationTargetKind
{
    /// <summary>A GGUF file, per-tensor precision chosen by a <see cref="GgufQuantPolicy"/>.</summary>
    Gguf,

    /// <summary>A safetensors file in ComfyUI's <c>fp8_scaled</c> shape: each eligible weight stored as F8E4M3
    /// beside a <c>.scale_weight</c> companion holding the scalar it was divided by.</summary>
    Fp8Scaled,

    /// <summary>A safetensors file in ComfyUI's <c>int8</c> shape with the convolution rotation applied: each
    /// eligible weight stored as I8 beside a <c>[rows,1]</c> <c>.weight_scale</c> and a <c>.comfy_quant</c>
    /// descriptor naming the format, which is what a reader trusts rather than the file-level mirror.</summary>
    Int8ConvRot,
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
        if (job.Target.Kind == QuantizationTargetKind.Gguf && job.Target.Policy is null)
            throw new HartsyInferenceException("A GGUF target needs a quantization policy.");

        // Through the container, not SafeTensorsLoader: a fp8_scaled or int8 checkpoint carries its scales in
        // companion tensors, and quantizing the raw values without folding them first produces a file that is
        // wrong by whatever those scales were. Opening this way also lets a GGUF be re-quantized.
        using CheckpointSource source = CheckpointSource.Open(job.SourcePath);
        string architecture = job.Architecture ?? "unknown";
        string targetLabel = job.Target.Kind == QuantizationTargetKind.Gguf
            ? $"{job.Target.Policy!.BackboneDType.Name} GGUF (architecture '{architecture}')"
            : job.Target.Kind.ToString();
        Logs.Info($"[Quantize] {Path.GetFileName(job.SourcePath)} ({source.Format}, {source.Weights.Count} tensors) "
            + $"→ {targetLabel}.");

        // Only the safetensors targets still widen the whole checkpoint at once; GGUF streams, so it is exempt.
        if (job.Target.Kind != QuantizationTargetKind.Gguf)
        {
            RefuseIfWorkingSetWontFit(source, job.SourcePath);
        }

        Dictionary<string, Tensor> dense = new(source.Weights.Count, StringComparer.Ordinal);
        List<Tensor> owned = new();
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(job.OutputPath))!);
            int written, quantized;
            if (job.Target.Kind == QuantizationTargetKind.Gguf)
            {
                (written, quantized) = WriteGguf(job, source, architecture, cancel);
            }
            else
            {
                foreach (KeyValuePair<string, Tensor> kv in source.Weights)
                {
                    cancel.ThrowIfCancellationRequested();
                    dense[kv.Key] = MaterializeF32(kv.Value, kv.Key, owned);
                }
                (written, quantized) = WriteSafetensors(job, dense, owned);
            }
            return new QuantizationReport
            {
                TensorCount = written,
                QuantizedCount = quantized,
                SourceBytes = new FileInfo(job.SourcePath).Length,
                OutputBytes = new FileInfo(job.OutputPath).Length,
            };
        }
        finally
        {
            foreach (Tensor t in owned) t.Dispose();
        }
    }

    /// <summary>Quantizes straight into the writer, one tensor at a time.
    /// <para>Interleaving is what makes a large source possible at all. Widening every tensor first needs the
    /// WHOLE checkpoint as F32 at once — a 13 GB Q4_K build wants about 74 GiB and the process is OOM-killed
    /// rather than slow. Materializing per tensor and freeing each wide copy the moment its quantized form exists
    /// holds one instead of all of them; what remains is the output, which the writer keeps until <c>Flush</c>
    /// by design.</para></summary>
    private static (int Written, int Quantized) WriteGguf(
        QuantizationJob job, CheckpointSource source, string architecture, CancellationToken cancel)
    {
        GgufQuantPolicy policy = job.Target.Policy!;
        using GgufWriter writer = new(job.OutputPath);
        writer.SetMetadata("general.architecture", architecture);
        writer.SetMetadata("general.name", $"{architecture} (HartsyInference quantized)");
        List<Tensor> pending = new();
        int written = 0, quantized = 0;
        try
        {
            foreach (KeyValuePair<string, Tensor> kv in source.Weights)
            {
                cancel.ThrowIfCancellationRequested();
                List<Tensor> scratch = new(1);
                Tensor wide = MaterializeF32(kv.Value, kv.Key, scratch);
                DType target = policy.ResolveTargetDType(kv.Key, wide);
                Tensor toWrite = target == wide.DType ? wide
                    : target.IsQuantized ? GgufQuantizer.Quantize(wide, target)
                    : wide.CastTo(target);
                if (target.IsQuantized) quantized++;
                writer.AddTensor(kv.Key, toWrite);
                // `scratch` holds only what MaterializeF32 CREATED — empty when the tensor was already F32, in
                // which case `wide` is the container's mmap view and freeing it would pull the mapping out from
                // under the writer.
                foreach (Tensor created in scratch)
                {
                    if (ReferenceEquals(created, toWrite)) pending.Add(created);
                    else created.Dispose();
                }
                if (!ReferenceEquals(toWrite, wide)) pending.Add(toWrite);
                written++;
            }
            writer.Flush();
        }
        finally
        {
            foreach (Tensor t in pending) t.Dispose();
        }
        return (written, quantized);
    }

    /// <summary>Refuses a source whose F32 working set will not fit, by name and with the numbers.
    /// <para>Every tensor is widened to F32 before the writer sees it, and the writer holds its output until
    /// <c>Flush</c>, so the peak is roughly the whole checkpoint as F32 plus the whole output. A 13 GB Q4_K source
    /// wants about 50 GB and gets the process OOM-killed — no message, no partial file, nothing to read. An
    /// up-front refusal that names the requirement is worth more than a kill, and this is measured from the real
    /// element counts rather than the file size, because a block-quantized source is several times its own size
    /// once widened. Passing it is a necessary condition, not a guarantee: another process can take the memory
    /// between this check and the allocation.</para>
    /// <para>Interleaving the widen with the write would hold one tensor instead of all of them and lift this
    /// entirely. That is the right fix and is not this one: a first attempt at it aborted in the allocator, and a
    /// memory rewrite wants verification time rather than confidence.</para></summary>
    private static void RefuseIfWorkingSetWontFit(CheckpointSource source, string sourcePath)
    {
        long f32Bytes = 0;
        foreach (KeyValuePair<string, Tensor> kv in source.Weights)
        {
            f32Bytes = checked(f32Bytes + (kv.Value.ElementCount * sizeof(float)));
        }
        // The output roughly tracks the source on disk; the F32 intermediate is what actually varies.
        long needed = f32Bytes + new FileInfo(sourcePath).Length;
        long available = AvailableMemoryBytes();
        if (available <= 0 || needed <= available)
        {
            return;
        }
        throw new HartsyInferenceException(
            $"Quantizing '{Path.GetFileName(sourcePath)}' needs about {needed / (1024L * 1024 * 1024)} GiB of RAM — "
            + $"every tensor is widened to F32 first, and this checkpoint is {f32Bytes / (1024L * 1024 * 1024)} GiB "
            + $"wide — but only about {available / (1024L * 1024 * 1024)} GiB is free. Quantize from a smaller "
            + "source, or from the dense build this one was made from.");
    }

    /// <summary>Memory this process could actually get, not what the machine has.
    /// <para><c>GC.GetGCMemoryInfo().TotalAvailableMemoryBytes</c> is the wrong number here: with no cgroup limit
    /// it reports total physical RAM, so on a box with several GB already resident it promises a fit it cannot
    /// deliver and the caller is OOM-killed anyway — the failure this check exists to replace. Linux publishes
    /// the honest figure as <c>MemAvailable</c>; elsewhere the GC's number is used with the same caveat, which is
    /// why passing this check is a necessary condition rather than a guarantee.</para></summary>
    private static long AvailableMemoryBytes()
    {
        try
        {
            foreach (string line in File.ReadLines("/proc/meminfo"))
            {
                if (!line.StartsWith("MemAvailable:", StringComparison.Ordinal)) continue;
                string[] parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length >= 2 && long.TryParse(parts[1], out long kb)) return kb * 1024;
            }
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or NotSupportedException)
        {
            // Not Linux, or /proc is not readable. Fall through.
        }
        return GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
    }

    /// <summary>Writes the two ComfyUI safetensors shapes. Both quantize only what they can: a weight that is not
    /// an eligible rank-2 <c>.weight</c>, or is too small to be worth it, is written wide rather than forced — the
    /// same rule the GGUF policy applies, and the reason a quantized file still carries F32 norms and biases.</summary>
    private static (int Written, int Quantized) WriteSafetensors(
        QuantizationJob job, Dictionary<string, Tensor> dense, List<Tensor> owned)
    {
        Dictionary<string, Tensor> output = new(dense.Count, StringComparer.Ordinal);
        int quantized = 0;
        foreach (KeyValuePair<string, Tensor> kv in dense)
        {
            if (job.Target.Kind == QuantizationTargetKind.Fp8Scaled)
            {
                // The helper takes BF16/F16 because that is what a published fp8_scaled build is made from; our
                // dense copy is F32, so it is narrowed first and the narrowed copy is what gets stored if the
                // weight turns out ineligible.
                Tensor half = kv.Value.DType == DType.BF16 ? kv.Value : kv.Value.CastTo(DType.BF16);
                if (!ReferenceEquals(half, kv.Value)) owned.Add(half);
                // Claim ownership of exactly what THIS call added, by diffing against the keys already present.
                // Rescanning the whole output instead both re-registered every earlier companion on each pass and
                // missed the fp8 weight itself — it reuses the source's key, so a "not already in dense" test
                // excludes it and nothing ever frees it.
                string[] before = [.. output.Keys];
                if (CheckpointConverters.Utils.CheckpointConvertUtils.TryQuantizeWeightToFp8(output, kv.Key, half))
                {
                    HashSet<string> existing = new(before, StringComparer.Ordinal);
                    foreach (KeyValuePair<string, Tensor> produced in output)
                    {
                        if (!existing.Contains(produced.Key) && !ReferenceEquals(produced.Value, half))
                        {
                            owned.Add(produced.Value);
                        }
                    }
                    quantized++;
                    continue;
                }
                output[kv.Key] = half;
                continue;
            }
            if (IsInt8Eligible(kv.Key, kv.Value))
            {
                // QuantizeFromF32 rotates in place, so it gets a copy rather than the container's mapped tensor.
                Tensor scratch = kv.Value.CastTo(DType.F32);
                (Tensor weight, Tensor rowScale) = Core.Tensors.Int8ConvRotCodec.QuantizeFromF32(scratch, ConvRotGroup);
                scratch.Dispose();
                owned.Add(weight);
                owned.Add(rowScale);
                // [rows,1] rather than the codec's flat [rows]: that is the shape published ComfyUI int8 repacks
                // carry, and interoperating with those is the whole reason to write this format. Our own reader
                // goes by element count either way.
                Tensor shapedScale = rowScale.Reshape(new TensorShape(rowScale.ElementCount, 1));
                owned.Add(shapedScale);
                output[kv.Key] = weight;
                output[kv.Key[..^".weight".Length] + ".weight_scale"] = shapedScale;
                byte[] blob = new ComfyQuantDescriptor
                {
                    Format = "int8_tensorwise",
                    ConvRotGroupSize = ConvRotGroup,
                }.Serialize();
                Tensor descriptor = new Tensor(new TensorShape(blob.Length), DType.U8);
                blob.CopyTo(descriptor.AsSpan<byte>());
                owned.Add(descriptor);
                output[kv.Key[..^".weight".Length] + ComfyQuantDescriptor.Suffix] = descriptor;
                quantized++;
                continue;
            }
            output[kv.Key] = kv.Value;
        }
        SafeTensorsWriter.Save(job.OutputPath, output);
        return (output.Count, quantized);
    }

    /// <summary>ConvRot's group size. 256 is what the published ComfyUI int8 repacks use, and a reader takes it
    /// from the descriptor rather than assuming, so this only has to be self-consistent with what we write.</summary>
    private const int ConvRotGroup = 256;

    /// <summary>Whether a tensor is worth storing as int8: a rank-2 <c>.weight</c> whose row length divides the
    /// rotation group, since a partial group cannot be rotated.</summary>
    private static bool IsInt8Eligible(string key, Tensor tensor) =>
        key.EndsWith(".weight", StringComparison.Ordinal)
        && tensor.Shape.Rank == 2
        && tensor.ElementCount >= (1L << 20)
        && tensor.Shape[1] % ConvRotGroup == 0;

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
