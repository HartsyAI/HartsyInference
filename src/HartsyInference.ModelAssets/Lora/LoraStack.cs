using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;

namespace HartsyInference.ModelAssets.Lora;

/// <summary>Composes one or more LoRAs into a single weight-space delta and applies it to a model's weight dictionary. Use one stack per model component (UNet / Transformer / ClipL / ClipG / TextEncoder2) — each ApplyTo call walks the stacked LoRAs and produces freshly-allocated owned tensors that replace the borrowed mmap entries in the dictionary. The stack owns those new tensors; dispose the stack only after the model is no longer used. An fp8 target is dequantized, merged in F32, and requantized back to fp8 with a recomputed scale (ComfyUI's approach) rather than rejected, so the weight stays on the native fp8 GEMM path; an int8_tensorwise (± ConvRot) target takes the same dequant-merge-requant round trip with a fresh per-row scale, so it stays packed on the resident IMMA path instead of doubling to BF16. A block-quantized target (GGUF, nvfp4) keeps its packed bytes and carries the delta as a <see cref="LowRankAdjunct"/> the GEMM adds at runtime — those codecs have no quantizer to write a merged result back with. Full-weight .diff/.diff_b deltas (Comfy-style Wan repacks) apply through the same passes as W' = W + strength·diff.</summary>
public sealed class LoraStack : IDisposable
{
    private readonly List<Entry> _entries = [];
    private readonly List<Tensor> _ownedMerged = [];
    private readonly List<LoraFile> _ownedFiles = [];
    private int _disposed;

    /// <summary>Adds a LoRA to the stack. The LoraFile remains owned by the caller — keep it alive at least until ApplyTo has been called for every target component.</summary>
    /// <param name="tencStrength">Strength for text-encoder targets; defaults to <paramref name="strength"/>, matching SwarmUI's <c>strength_clip</c> falling back to <c>strength_model</c>.</param>
    public void Add(LoraFile file, float strength = 1.0f, float? tencStrength = null)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ArgumentNullException.ThrowIfNull(file);
        _entries.Add(new Entry(file, strength, tencStrength ?? strength));
    }

    /// <summary>Opens a LoRA safetensors file and adds it to the stack in one call. The stack takes ownership of the loaded file and disposes it when the stack itself is disposed.</summary>
    /// <param name="tencStrength">Strength for text-encoder targets; defaults to <paramref name="strength"/>.</param>
    public void AddFromPath(string filePath, float strength = 1.0f, float? tencStrength = null)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        LoraFile file = LoraFile.Load(filePath);
        _ownedFiles.Add(file);
        _entries.Add(new Entry(file, strength, tencStrength ?? strength));
    }

    /// <summary>Convenience entry point that applies the stack to multiple component weight dictionaries in one call. Pass null for components the model does not have (e.g., a Flux pipeline has no clipG). Returns the total number of weights modified across all components.</summary>
    public int ApplyToWeights(
        IBackend backend,
        IDictionary<string, Tensor>? unetWeights = null,
        IDictionary<string, Tensor>? transformerWeights = null,
        IDictionary<string, Tensor>? clipLWeights = null,
        IDictionary<string, Tensor>? clipGWeights = null,
        IDictionary<string, Tensor>? textEncoder2Weights = null)
    {
        int total = 0;
        if (unetWeights is not null) total += ApplyTo(unetWeights, LoraTarget.UNet, backend);
        else WarnUnservedTarget(LoraTarget.UNet);
        if (transformerWeights is not null) total += ApplyTo(transformerWeights, LoraTarget.Transformer, backend);
        else WarnUnservedTarget(LoraTarget.Transformer);
        if (clipLWeights is not null) total += ApplyTo(clipLWeights, LoraTarget.ClipL, backend);
        else WarnUnservedTarget(LoraTarget.ClipL);
        if (clipGWeights is not null) total += ApplyTo(clipGWeights, LoraTarget.ClipG, backend);
        else WarnUnservedTarget(LoraTarget.ClipG);
        if (textEncoder2Weights is not null) total += ApplyTo(textEncoder2Weights, LoraTarget.TextEncoder2, backend);
        else WarnUnservedTarget(LoraTarget.TextEncoder2);
        return total;
    }

    /// <summary>Warns when the stack holds layers for a component the caller passed no dictionary for.</summary>
    /// <remarks>Silence here is the same failure the zero-match refusal exists for, one component down: the LoRA
    /// applies to the parts the recipe DID pass, the generation succeeds, and the missing arm reads as a weak LoRA.
    /// It warns rather than throws because a LoRA legitimately carrying a text-encoder arm is not a reason to refuse
    /// a pipeline that only exposes its transformer.</remarks>
    private void WarnUnservedTarget(LoraTarget target)
    {
        int layers = 0;
        foreach (Entry entry in _entries)
        {
            foreach (LoraLayer layer in entry.File.Layers)
            {
                if (layer.Target == target) layers++;
            }
            foreach (LoraFullWeightDiff diff in entry.File.FullWeightDiffs)
            {
                if (diff.Target == target) layers++;
            }
        }
        if (layers > 0)
        {
            Logs.Warning($"{layers} LoRA layer(s) target {target}, which this pipeline did not pass a weight "
                + "dictionary for; that part of the LoRA is not applied.");
        }
    }

    /// <summary>Merges every stacked LoRA layer matching the given target into the weight dictionary. Replaces affected entries with freshly-allocated owned tensors. Returns the number of weights modified.</summary>
    public int ApplyTo(IDictionary<string, Tensor> weights, LoraTarget target, IBackend backend)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(backend);

        Dictionary<string, List<(LoraLayer layer, float strength)>> grouped = [];
        foreach (Entry entry in _entries)
        {
            float strength = entry.StrengthFor(target);
            foreach (LoraLayer layer in entry.File.Layers)
            {
                if (layer.Target != target) continue;
                if (!grouped.TryGetValue(layer.TargetKey, out List<(LoraLayer, float)>? list))
                {
                    list = [];
                    grouped[layer.TargetKey] = list;
                }
                list.Add((layer, strength));
            }
        }

        int merged = 0, skippedShape = 0;
        // Adjunct terms are collected per key and attached ONCE at the end: a key can take deltas from both the
        // direct pass and the fused-slice pass, and attaching twice would drop the first attachment's terms.
        Dictionary<string, List<LowRankAdjunctTerm>> adjuncts = [];
        Dictionary<string, List<(int SliceIndex, int SliceCount, LoraLayer Layer, float Strength)>> fusedPending = [];
        foreach ((string canonicalKey, List<(LoraLayer layer, float strength)> deltas) in grouped)
        {
            if (!weights.TryGetValue(canonicalKey, out Tensor? baseW))
            {
                // Fused-QKV fallback: fp8 builds of Flux-lineage checkpoints keep attention fused
                // (attn.qkv / attn.add_qkv) while LoRA canonical keys are the split names. Without this, every
                // attention delta silently missed on those builds and the LoRA came out visibly weakened.
                if (FusedProjectionLayouts.TryResolve(canonicalKey, weights.ContainsKey,
                        out string fusedKey, out int sliceIndex, out int sliceCount))
                {
                    if (!fusedPending.TryGetValue(fusedKey, out List<(int, int, LoraLayer, float)>? slices))
                    {
                        slices = [];
                        fusedPending[fusedKey] = slices;
                    }
                    foreach ((LoraLayer layer, float strength) in deltas)
                    {
                        slices.Add((sliceIndex, sliceCount, layer, strength));
                    }
                    continue;
                }
                Logs.Warning($"LoRA target '{canonicalKey}' not present in {target} weights; skipping.");
                continue;
            }
            DType originalDtype = baseW.DType;
            if (originalDtype.IsFp8 && baseW.Shape.Rank != 2)
            {
                throw new HartsyInferenceException(
                    $"LoRA cannot merge into fp8 weight '{canonicalKey}': requantizing against a per-tensor scale needs "
                    + $"a 2-D Linear weight, got {baseW.Shape} ({originalDtype}).");
            }
            // A LoRA trained against a DIFFERENT build of the same architecture reaches here with the right key and
            // the wrong shape — MiniMax-H3's Turbo LoRA carries the unpruned [96768, 2688] adaln projection while a
            // pruned build stores the [96768, 8] curve-table form. AccumulateDelta would hand that straight to
            // backend.Add against a smaller destination, so the mismatch has to be caught here.
            (LoraLayer badLayer, float _) = deltas.Find(d => !DeltaShapeMatches(d.layer, baseW));
            if (badLayer is not null)
            {
                Logs.Warning($"LoRA delta for '{canonicalKey}' is "
                    + $"[{badLayer.Delta.OutFeatures}, {badLayer.Delta.InFeatures}] but the checkpoint's weight is "
                    + $"{baseW.Shape} — this LoRA was trained against a different build of this architecture; "
                    + "skipping this weight.");
                skippedShape++;
                continue;
            }

            if (RequiresRuntimeAdjunct(baseW))
            {
                List<LowRankAdjunctTerm> terms = AdjunctTermsFor(adjuncts, canonicalKey);
                foreach ((LoraLayer layer, float strength) in deltas)
                {
                    terms.Add(BuildAdjunctTerm(backend, layer.Delta, strength, canonicalKey, baseW, 0, baseW.Shape[0]));
                }
                continue;
            }

            Tensor accumF32 = DequantForMerge(baseW, canonicalKey);
            try
            {
                foreach ((LoraLayer layer, float strength) in deltas)
                {
                    AccumulateDelta(backend, accumF32, layer.Delta, strength);
                }

                weights[canonicalKey] = FinalizeMerged(accumF32, baseW, canonicalKey);
                merged++;
            }
            catch
            {
                accumF32.Dispose();
                throw;
            }
        }

        foreach ((string fusedKey, List<(int SliceIndex, int SliceCount, LoraLayer Layer, float Strength)> slices) in fusedPending)
        {
            Tensor fusedBase = weights[fusedKey];
            long sliceRows = fusedBase.Shape[0] / slices[0].SliceCount;
            bool anyBad = slices.Exists(sl =>
                fusedBase.Shape.Rank != 2 || fusedBase.Shape[0] % sl.SliceCount != 0
                || sl.Layer.Delta.OutFeatures != sliceRows || sl.Layer.Delta.InFeatures != fusedBase.Shape[1]);
            if (anyBad)
            {
                Logs.Warning($"LoRA fused-slice merge into '{fusedKey}' skipped: delta shapes do not tile the fused weight {fusedBase.Shape}.");
                skippedShape++;
                continue;
            }
            (int _, int _, LoraLayer doraLayer, float _) = slices.Find(sl => sl.Layer.Delta.DoraScale is not null);
            if (doraLayer is not null)
            {
                // The magnitude vector describes the adapter's own rows. On the output axis those are a window of the
                // fused weight and could be handled; on the input axis the normalizer is a column norm over ALL rows
                // of the weight, which a slice cannot supply, and picking the branch is the file's choice, not ours.
                throw new NotSupportedException(
                    $"LoRA for '{fusedKey}' is a DoRA adapter targeting one slice of a fused projection. Its magnitude "
                    + "rescaling is defined over the whole weight, so it cannot be applied to a slice of one. Use a "
                    + "build of this model that keeps the projection split, or a plain LoRA.");
            }
            if (RequiresRuntimeAdjunct(fusedBase))
            {
                List<LowRankAdjunctTerm> terms = AdjunctTermsFor(adjuncts, fusedKey);
                foreach ((int sliceIndex, int _, LoraLayer layer, float strength) in slices)
                {
                    terms.Add(BuildAdjunctTerm(backend, layer.Delta, strength, fusedKey, fusedBase,
                        sliceIndex * sliceRows, sliceRows));
                }
                continue;
            }
            Tensor accumF32 = DequantForMerge(fusedBase, fusedKey);
            try
            {
                foreach ((int sliceIndex, int _, LoraLayer layer, float strength) in slices)
                {
                    AccumulateDeltaIntoRows(backend, accumF32, layer.Delta, strength, sliceIndex * sliceRows);
                }
                weights[fusedKey] = FinalizeMerged(accumF32, fusedBase, fusedKey);
                merged++;
            }
            catch
            {
                accumF32.Dispose();
                throw;
            }
        }

        // Full-weight .diff/.diff_b deltas (Comfy-style Wan repacks): W' = W + strength·diff, no decomposition.
        // Bias and weight diffs on the same module carry different canonical keys, so they never collide here, and a
        // key can never collide with the low-rank pass above (.weight low-rank targets are linears, .weight diffs
        // are norms). No fused-QKV fallback: Wan checkpoints keep attention split, and a miss warns by name.
        Dictionary<string, List<(Tensor Diff, float Strength)>> diffGrouped = [];
        foreach (Entry entry in _entries)
        {
            float strength = entry.StrengthFor(target);
            foreach (LoraFullWeightDiff diff in entry.File.FullWeightDiffs)
            {
                if (diff.Target != target) continue;
                if (!diffGrouped.TryGetValue(diff.TargetKey, out List<(Tensor, float)>? list))
                {
                    list = [];
                    diffGrouped[diff.TargetKey] = list;
                }
                list.Add((diff.Diff, strength));
            }
        }
        foreach ((string canonicalKey, List<(Tensor Diff, float Strength)> deltas) in diffGrouped)
        {
            if (!weights.TryGetValue(canonicalKey, out Tensor? baseW))
            {
                Logs.Warning($"LoRA full-weight diff target '{canonicalKey}' not present in {target} weights; skipping.");
                continue;
            }
            DType originalDtype = baseW.DType;
            if (originalDtype.IsFp8 && baseW.Shape.Rank != 2)
            {
                throw new HartsyInferenceException(
                    $"LoRA cannot merge into fp8 weight '{canonicalKey}': requantizing against a per-tensor scale needs "
                    + $"a 2-D Linear weight, got {baseW.Shape} ({originalDtype}).");
            }
            if (deltas.Exists(d => d.Diff.Shape != baseW.Shape))
            {
                Logs.Warning($"LoRA full-weight diff for '{canonicalKey}' does not match the checkpoint's weight shape "
                    + $"{baseW.Shape} — this LoRA was trained against a different build of this architecture; "
                    + "skipping this weight.");
                skippedShape++;
                continue;
            }
            if (RequiresRuntimeAdjunct(baseW))
            {
                RequireRank2AdjunctTarget(baseW, canonicalKey);
                List<LowRankAdjunctTerm> terms = AdjunctTermsFor(adjuncts, canonicalKey);
                foreach ((Tensor diff, float strength) in deltas)
                {
                    // A full-weight diff IS the delta, so it becomes a rank-full term: one GEMM, no factorization.
                    Tensor diffF32 = OwnF32Copy(diff);
                    terms.Add(new LowRankAdjunctTerm { Down = diffF32, Scale = strength });
                }
                continue;
            }
            Tensor accumF32 = DequantForMerge(baseW, canonicalKey);
            try
            {
                foreach ((Tensor diff, float strength) in deltas)
                {
                    Tensor diffF32 = diff.CastTo(DType.F32);
                    try
                    {
                        backend.Scale(diffF32, diffF32, strength);
                        backend.Add(accumF32, accumF32, diffF32);
                    }
                    finally
                    {
                        diffF32.Dispose();
                    }
                }

                weights[canonicalKey] = FinalizeMerged(accumF32, baseW, canonicalKey);
                merged++;
            }
            catch
            {
                accumF32.Dispose();
                throw;
            }
        }

        foreach ((string canonicalKey, List<LowRankAdjunctTerm> terms) in adjuncts)
        {
            weights[canonicalKey] = AttachAdjunct(weights[canonicalKey], terms, canonicalKey);
            merged++;
        }

        if (merged > 0 || skippedShape > 0)
        {
            // The skipped count is load-bearing, not decoration: a partially applied LoRA still generates, so a
            // silent skip reads as success while the LoRA does only part of its job.
            string skipped = skippedShape > 0 ? $" ({skippedShape} skipped on shape mismatch)" : "";
            Logs.Info($"Merged {merged} of {grouped.Count + diffGrouped.Count} LoRA-targeted weights into {target}{skipped}.");
        }
        return merged;
    }

    /// <summary>Whether this weight is a packed ComfyUI <c>int8_tensorwise</c> Linear (± ConvRot) that can take the dequant-merge-requant round trip.</summary>
    private static bool IsInt8Tensorwise(Tensor weight) =>
        weight.DType == DType.I8 && weight.QuantInfo is { Format: "int8_tensorwise", RowScale: not null };

    /// <summary>Whether this weight's delta must live on the weight at runtime instead of being merged into it.</summary>
    /// <remarks>Block-quantized formats only. The classic GGUF codecs an H3/Wan community build uses (Q4_0/Q5_0/Q5_1)
    /// have no quantizer at all, and even where one exists, requantizing a merged result degrades the base as well as
    /// the LoRA. fp8 and int8_tensorwise are excluded on purpose: both have a working requantizer, and the round trip
    /// keeps them on their native packed GEMM path — which is also what ComfyUI does for those two.</remarks>
    private static bool RequiresRuntimeAdjunct(Tensor weight) => weight.DType.IsQuantized;

    /// <summary>Refuses a block-quantized target that is not a 2-D Linear weight — a quantized rank-4 convolution has no GEMM to hang an adjunct off.</summary>
    private static void RequireRank2AdjunctTarget(Tensor baseW, string canonicalKey)
    {
        if (baseW.Shape.Rank != 2)
        {
            throw new NotSupportedException(
                $"LoRA cannot patch '{canonicalKey}': it is a {baseW.DType.Name} {baseW.Shape} weight, and a runtime "
                + "LoRA adjunct applies only to 2-D Linear weights. Use a dense build of this model with LoRAs.");
        }
    }

    private static List<LowRankAdjunctTerm> AdjunctTermsFor(Dictionary<string, List<LowRankAdjunctTerm>> adjuncts, string key)
    {
        if (!adjuncts.TryGetValue(key, out List<LowRankAdjunctTerm>? terms))
        {
            terms = [];
            adjuncts[key] = terms;
        }
        return terms;
    }

    /// <summary>Builds one runtime term for <paramref name="delta"/>, zero-padded to the fused weight's full row count when it covers only a slice of it.</summary>
    /// <remarks>Padding rather than carrying a row offset is what keeps the term usable by the windowed GEMM path:
    /// <c>LinearWeightRows</c> narrows the up matrix with the window and takes the down matrix whole, which only works
    /// if the up matrix is indexed by the same rows the weight is. At LoRA ranks the padded rows cost nothing.</remarks>
    private LowRankAdjunctTerm BuildAdjunctTerm(IBackend backend, LoraDelta delta, float strength, string canonicalKey,
        Tensor baseW, long rowOffset, long rowCount)
    {
        RequireRank2AdjunctTarget(baseW, canonicalKey);
        if (delta.DoraScale is not null)
        {
            // DoRA rescales the whole (W + ΔW) row-wise, so it is not an additive term and cannot be carried here.
            throw new NotSupportedException(
                $"LoRA '{canonicalKey}' is a DoRA adapter, whose magnitude rescaling is not additive and so cannot be "
                + $"applied at runtime to a packed {baseW.DType.Name} weight. Use a dense or fp8 build of this model.");
        }
        long fullRows = baseW.Shape[0];
        if (delta is StandardLoraDelta standard)
        {
            Tensor down = OwnF32Flattened(standard.Down);
            Tensor up = PadRowsToFull(OwnF32Flattened(standard.Up), fullRows, rowOffset, rowCount);
            return new LowRankAdjunctTerm { Down = down, Up = up, Scale = strength * delta.Scale };
        }
        // LoHa/LoKr have no low-rank pair to carry, so their ΔW materializes once here and rides as a rank-full term.
        Tensor full = delta.ComputeF32(backend);
        _ownedMerged.Add(full);
        return new LowRankAdjunctTerm
        {
            Down = PadRowsToFull(full, fullRows, rowOffset, rowCount),
            Scale = strength * delta.Scale,
        };
    }

    /// <summary>Replaces the dictionary entry with a new tensor aliasing the same bytes and carrying <paramref name="terms"/>, registering it as stack-owned.</summary>
    private Tensor AttachAdjunct(Tensor baseW, List<LowRankAdjunctTerm> terms, string canonicalKey)
    {
        LowRankAdjunct adjunct = new() { Terms = terms };
        adjunct.Validate(baseW.Shape[0], baseW.Shape[1], canonicalKey);
        Tensor patched = baseW.WithLowRankAdjunct(adjunct);
        _ownedMerged.Add(patched);
        return patched;
    }

    /// <summary>Returns a stack-owned F32 copy of <paramref name="source"/> with its trailing axes folded into the columns.</summary>
    private Tensor OwnF32Flattened(Tensor source)
    {
        Tensor f32 = source.CastTo(DType.F32);
        if (f32.Shape.Rank == 2)
        {
            _ownedMerged.Add(f32);
            return f32;
        }
        try
        {
            Tensor flat = new(new TensorShape(f32.Shape[0], f32.Shape.ElementCount / f32.Shape[0]), DType.F32);
            f32.AsReadOnlySpan<float>().CopyTo(flat.AsSpan<float>());
            _ownedMerged.Add(flat);
            return flat;
        }
        finally
        {
            f32.Dispose();
        }
    }

    /// <summary>Returns a stack-owned F32 copy of <paramref name="source"/>, shape unchanged.</summary>
    private Tensor OwnF32Copy(Tensor source)
    {
        Tensor f32 = source.CastTo(DType.F32);
        _ownedMerged.Add(f32);
        return f32;
    }

    /// <summary>Grows a slice-sized matrix to the fused weight's full row count, zero outside the slice; returns it unchanged when it already spans every row.</summary>
    private Tensor PadRowsToFull(Tensor sliceRows, long fullRows, long rowOffset, long rowCount)
    {
        if (rowOffset == 0 && rowCount == fullRows)
        {
            return sliceRows;
        }
        long columns = sliceRows.Shape[1];
        Tensor padded = new(new TensorShape(fullRows, columns), DType.F32);
        try
        {
            Span<float> destination = padded.AsSpan<float>();
            destination.Clear();
            sliceRows.AsReadOnlySpan<float>().CopyTo(destination[(int)(rowOffset * columns)..]);
            _ownedMerged.Add(padded);
            return padded;
        }
        catch
        {
            padded.Dispose();
            throw;
        }
    }

    /// <summary>Produces the owned F32 accumulator a merge mutates in place, refusing by name any quantized format that cannot be requantized afterwards.</summary>
    private static Tensor DequantForMerge(Tensor baseW, string canonicalKey)
    {
        if (IsInt8Tensorwise(baseW))
        {
            QuantWeightInfo info = baseW.QuantInfo!;
            using Tensor bf16 = Int8ConvRotCodec.DequantToBf16(baseW, info.RowScale!, info.ConvRotGroupSize);
            return bf16.CastTo(DType.F32);
        }
        if (baseW.DType == DType.I8)
        {
            // DType.I8 has IsQuantized false, so without this the merge dies in CastTo's raw "I8 → F32" throw.
            throw new NotSupportedException(
                $"LoRA weights can't be merged into int8 weight '{canonicalKey}': it carries "
                + (baseW.QuantInfo is null ? "no quantization descriptor, so its dequant scale is unknowable."
                    : $"unsupported quantization format '{baseW.QuantInfo.Format}'.")
                + " Use a BF16/fp8_scaled build of this model with LoRAs, or remove the LoRA.");
        }
        // CastTo folds Fp8ScaleFactor into the values and returns factor 1.0, so the merge stays quant-unaware.
        return baseW.CastTo(DType.F32);
    }

    /// <summary>Converts the merged F32 accumulator back to the base weight's storage form, registers the result (and any companion scale) as stack-owned, and disposes the accumulator when it is not itself the result.</summary>
    private Tensor FinalizeMerged(Tensor accumF32, Tensor baseW, string canonicalKey)
    {
        Tensor finalTensor;
        if (IsInt8Tensorwise(baseW))
        {
            if (baseW.Shape.Rank != 2)
            {
                // Reachable only since convolution targets were allowed through. ConvRot's rotation is defined
                // over a 2-D weight's rows and its codec refuses anything else — correctly, but with a message
                // about its own contract rather than the user's situation.
                accumF32.Dispose();
                throw new NotSupportedException(
                    $"'{canonicalKey}' is an int8 convolution weight. The ConvRot rotation a LoRA merge has to "
                    + "reverse and reapply is defined over a 2-D weight, so it cannot be repacked at rank "
                    + $"{baseW.Shape.Rank}. Use a BF16/fp8 build of this model with convolution LoRAs.");
            }
            finalTensor = RequantizeF32ToInt8ConvRot(accumF32, baseW.QuantInfo!);
        }
        else if (baseW.DType.IsFp8)
        {
            finalTensor = CheckpointConvertUtils.QuantizeF32ToFp8Scaled(accumF32, canonicalKey);
            // A weight-side LoRA must not change activation scaling: the input scale is carried, not recomputed.
            finalTensor.Fp8InputScaleFactor = baseW.Fp8InputScaleFactor;
        }
        else
        {
            finalTensor = baseW.DType == DType.F32 ? accumF32 : accumF32.CastTo(baseW.DType);
        }
        if (!ReferenceEquals(finalTensor, accumF32))
        {
            accumF32.Dispose();
        }
        _ownedMerged.Add(finalTensor);
        return finalTensor;
    }

    /// <summary>Packs the merged rows back into ConvRot storage through <see cref="Int8ConvRotCodec.QuantizeFromF32"/> and re-attaches the descriptor. The fresh RowScale is stack-owned, unlike the base weight's, which is borrowed from the loader.</summary>
    private Tensor RequantizeF32ToInt8ConvRot(Tensor accumF32, QuantWeightInfo info)
    {
        (Tensor quantized, Tensor rowScale) = Int8ConvRotCodec.QuantizeFromF32(accumF32, info.ConvRotGroupSize);
        quantized.QuantInfo = new QuantWeightInfo
        {
            Format = info.Format,
            RowScale = rowScale,
            ConvRotGroupSize = info.ConvRotGroupSize,
            FullPrecisionMatMul = info.FullPrecisionMatMul,
        };
        _ownedMerged.Add(rowScale);
        return quantized;
    }

    /// <summary>Whether this layer's ΔW is the shape of the weight it would be added to.
    /// <para>Rank 2 is a Linear; rank 4 is a convolution, where the delta arrives flattened to
    /// <c>[out, in·kh·kw]</c>. Those are the SAME memory in row-major order — element <c>(o, i, kh, kw)</c> of the
    /// weight sits at the same offset as <c>(o, i·kh·kw + …)</c> of the delta — so the add lands on the right
    /// elements once the shapes are made to agree. Anything else is refused.</para></summary>
    private static bool DeltaShapeMatches(LoraLayer layer, Tensor baseW) =>
        baseW.Shape.Rank is 2 or 4 && layer.Delta.MatchesShape(baseW);

    /// <summary>Folds one layer's ΔW into <paramref name="accumF32"/>, which holds the weight itself rather than a delta.</summary>
    /// <remarks>A DoRA adapter is not additive — the magnitude vector rescales the whole LoRA'd weight row- or
    /// column-wise — so it takes <see cref="LoraDoraDecompose"/> instead of the scale-and-add, and needs the base
    /// weight this accumulator already carries. Stacking several adapters on one weight applies them in sequence,
    /// each decomposing against the result of the last, which is what ComfyUI does patch by patch.</remarks>
    private static void AccumulateDelta(IBackend backend, Tensor accumF32, LoraDelta delta, float strength)
    {
        Tensor deltaF32 = delta.ComputeF32(backend);
        Tensor? flatView = null;
        try
        {
            if (delta.DoraScale is not null)
            {
                if (accumF32.Shape.Rank != 2)
                {
                    // The magnitude vector normalizes by a row norm of the weight-as-a-matrix. A convolution has no
                    // such matrix until it is flattened, and which axis the vector describes is the file's choice,
                    // not ours — so this is refused rather than guessed at.
                    throw new NotSupportedException(
                        "A DoRA adapter targets a convolution weight. Its magnitude rescaling is defined over a "
                        + "2-D weight, so it cannot be applied to a rank-4 one. Use a plain LoRA for this model.");
                }
                LoraDoraDecompose.Apply(accumF32, deltaF32, delta.DoraScale, delta.Scale, strength);
                return;
            }
            backend.Scale(deltaF32, deltaF32, strength * delta.Scale);
            // A convolution's ΔW arrives flattened to [out, in·kh·kw] while the weight is [out, in, kh, kw]. Same
            // bytes in the same order, so the add only needs the two to agree on a shape; the view costs nothing.
            Tensor addend = deltaF32;
            if (accumF32.Shape.Rank != deltaF32.Shape.Rank)
            {
                flatView = deltaF32.Reshape(accumF32.Shape);
                addend = flatView;
            }
            backend.Add(accumF32, accumF32, addend);
        }
        finally
        {
            flatView?.Dispose();
            deltaF32.Dispose();
        }
    }

    /// <summary>Like <see cref="AccumulateDelta"/> but adds ΔW into the row window [<paramref name="rowOffset"/>, rowOffset + delta.OutFeatures) of <paramref name="accumF32"/> — the fused-QKV slice merge. Host math: F32 tensors are host-resident at this point and the add is a one-time load-path cost.</summary>
    private static unsafe void AccumulateDeltaIntoRows(IBackend backend, Tensor accumF32, LoraDelta delta,
        float strength, long rowOffset)
    {
        Tensor deltaF32 = delta.ComputeF32(backend);
        try
        {
            float scale = strength * delta.Scale;
            long rows = deltaF32.Shape[0];
            long cols = deltaF32.Shape[1];
            float* dp = (float*)deltaF32.DataPointer;
            float* ap = (float*)accumF32.DataPointer;
            for (long r = 0; r < rows; r++)
            {
                long srcBase = r * cols;
                long dstBase = (rowOffset + r) * cols;
                for (long c = 0; c < cols; c++)
                {
                    ap[dstBase + c] += dp[srcBase + c] * scale;
                }
            }
        }
        finally
        {
            deltaF32.Dispose();
        }
    }

    /// <summary>Disposes every merged tensor allocated by ApplyTo and every LoraFile loaded via AddFromPath. Only call this after the model that consumed the merged weights is no longer in use.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (Tensor t in _ownedMerged)
        {
            t.Dispose();
        }
        _ownedMerged.Clear();
        foreach (LoraFile f in _ownedFiles)
        {
            f.Dispose();
        }
        _ownedFiles.Clear();
    }

    /// <summary>One stacked LoRA and the strengths it applies at, split the way SwarmUI splits them.</summary>
    private readonly record struct Entry(LoraFile File, float Strength, float TencStrength)
    {
        /// <summary>The strength this entry applies to <paramref name="target"/>: the text-encoder strength for every encoder component, the model strength for the diffusion body.</summary>
        public float StrengthFor(LoraTarget target) => target switch
        {
            LoraTarget.ClipL or LoraTarget.ClipG or LoraTarget.TextEncoder2 => TencStrength,
            _ => Strength,
        };
    }
}
