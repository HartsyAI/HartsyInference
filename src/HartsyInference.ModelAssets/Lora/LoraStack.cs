using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;

namespace HartsyInference.ModelAssets.Lora;

/// <summary>Composes one or more LoRAs into a single weight-space delta and merges it into a model's weight dictionary. Use one stack per model component (UNet / Transformer / ClipL / ClipG) — each ApplyTo call walks the stacked LoRAs and produces freshly-allocated owned tensors that replace the borrowed mmap entries in the dictionary. The stack owns those new tensors; dispose the stack only after the model is no longer used. An fp8 target is dequantized, merged in F32, and requantized back to fp8 with a recomputed scale (ComfyUI's approach) rather than rejected, so the weight stays on the native fp8 GEMM path. Full-weight .diff/.diff_b deltas (Comfy-style Wan repacks) merge through the same pass as W' = W + strength·diff.</summary>
public sealed class LoraStack : IDisposable
{
    private readonly List<Entry> _entries = [];
    private readonly List<Tensor> _ownedMerged = [];
    private readonly List<LoraFile> _ownedFiles = [];
    private int _disposed;

    /// <summary>Adds a LoRA to the stack. The LoraFile remains owned by the caller — keep it alive at least until ApplyTo has been called for every target component.</summary>
    public void Add(LoraFile file, float strength = 1.0f)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ArgumentNullException.ThrowIfNull(file);
        _entries.Add(new Entry(file, strength));
    }

    /// <summary>Opens a LoRA safetensors file and adds it to the stack in one call. The stack takes ownership of the loaded file and disposes it when the stack itself is disposed.</summary>
    public void AddFromPath(string filePath, float strength = 1.0f)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        LoraFile file = LoraFile.Load(filePath);
        _ownedFiles.Add(file);
        _entries.Add(new Entry(file, strength));
    }

    /// <summary>Convenience entry point that applies the stack to multiple component weight dictionaries in one call. Pass null for components the model does not have (e.g., a Flux pipeline has no clipG). Returns the total number of weights modified across all components.</summary>
    public int ApplyToWeights(
        IBackend backend,
        IDictionary<string, Tensor>? unetWeights = null,
        IDictionary<string, Tensor>? transformerWeights = null,
        IDictionary<string, Tensor>? clipLWeights = null,
        IDictionary<string, Tensor>? clipGWeights = null)
    {
        int total = 0;
        if (unetWeights is not null) total += ApplyTo(unetWeights, LoraTarget.UNet, backend);
        if (transformerWeights is not null) total += ApplyTo(transformerWeights, LoraTarget.Transformer, backend);
        if (clipLWeights is not null) total += ApplyTo(clipLWeights, LoraTarget.ClipL, backend);
        if (clipGWeights is not null) total += ApplyTo(clipGWeights, LoraTarget.ClipG, backend);
        return total;
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
            foreach (LoraLayer layer in entry.File.Layers)
            {
                if (layer.Target != target) continue;
                if (!grouped.TryGetValue(layer.TargetKey, out List<(LoraLayer, float)>? list))
                {
                    list = [];
                    grouped[layer.TargetKey] = list;
                }
                list.Add((layer, entry.Strength));
            }
        }

        int merged = 0, skippedShape = 0;
        Dictionary<string, List<(int SliceIndex, int SliceCount, LoraLayer Layer, float Strength)>> fusedPending = [];
        foreach ((string canonicalKey, List<(LoraLayer layer, float strength)> deltas) in grouped)
        {
            if (!weights.TryGetValue(canonicalKey, out Tensor? baseW))
            {
                // Fused-QKV fallback: fp8 builds of Flux-lineage checkpoints keep attention fused
                // (attn.qkv / attn.add_qkv) while LoRA canonical keys are the split names. Without this, every
                // attention delta silently missed on those builds and the LoRA came out visibly weakened.
                if (TryResolveFusedSlice(canonicalKey, weights, out string fusedKey, out int sliceIndex, out int sliceCount))
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
                    + $"[{badLayer.LoraUp.Shape[0]}, {badLayer.LoraDown.Shape[1]}] but the checkpoint's weight is "
                    + $"{baseW.Shape} — this LoRA was trained against a different build of this architecture; "
                    + "skipping this weight.");
                skippedShape++;
                continue;
            }

            if (baseW.DType.IsQuantized && !baseW.DType.IsFp8)
            {
                // GGUF K-quant blocks: dequantizing is possible but requantizing a merged result back into
                // block form is not implemented, so the merge would silently change the model's size/precision.
                // Refuse with the fix named rather than surfacing the internal dequantizer error.
                throw new NotSupportedException(
                    $"LoRA weights can't be merged into a GGUF-quantized checkpoint (weight '{canonicalKey}' is "
                    + $"{baseW.DType.Name}). Use a safetensors build of this model (fp16/bf16/fp8) with LoRAs, or "
                    + "remove the LoRA.");
            }

            // CastTo folds Fp8ScaleFactor into the values and returns factor 1.0, so the merge below stays quant-unaware.
            Tensor accumF32 = baseW.CastTo(DType.F32); // owned copy, will be mutated in place
            try
            {
                foreach ((LoraLayer layer, float strength) in deltas)
                {
                    AccumulateDelta(backend, accumF32, layer, strength);
                }

                Tensor finalTensor;
                if (originalDtype.IsFp8)
                {
                    finalTensor = CheckpointConvertUtils.QuantizeF32ToFp8Scaled(accumF32, canonicalKey);
                    // A weight-side LoRA must not change activation scaling: the input scale is carried, not recomputed.
                    finalTensor.Fp8InputScaleFactor = baseW.Fp8InputScaleFactor;
                }
                else
                {
                    finalTensor = originalDtype == DType.F32 ? accumF32 : accumF32.CastTo(originalDtype);
                }
                if (!ReferenceEquals(finalTensor, accumF32))
                {
                    accumF32.Dispose();
                }
                _ownedMerged.Add(finalTensor);
                weights[canonicalKey] = finalTensor;
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
            DType fusedDtype = fusedBase.DType;
            long sliceRows = fusedBase.Shape[0] / slices[0].SliceCount;
            bool anyBad = slices.Exists(sl =>
                fusedBase.Shape.Rank != 2
                || fusedBase.Shape[0] % sl.SliceCount != 0
                || sl.Layer.LoraUp.Shape[0] != sliceRows
                || sl.Layer.LoraDown.Shape[1] != fusedBase.Shape[1]);
            if (anyBad)
            {
                Logs.Warning($"LoRA fused-slice merge into '{fusedKey}' skipped: delta shapes do not tile the fused weight {fusedBase.Shape}.");
                skippedShape++;
                continue;
            }
            Tensor accumF32 = fusedBase.CastTo(DType.F32);
            try
            {
                foreach ((int sliceIndex, int _, LoraLayer layer, float strength) in slices)
                {
                    AccumulateDeltaIntoRows(backend, accumF32, layer, strength, sliceIndex * sliceRows);
                }
                Tensor finalTensor;
                if (fusedDtype.IsFp8)
                {
                    finalTensor = CheckpointConvertUtils.QuantizeF32ToFp8Scaled(accumF32, fusedKey);
                    finalTensor.Fp8InputScaleFactor = fusedBase.Fp8InputScaleFactor;
                }
                else
                {
                    finalTensor = fusedDtype == DType.F32 ? accumF32 : accumF32.CastTo(fusedDtype);
                }
                if (!ReferenceEquals(finalTensor, accumF32))
                {
                    accumF32.Dispose();
                }
                _ownedMerged.Add(finalTensor);
                weights[fusedKey] = finalTensor;
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
            foreach (LoraFullWeightDiff diff in entry.File.FullWeightDiffs)
            {
                if (diff.Target != target) continue;
                if (!diffGrouped.TryGetValue(diff.TargetKey, out List<(Tensor, float)>? list))
                {
                    list = [];
                    diffGrouped[diff.TargetKey] = list;
                }
                list.Add((diff.Diff, entry.Strength));
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
            if (baseW.DType.IsQuantized && !baseW.DType.IsFp8)
            {
                throw new NotSupportedException(
                    $"LoRA weights can't be merged into a GGUF-quantized checkpoint (weight '{canonicalKey}' is "
                    + $"{baseW.DType.Name}). Use a safetensors build of this model (fp16/bf16/fp8) with LoRAs, or "
                    + "remove the LoRA.");
            }

            Tensor accumF32 = baseW.CastTo(DType.F32);
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

                Tensor finalTensor;
                if (originalDtype.IsFp8)
                {
                    finalTensor = CheckpointConvertUtils.QuantizeF32ToFp8Scaled(accumF32, canonicalKey);
                    finalTensor.Fp8InputScaleFactor = baseW.Fp8InputScaleFactor;
                }
                else
                {
                    finalTensor = originalDtype == DType.F32 ? accumF32 : accumF32.CastTo(originalDtype);
                }
                if (!ReferenceEquals(finalTensor, accumF32))
                {
                    accumF32.Dispose();
                }
                _ownedMerged.Add(finalTensor);
                weights[canonicalKey] = finalTensor;
                merged++;
            }
            catch
            {
                accumF32.Dispose();
                throw;
            }
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

    /// <summary>Whether this layer's <c>up @ down</c> product is the shape of the weight it would be added to.</summary>
    private static bool DeltaShapeMatches(LoraLayer layer, Tensor baseW) =>
        baseW.Shape.Rank == 2
        && layer.LoraUp.Shape[0] == baseW.Shape[0]
        && layer.LoraDown.Shape[1] == baseW.Shape[1];

    private static void AccumulateDelta(IBackend backend, Tensor accumF32, LoraLayer layer, float strength)
    {
        Tensor upF32 = layer.LoraUp.DType == DType.F32 ? layer.LoraUp.CastTo(DType.F32) : layer.LoraUp.CastTo(DType.F32);
        Tensor downF32 = layer.LoraDown.DType == DType.F32 ? layer.LoraDown.CastTo(DType.F32) : layer.LoraDown.CastTo(DType.F32);
        Tensor delta = new Tensor(new TensorShape(upF32.Shape[0], downF32.Shape[1]), DType.F32);
        try
        {
            backend.MatMul(delta, upF32, downF32);
            float scale = strength * (layer.Alpha / layer.Rank);
            backend.Scale(delta, delta, scale);
            backend.Add(accumF32, accumF32, delta);
        }
        finally
        {
            delta.Dispose();
            upF32.Dispose();
            downF32.Dispose();
        }
    }

    /// <summary>Maps a split attention canonical key onto its FUSED sibling when the dictionary carries the fused form: <c>…attn.to_q/k/v.weight → …attn.qkv.weight</c> rows [q;k;v], and <c>…attn.add_{q,k,v}_proj.weight → …attn.add_qkv.weight</c> — the layout ChromaCheckpointConverter (and the other Flux-lineage converters) keep for fp8 builds.
    /// <para>The <c>attention.qkv</c> and bare <c>qkv</c> spellings cover two families that keep attention fused in their ordinary (not just fp8) layout: Ideogram 4 reads <c>layers.{i}.attention.qkv.weight</c> and F-Lite reads <c>blocks.{i}.qkv.weight</c>. A LoRA trained directly against either names the fused module and matches without this path; one converted to split form would otherwise miss EVERY attention delta and still merge its non- attention weights, so the stack would report a successful merge and the image would come out subtly under-LoRA'd — the Chroma failure mode this whole fallback exists for, with no error to point at it.</para>
    /// <para>A miss here is safe by construction: the fallback only fires when the split key is absent from the dict AND the fused candidate is present, so it cannot capture a key the direct lookup would have served.</para></summary>
    private static bool TryResolveFusedSlice(string canonicalKey, IDictionary<string, Tensor> weights,
        out string fusedKey, out int sliceIndex, out int sliceCount)
    {
        fusedKey = string.Empty;
        sliceCount = 3;
        (string Suffix, string FusedSuffix, int Index)[] table =
        [
            (".attn.to_q.weight", ".attn.qkv.weight", 0),
            (".attn.to_k.weight", ".attn.qkv.weight", 1),
            (".attn.to_v.weight", ".attn.qkv.weight", 2),
            (".attn.add_q_proj.weight", ".attn.add_qkv.weight", 0),
            (".attn.add_k_proj.weight", ".attn.add_qkv.weight", 1),
            (".attn.add_v_proj.weight", ".attn.add_qkv.weight", 2),
            // Ideogram 4: layers.{i}.attention.qkv.weight
            (".attention.to_q.weight", ".attention.qkv.weight", 0),
            (".attention.to_k.weight", ".attention.qkv.weight", 1),
            (".attention.to_v.weight", ".attention.qkv.weight", 2),
            // F-Lite: blocks.{i}.qkv.weight — no attention segment at all.
            (".to_q.weight", ".qkv.weight", 0),
            (".to_k.weight", ".qkv.weight", 1),
            (".to_v.weight", ".qkv.weight", 2),
        ];
        foreach ((string suffix, string fusedSuffix, int index) in table)
        {
            if (canonicalKey.EndsWith(suffix, StringComparison.Ordinal))
            {
                string candidate = canonicalKey[..^suffix.Length] + fusedSuffix;
                if (weights.ContainsKey(candidate))
                {
                    fusedKey = candidate;
                    sliceIndex = index;
                    return true;
                }
            }
        }
        sliceIndex = 0;
        return false;
    }

    /// <summary>Like <see cref="AccumulateDelta"/> but adds the <c>up @ down</c> delta into the row window [<paramref name="rowOffset"/>, rowOffset + up.rows) of <paramref name="accumF32"/> — the fused-QKV slice merge. Host math: F32 tensors are host-resident at this point and the add is a one-time load-path cost.</summary>
    private static unsafe void AccumulateDeltaIntoRows(IBackend backend, Tensor accumF32, LoraLayer layer, float strength, long rowOffset)
    {
        Tensor upF32 = layer.LoraUp.CastTo(DType.F32);
        Tensor downF32 = layer.LoraDown.CastTo(DType.F32);
        Tensor delta = new Tensor(new TensorShape(upF32.Shape[0], downF32.Shape[1]), DType.F32);
        try
        {
            backend.MatMul(delta, upF32, downF32);
            float scale = strength * (layer.Alpha / layer.Rank);
            long rows = delta.Shape[0];
            long cols = delta.Shape[1];
            float* dp = (float*)delta.DataPointer;
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
            delta.Dispose();
            upF32.Dispose();
            downF32.Dispose();
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

    private readonly record struct Entry(LoraFile File, float Strength);
}
