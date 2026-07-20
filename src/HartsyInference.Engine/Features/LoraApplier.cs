using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;

namespace HartsyInference.Engine.Features;

/// <summary>Builds and applies a <see cref="MergedLoraStack"/> from resolved <see cref="LoraResolver.LoraSpec"/> entries
/// against per-architecture weight dictionaries. The dicts are mutated in place — matching keys get newly-allocated
/// merged tensors owned by the returned stack, so dispose the stack only <em>after</em> the components built from those
/// dicts are done with.</summary>
public static class LoraApplier
{
    /// <summary>Opens every LoRA file, partitions layers by target, and merges into the supplied weight dicts. Returns
    /// the stack (null for an empty spec list) so the caller can free the merged tensors when generation finishes.</summary>
    /// <param name="unetWeights">Pass when targeting an SD 1.5 / SDXL UNet, else null.</param>
    /// <param name="transformerWeights">Pass when targeting a DiT transformer, else null.</param>
    /// <param name="clipGWeights">Pass when the architecture has a CLIP-G encoder (SDXL only).</param>
    public static MergedLoraStack? BuildAndApply(
        IReadOnlyList<LoraResolver.LoraSpec>? loras,
        IBackend backend,
        IDictionary<string, Tensor>? unetWeights = null,
        IDictionary<string, Tensor>? transformerWeights = null,
        IDictionary<string, Tensor>? clipLWeights = null,
        IDictionary<string, Tensor>? clipGWeights = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (loras is null || loras.Count == 0)
        {
            return null;
        }
        MergedLoraStack stack = new MergedLoraStack();
        try
        {
            foreach (LoraResolver.LoraSpec spec in loras)
            {
                Logs.Verbose($"[Features][LoRA] Loading '{spec.ModelId ?? Path.GetFileName(spec.FilePath)}' (model={spec.ModelStrength}, tenc={spec.TencStrength}).");
                stack.AddFromPath(spec.FilePath, strength: spec.ModelStrength);
            }
            // A single combined "model" strength applies everywhere. Splitting model vs tenc needs a second stack whose
            // tenc strength is applied only to ClipL/ClipG — deferred until users commonly tune the two independently.
            int merged = stack.ApplyToWeights(
                backend,
                unetWeights: unetWeights,
                transformerWeights: transformerWeights,
                clipLWeights: clipLWeights,
                clipGWeights: clipGWeights);
            if (merged == 0)
            {
                Logs.Warning(
                    "[Features][LoRA] Stack matched 0 weights. The LoRA's target keys may not align with this architecture, "
                    + "or the format may not be supported. Generation will proceed without LoRA effect.");
            }
            else
            {
                Logs.Verbose($"[Features][LoRA] Stack merged {merged} weights across components.");
            }
            return stack;
        }
        catch (Exception ex)
        {
            Logs.Error("[Features][LoRA] Failed to build/apply the LoRA stack.", ex);
            stack.Dispose();
            throw;
        }
    }

    /// <summary>Shallow-copies a weight dict so LoRA replacement doesn't poison the cache's original. Tensor instances
    /// are referenced, not copied; the replacements written into the copy are owned by the stack from <see cref="BuildAndApply"/>.</summary>
    public static Dictionary<string, Tensor>? ShallowClone(IReadOnlyDictionary<string, Tensor>? source)
    {
        if (source is null)
        {
            return null;
        }
        Dictionary<string, Tensor> copy = new Dictionary<string, Tensor>(source.Count);
        foreach (KeyValuePair<string, Tensor> kv in source)
        {
            copy[kv.Key] = kv.Value;
        }
        return copy;
    }
}
