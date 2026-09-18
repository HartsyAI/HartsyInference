using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Recipes;
using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;

namespace HartsyInference.Engine.Features;

/// <summary>The one place a recipe applies the request's LoRAs. Resolves the request stack, opens every file, and applies each layer to the component dictionary its target names — merging into dense/fp8/int8 weights and attaching a runtime adjunct to block-quantized ones.</summary>
/// <remarks><para><b>Call this BEFORE <c>LoadWeights</c>.</b> Device caches are identity-keyed, so a model that has
/// already bound the pre-merge tensors keeps serving them and the LoRA silently does nothing — the rule every recipe
/// used to restate for itself.</para>
/// <para>The returned stack owns every tensor it put in those dictionaries. Dispose it only once the components built
/// from them are done, and null means the request asked for no LoRAs.</para></remarks>
public static class RecipeLoraMerge
{
    /// <summary>Applies <paramref name="context"/>'s LoRA stack to <paramref name="targets"/>, or returns null when the request carries none.</summary>
    /// <param name="logTag">Recipe name for the log lines, e.g. <c>Flux1Recipe</c>.</param>
    public static MergedLoraStack? Apply(RecipeContext context, LoraMergeTargets targets, string logTag)
    {
        ArgumentNullException.ThrowIfNull(context);
        ArgumentNullException.ThrowIfNull(targets);
        return Apply(LoraResolver.Resolve(context.Loras), context.Backend, targets, logTag);
    }

    /// <summary>Same as <see cref="Apply(RecipeContext, LoraMergeTargets, string)"/> for a recipe that resolved (and possibly filtered) the specs itself — MiniMax-H3 splits acceleration adapters out during planning.</summary>
    public static MergedLoraStack? Apply(IReadOnlyList<LoraResolver.LoraSpec>? loras, IBackend backend,
        LoraMergeTargets targets, string logTag)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(targets);
        if (loras is null || loras.Count == 0)
        {
            return null;
        }
        MergedLoraStack stack = new MergedLoraStack();
        try
        {
            foreach (LoraResolver.LoraSpec spec in loras)
            {
                Logs.Verbose($"[{logTag}][LoRA] Loading '{spec.ModelId ?? Path.GetFileName(spec.FilePath)}' "
                    + $"(model={spec.ModelStrength}, tenc={spec.TencStrength}).");
                stack.AddFromPath(spec.FilePath, strength: spec.ModelStrength, tencStrength: spec.TencStrength);
            }
            int merged = stack.ApplyToWeights(
                backend,
                unetWeights: targets.Unet,
                transformerWeights: targets.Transformer,
                clipLWeights: targets.ClipL,
                clipGWeights: targets.ClipG,
                textEncoder2Weights: targets.TextEncoder2);
            if (merged == 0)
            {
                // A zero-match LoRA used to warn and proceed — a generation that succeeds and looks unaffected,
                // which is the worst failure mode there is: the user thinks the LoRA is weak, not broken. Refuse
                // with the files named instead; SwarmUI surfaces this message directly.
                throw new NotSupportedException(
                    $"LoRA {string.Join(", ", loras.Select(l => $"'{l.ModelId ?? Path.GetFileName(l.FilePath)}'"))} matched 0 weights "
                    + "on this model — its key format doesn't align with this architecture (wrong family's LoRA, or an "
                    + "unrecognized training-tool format). Remove it, or pick a LoRA trained for this model family.");
            }
            Logs.Verbose($"[{logTag}][LoRA] Stack applied {merged} weights across components.");
            return stack;
        }
        catch (Exception ex)
        {
            Logs.Error($"[{logTag}][LoRA] Failed to build/apply the LoRA stack.", ex);
            stack.Dispose();
            throw;
        }
    }

    /// <summary>Shallow-copies a weight dict so LoRA replacement doesn't poison the cache's original. Tensor instances are referenced, not copied; the replacements written into the copy are owned by the stack from <see cref="Apply(RecipeContext, LoraMergeTargets, string)"/>.</summary>
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
