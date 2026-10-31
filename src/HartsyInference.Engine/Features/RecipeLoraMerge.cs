using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Recipes;
using HartsyInference.ModelAssets.Lora;
using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;

namespace HartsyInference.Engine.Features;

/// <summary>A LoRA a recipe attaches on its own gate rather than from the request: its file, the gate that switches it, and its strength.</summary>
/// <remarks>The layers the merge attaches own copies of the file's factors, so the caller may dispose the file once the merge returns.
/// The stack's entries still refer to the file, so the stack must not be applied again after the file is disposed.</remarks>
public sealed record GatedLora(LoraFile File, AdjunctGate Gate, float Strength = 1.0f);

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
    /// <param name="gated">LoRAs switched by their own gates, such as a reference adapter. They are applied with the request's LoRAs.</param>
    public static MergedLoraStack? Apply(RecipeContext context, LoraMergeTargets targets, string logTag,
        IReadOnlyList<GatedLora>? gated = null)
    {
        ArgumentNullException.ThrowIfNull(context);
        ArgumentNullException.ThrowIfNull(targets);
        return Apply(LoraResolver.Resolve(context.Loras), context.Backend, targets, logTag, gated);
    }

    /// <summary>Same as <see cref="Apply(RecipeContext, LoraMergeTargets, string, IReadOnlyList{GatedLora}?)"/> for a recipe that resolved (and possibly filtered) the specs itself — MiniMax-H3 splits acceleration adapters out during planning.</summary>
    public static MergedLoraStack? Apply(IReadOnlyList<LoraResolver.LoraSpec>? loras, IBackend backend,
        LoraMergeTargets targets, string logTag, IReadOnlyList<GatedLora>? gated = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(targets);
        bool hasGated = gated is { Count: > 0 };
        if ((loras is null || loras.Count == 0) && !hasGated)
        {
            return null;
        }
        MergedLoraStack stack = new MergedLoraStack();
        try
        {
            foreach (LoraResolver.LoraSpec spec in loras ?? [])
            {
                Logs.Verbose($"[{logTag}][LoRA] Loading '{spec.ModelId ?? Path.GetFileName(spec.FilePath)}' "
                    + $"(model={spec.ModelStrength}, tenc={spec.TencStrength}).");
                stack.AddFromPath(spec.FilePath, strength: spec.ModelStrength, tencStrength: spec.TencStrength);
            }
            foreach (GatedLora adapter in gated ?? [])
            {
                Logs.Verbose($"[{logTag}][LoRA] Attaching gated '{Path.GetFileName(adapter.File.FilePath)}' at strength {adapter.Strength}.");
                stack.AddGated(adapter.File, adapter.Gate, adapter.Strength);
            }
            int merged = stack.ApplyToWeights(
                backend,
                unetWeights: targets.Unet,
                transformerWeights: targets.Transformer,
                clipLWeights: targets.ClipL,
                clipGWeights: targets.ClipG,
                textEncoder2Weights: targets.TextEncoder2,
                transformerNativeNames: targets.TransformerNativeNames);
            if (merged == 0)
            {
                // A zero-match LoRA used to warn and proceed — a generation that succeeds and looks unaffected,
                // which is the worst failure mode there is: the user thinks the LoRA is weak, not broken. Refuse
                // with the files named instead; SwarmUI surfaces this message directly.
                IEnumerable<string> names = (loras ?? []).Select(l => $"'{l.ModelId ?? Path.GetFileName(l.FilePath)}'")
                    .Concat((gated ?? []).Select(g => $"'{Path.GetFileName(g.File.FilePath)}'"));
                throw new NotSupportedException(
                    $"LoRA {string.Join(", ", names)} matched 0 weights "
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
