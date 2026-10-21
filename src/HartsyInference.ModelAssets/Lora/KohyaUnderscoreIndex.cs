using HartsyInference.Core.Exceptions;

namespace HartsyInference.ModelAssets.Lora;

/// <summary>Resolves kohya <c>lora_unet_</c> module names against a model's weight names the way ComfyUI does: the weight <c>a.b.c.weight</c> is the module <c>a_b_c</c>. No per-family table, so any DiT a trainer wrote underscored names for resolves.</summary>
/// <remarks>Only <c>.weight</c> keys are indexed — those are what a LoRA adapts. Two weights collapsing to one underscored name (<c>a.b_c</c> and <c>a_b.c</c>) is not guessed at: the collision is recorded and the lookup that lands on it throws naming every colliding weight, while collisions no LoRA module touches stay harmless.</remarks>
public sealed class KohyaUnderscoreIndex
{
    private const string WeightSuffix = ".weight";

    private readonly Dictionary<string, List<(string Native, string Target)>> _byModule = new(StringComparer.Ordinal);

    private KohyaUnderscoreIndex()
    {
    }

    /// <summary>Indexes <paramref name="nativeToTarget"/>: each model weight's own name (what ComfyUI would underscore) paired with the key it holds in the weight dictionary the LoRA is applied to. The two differ when the loader renames checkpoint keys.</summary>
    public static KohyaUnderscoreIndex Build(IEnumerable<KeyValuePair<string, string>> nativeToTarget)
    {
        ArgumentNullException.ThrowIfNull(nativeToTarget);
        KohyaUnderscoreIndex index = new();
        foreach ((string native, string target) in nativeToTarget)
        {
            if (!native.EndsWith(WeightSuffix, StringComparison.Ordinal))
            {
                continue;
            }
            string module = native[..^WeightSuffix.Length].Replace('.', '_');
            if (!index._byModule.TryGetValue(module, out List<(string, string)>? weights))
            {
                weights = [];
                index._byModule[module] = weights;
            }
            weights.Add((native, target));
        }
        return index;
    }

    /// <summary>Indexes a weight dictionary whose keys are the model's own weight names.</summary>
    public static KohyaUnderscoreIndex Build(IEnumerable<string> weightKeys)
    {
        ArgumentNullException.ThrowIfNull(weightKeys);
        return Build(weightKeys.Select(key => KeyValuePair.Create(key, key)));
    }

    /// <summary>Resolves <paramref name="moduleName"/> (the <c>lora_unet_</c> body) to its dictionary key; false when no weight has that name. Throws when more than one does.</summary>
    public bool TryResolve(string moduleName, out string targetKey)
    {
        ArgumentNullException.ThrowIfNull(moduleName);
        if (!_byModule.TryGetValue(moduleName, out List<(string Native, string Target)>? weights))
        {
            targetKey = string.Empty;
            return false;
        }
        if (weights.Count > 1)
        {
            string names = string.Join(" and ", weights.Select(w => $"'{w.Native}'").Order(StringComparer.Ordinal));
            throw new HartsyInferenceException(
                $"LoRA module 'lora_unet_{moduleName}' is ambiguous on this model: weights {names} both map to "
                + $"'{moduleName}' once '.' is written as '_'. Refusing rather than pick one.");
        }
        targetKey = weights[0].Target;
        return true;
    }
}
