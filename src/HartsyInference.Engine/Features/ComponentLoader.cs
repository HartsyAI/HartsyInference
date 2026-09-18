using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Checkpoints;

namespace HartsyInference.Engine.Features;

/// <summary>Loads one standalone model component the way every mix-and-match image recipe needs it: opened through <see cref="CheckpointSource"/> so GGUF and safetensors are indistinguishable and the quantization companions are folded first, <c>scaled_fp8</c> marker tensors dropped, then the keys routed through the recipe's transform.</summary>
/// <remarks>The order is the point. Folding happens inside the container, before <paramref name="keyTransform"/> renames
/// anything, because a transform that renames <c>.weight</c> has no rule for <c>.weight_scale</c> — folding after it
/// pairs nothing, drops the scale, and the component runs at <c>1/scale</c> with no symptom until the image does.</remarks>
internal static class ComponentLoader
{
    /// <summary>Loads <paramref name="filePath"/> and returns its weights plus the source that owns their memory — the caller must keep the source alive for as long as the weights are used. A null <paramref name="keyTransform"/> keeps keys as-is; a null transform result drops that key. On failure the source is disposed and the error logged under <paramref name="logTag"/>.</summary>
    internal static (Dictionary<string, Tensor> Weights, CheckpointSource Source) Load(
        string filePath,
        string logTag,
        Func<string, string?>? keyTransform,
        bool applyFp8Dequant,
        bool nvfp4ToFp8 = false)
    {
        CheckpointSource source = CheckpointSource.Open(filePath,
            new CheckpointOpenOptions { FoldQuantCompanions = applyFp8Dequant, Nvfp4ToFp8 = nvfp4ToFp8 });
        try
        {
            Dictionary<string, Tensor> merged = new Dictionary<string, Tensor>(source.Weights.Count);
            foreach (KeyValuePair<string, Tensor> kv in source.Weights)
            {
                if (kv.Key.EndsWith(".scaled_fp8", StringComparison.Ordinal) || kv.Key == "scaled_fp8")
                {
                    continue;
                }
                string? mapped = keyTransform is null ? kv.Key : keyTransform(kv.Key);
                if (mapped is not null)
                {
                    merged[mapped] = kv.Value;
                }
            }
            return (merged, source);
        }
        catch (Exception ex)
        {
            Logs.Error($"[{logTag}] Failed to load component '{Path.GetFileName(filePath)}'.", ex);
            source.Dispose();
            throw;
        }
    }

    /// <summary>Registers the source in <paramref name="sources"/> for the caller's bulk disposal instead of handing it back; only a successful load is registered, so a throw disposes the source here.</summary>
    internal static Dictionary<string, Tensor> Load(
        string filePath,
        string logTag,
        Func<string, string?>? keyTransform,
        bool applyFp8Dequant,
        List<IDisposable> sources,
        bool nvfp4ToFp8 = false)
    {
        ArgumentNullException.ThrowIfNull(sources);
        (Dictionary<string, Tensor> weights, CheckpointSource source) = Load(filePath, logTag, keyTransform, applyFp8Dequant, nvfp4ToFp8);
        sources.Add(source);
        return weights;
    }
}
