using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Engine.Features;

/// <summary>Loads one standalone side-model safetensors component the way every mix-and-match image recipe needs it: <c>scaled_fp8</c> marker tensors dropped, keys routed through the recipe's transform, then the fp8 <c>*.scale_weight</c> companions folded in.</summary>
internal static class ComponentLoader
{
    /// <summary>Loads <paramref name="filePath"/> and returns its weights plus the loader that owns their memory — the caller must keep the loader alive for as long as the weights are used. A null <paramref name="keyTransform"/> keeps keys as-is; a null transform result drops that key. On failure the loader is disposed and the error logged under <paramref name="logTag"/>.</summary>
    internal static (Dictionary<string, Tensor> Weights, SafeTensorsLoader Loader) Load(
        string filePath,
        string logTag,
        Func<string, string?>? keyTransform,
        bool applyFp8Dequant,
        bool nvfp4ToFp8 = false)
    {
        SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(filePath);
        try
        {
            Dictionary<string, Tensor> merged = new Dictionary<string, Tensor>();
            foreach (KeyValuePair<string, Tensor> kv in loader.GetAllTensors())
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
            return (applyFp8Dequant ? CheckpointConvertUtils.ApplyFp8ScaledDequant(merged, nvfp4ToFp8) : merged, loader);
        }
        catch (Exception ex)
        {
            Logs.Error($"[{logTag}] Failed to load component '{Path.GetFileName(filePath)}'.", ex);
            loader.Dispose();
            throw;
        }
    }

    /// <summary>Registers the loader in <paramref name="loaders"/> for the caller's bulk disposal instead of handing it back; only a successful load is registered, so a throw disposes the loader here.</summary>
    internal static Dictionary<string, Tensor> Load(
        string filePath,
        string logTag,
        Func<string, string?>? keyTransform,
        bool applyFp8Dequant,
        List<SafeTensorsLoader> loaders,
        bool nvfp4ToFp8 = false)
    {
        ArgumentNullException.ThrowIfNull(loaders);
        (Dictionary<string, Tensor> weights, SafeTensorsLoader loader) = Load(filePath, logTag, keyTransform, applyFp8Dequant, nvfp4ToFp8);
        loaders.Add(loader);
        return weights;
    }
}
