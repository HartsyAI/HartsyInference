using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Vision.Upscale;

/// <summary>Normalizes Real-ESRGAN / ESRGAN checkpoints into the flat key layout
/// <see cref="RrdbNet.LoadWeights"/> expects. BasicSR checkpoints wrap the state dict under
/// <c>params_ema</c> or <c>params</c>; some exports prefix every key with <c>model.</c>. This strips
/// those wrappers so the remaining keys are <c>conv_first.weight</c>, <c>body.0.rdb1.conv1.weight</c>, etc.</summary>
public static class RealEsrganConverter
{
    private static readonly string[] StripPrefixes = ["params_ema.", "params.", "model."];

    /// <summary>Strips the known BasicSR wrapper prefixes from each key.</summary>
    public static Dictionary<string, Tensor> Convert(IReadOnlyDictionary<string, Tensor> weights)
    {
        Dictionary<string, Tensor> result = new(weights.Count);
        foreach (KeyValuePair<string, Tensor> kvp in weights)
        {
            string key = kvp.Key;
            foreach (string prefix in StripPrefixes)
            {
                if (key.StartsWith(prefix, StringComparison.Ordinal))
                {
                    key = key[prefix.Length..];
                    break;
                }
            }
            result[key] = kvp.Value;
        }
        return result;
    }

    /// <summary>Infers the RRDBNet config from a converted weight set: scale from <c>conv_first</c>'s input channels
    /// (12 means BasicSR's 2× pixel-unshuffle front end, i.e. an x2plus checkpoint; 3 means 4× — every Real-ESRGAN
    /// checkpoint carries both <c>conv_up</c> stages, so their presence says nothing about the factor), block count
    /// from the highest <c>body.{i}</c> index.</summary>
    public static RealEsrganConfig InferConfig(IReadOnlyDictionary<string, Tensor> weights)
    {
        int inputChannels = weights.TryGetValue("conv_first.weight", out Tensor? first) ? (int)first.Shape[1] : 3;
        int scale = inputChannels == 12 ? 2 : 4;

        int maxBlock = -1;
        foreach (string key in weights.Keys)
        {
            if (key.StartsWith("body.", StringComparison.Ordinal))
            {
                int dot = key.IndexOf('.', 5);
                if (dot > 5 && int.TryParse(key.AsSpan(5, dot - 5), out int idx))
                {
                    maxBlock = Math.Max(maxBlock, idx);
                }
            }
        }
        int numBlock = maxBlock + 1;

        int numFeat = weights.TryGetValue("conv_first.weight", out Tensor? cf) ? (int)cf.Shape[0] : 64;
        int numGrowCh = weights.TryGetValue("body.0.rdb1.conv1.weight", out Tensor? g) ? (int)g.Shape[0] : 32;

        return new RealEsrganConfig
        {
            NumFeat = numFeat,
            NumBlock = numBlock > 0 ? numBlock : 23,
            NumGrowCh = numGrowCh,
            Scale = scale,
        };
    }

    /// <summary>Loads a single-file safetensors Real-ESRGAN checkpoint, normalizes keys, and returns the
    /// converted weights plus the loader (keep it alive while the weights are in use).</summary>
    public static (Dictionary<string, Tensor> weights, SafeTensorsLoader loader) LoadAndConvert(string safetensorsPath)
    {
        SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(safetensorsPath);
        Dictionary<string, Tensor> converted = Convert(loader.GetAllTensors());
        return (converted, loader);
    }
}
