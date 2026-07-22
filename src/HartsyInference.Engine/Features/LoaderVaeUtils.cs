using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Engine.Features;

/// <summary>Shared staging for the FLUX.1 16-channel autoencoder (<c>ae.safetensors</c>, original LDM naming): remaps
/// every key to the diffusers naming <c>VaeDecoder</c> expects and upcasts 16-bit weights to F32.</summary>
public static class LoaderVaeUtils
{
    /// <summary>Loads a FLUX.1-family VAE file and returns diffusers-keyed F32 weights; keys already in diffusers naming
    /// pass through (ConvertVaeKey is identity-tolerant) and unknown keys are dropped. The returned loader owns the
    /// tensor memory — keep it alive as long as the weights are used.</summary>
    public static (Dictionary<string, Tensor> Weights, SafeTensorsLoader Loader) LoadFluxVaeF32(string filePath)
    {
        ArgumentException.ThrowIfNullOrEmpty(filePath);
        SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(filePath);
        try
        {
            Dictionary<string, Tensor> result = new Dictionary<string, Tensor>();
            foreach (KeyValuePair<string, Tensor> kvp in loader.GetAllTensors())
            {
                string? diffusersKey = CheckpointConvertUtils.ConvertVaeKey(kvp.Key);
                if (diffusersKey is null)
                {
                    continue;
                }
                DType dt = kvp.Value.DType;
                result[diffusersKey] = (dt == DType.F16 || dt == DType.BF16) ? kvp.Value.CastTo(DType.F32) : kvp.Value;
            }
            return (result, loader);
        }
        catch (Exception ex)
        {
            HartsyInference.Core.Logging.Logs.Error($"[Features][VAE] Failed to stage Flux VAE weights from '{filePath}'.", ex);
            loader.Dispose();
            throw;
        }
    }
}
