using HartsyInference.Core.Logging;
using HartsyInference.Diffusion.Adapters;
using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Engine.Features;

/// <summary>Loads a ControlNet checkpoint and constructs an adapter against a base UNet config, refusing base-model mismatches (an SD 1.5 ControlNet on an SDXL generation, or a Flux DiT ControlNet which needs <see cref="FluxControlNetResolver"/> instead).</summary>
public static class ControlNetWeightLoader
{
    /// <summary>Models-root-relative folders searched for ControlNet weights.</summary>
    public static readonly string[] Folders = ["controlnet", "ControlNet", "controlnets"];

    /// <summary>Resolves <paramref name="modelIdOrPath"/> to a file, loads it, and builds the adapter for <paramref name="baseConfig"/>.</summary>
    public static ControlNetCacheEntry Load(string modelIdOrPath, UNetConfig baseConfig)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelIdOrPath);
        ArgumentNullException.ThrowIfNull(baseConfig);
        string path = ModelFileLocator.Require(modelIdOrPath, "ControlNet model", Folders);
        ControlNetFile file = ControlNetLoader.Load(path);
        try
        {
            // SDXL + SD 1.5 are wired through their pipelines; Flux DiT ControlNets need the separate Flux adapter class.
            if (file.BaseModel is not (ControlNetBaseModel.Sdxl or ControlNetBaseModel.Sd15))
            {
                throw new InvalidOperationException(
                    $"ControlNet '{modelIdOrPath}' detected as base={file.BaseModel}. UNet-family ControlNets must be SDXL or SD 1.5; "
                    + "Flux DiT ControlNets go through FluxControlNetResolver.");
            }
            bool baseIsSdxl = baseConfig.CrossAttentionDim == 2048;
            if ((file.BaseModel == ControlNetBaseModel.Sdxl) != baseIsSdxl)
            {
                throw new InvalidOperationException(
                    $"ControlNet '{modelIdOrPath}' is a {file.BaseModel} ControlNet but the current generation uses a "
                    + $"{(baseIsSdxl ? "SDXL" : "SD 1.5")} base model. Pick a matching ControlNet.");
            }
            ControlNet adapter = new ControlNet(file.Config, baseConfig);
            adapter.LoadWeights(file.Weights);
            return new ControlNetCacheEntry
            {
                FilePath = path,
                File = file,
                Adapter = adapter,
            };
        }
        catch (Exception ex)
        {
            Logs.Error($"[Features][ControlNet] Failed to load '{path}'.", ex);
            file.Dispose();
            throw;
        }
    }
}
