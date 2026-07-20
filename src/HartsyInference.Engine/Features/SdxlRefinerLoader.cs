using HartsyInference.Core.Logging;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Engine.Features;

/// <summary>Loads the second-pass SDXL refiner UNet named by a resolved <see cref="RefinerResolver.RefinerSpec"/>.
/// Only the UNet is taken — the refiner phase reuses the base pass's CLIP-G conditioning and VAE, which is what the
/// mid-loop StepSwap (<c>RefinerSwapConfig</c>) requires.</summary>
public static class SdxlRefinerLoader
{
    /// <summary>Models-root-relative folders searched for the refiner checkpoint.</summary>
    public static readonly string[] Folders = ["Stable-Diffusion", "checkpoints", "diffusion_models", "unet"];

    /// <summary>Locates, loads, and builds the refiner UNet for <paramref name="modelNameOrPath"/>. Caller owns the entry.</summary>
    public static SdxlRefinerEntry Load(string modelNameOrPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelNameOrPath);
        string path = ModelFileLocator.Require(modelNameOrPath, "SDXL refiner model", Folders);
        (SdxlCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) = SdxlCheckpointConverter.LoadAndConvert(path);
        try
        {
            if (converted.UNet.Count == 0)
            {
                throw new InvalidOperationException($"Refiner checkpoint '{modelNameOrPath}' contains no UNet weights.");
            }
            UNet unet = new UNet(UNetConfig.SdxlRefiner);
            unet.LoadWeights(converted.UNet);
            Logs.Info($"[Features][Refiner] Loaded SDXL refiner UNet from '{path}' ({converted.UNet.Count} keys).");
            return new SdxlRefinerEntry { FilePath = path, Unet = unet, Loader = loader };
        }
        catch (Exception ex)
        {
            Logs.Error($"[Features][Refiner] Failed to load refiner '{modelNameOrPath}'.", ex);
            loader.Dispose();
            throw;
        }
    }
}
