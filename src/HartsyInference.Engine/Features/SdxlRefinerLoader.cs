using HartsyInference.Core.Logging;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Engine.Features;

/// <summary>Loads the second-pass SDXL refiner UNet named by a resolved <see cref="RefinerResolver.RefinerSpec"/>. Only the UNet is taken — the refiner phase reuses the base pass's CLIP-G conditioning and VAE, which is what the mid-loop StepSwap (<c>RefinerSwapConfig</c>) requires.</summary>
public static class SdxlRefinerLoader
{
    /// <summary>Models-root-relative folders searched for the refiner checkpoint.</summary>
    public static readonly string[] Folders = ["Stable-Diffusion", "checkpoints", "diffusion_models", "unet"];

    /// <summary>Locates, loads, and builds the refiner UNet for <paramref name="modelNameOrPath"/>. Caller owns the entry.</summary>
    public static SdxlRefinerEntry Load(string modelNameOrPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelNameOrPath);
        string path = ModelFileLocator.Require(modelNameOrPath, "SDXL refiner model", Folders);
        // The refiner's UNet has a genuinely different block layout than base (4 levels vs 3, different
        // input/output block numbering and attention placement — see SdxlRefinerCheckpointConverter's own class
        // doc), so it needs its OWN key-conversion table, not the base SdxlCheckpointConverter. Using the base
        // converter here (as this method did until 2026-08-11 — a real, pre-existing bug caught while adding
        // Tier 3.1's PostApply hand-off, which was the first thing to ever exercise this loader against a real
        // official-refiner checkpoint; StepSwap had zero test coverage before this either) would silently produce
        // an empty or wrong-shaped UNet dict — LoadWeights would then fail on a shape mismatch, or worse, load
        // whatever base-shaped keys happen to coincide and run a corrupt-conditioning refiner pass.
        (SdxlRefinerCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) = SdxlRefinerCheckpointConverter.LoadAndConvert(path);
        try
        {
            if (converted.UNet.Count == 0)
            {
                throw new InvalidOperationException($"Refiner checkpoint '{modelNameOrPath}' contains no UNet weights.");
            }
            UNet unet = new UNet(UNetConfig.SdxlRefiner);
            try
            {
                unet.LoadWeights(converted.UNet);
            }
            catch (Exception ex)
            {
                // A base-architecture SDXL checkpoint (CrossAttentionDim=2048, 3-level) passed as the refiner
                // model converts to the WRONG diffusers keys for UNetConfig.SdxlRefiner's 4-level/1280-dim shape
                // and throws a shape-mismatch exception here that reads like an engine bug, not a request error.
                // Same-checkpoint hires-fix (no separate official-refiner model) is a real, common want — it's
                // just not this mechanism; it needs its own PostApply path that never loads a refiner UNet at
                // all (a second GenerateFromTokens img2img call on the BASE pipeline's own UNet). Not built in
                // this slice — see ROADMAP.md.
                throw new NotSupportedException(
                    $"Refiner model '{modelNameOrPath}' does not load as an official SDXL refiner checkpoint "
                    + "(stabilityai/stable-diffusion-xl-refiner-1.0's architecture: CLIP-G-only, aesthetic-score "
                    + "ADM). A base-architecture SDXL checkpoint used as the refiner model is not supported yet — "
                    + "same-checkpoint hires-fix needs a different, unbuilt code path. See ROADMAP.md.", ex);
            }
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
