using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Engine.Features;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Reference recipe: SDXL (dual-CLIP). The architecture every other family is measured against, and the worked example the lifted per-family recipes follow. Loads and converts the LDM checkpoint itself (rather than through <see cref="PipelineFactory.LoadSdxl"/>) so it can merge the request's LoRA stack into the UNet/CLIP weights, build the VAE <em>encoder</em> that img2img and inpaint need, and hand the text encoders to <see cref="SdxlRecipePipeline"/> for weighted-prompt conditioning.</summary>
public sealed class SdxlRecipe : IArchitectureRecipe
{
    /// <inheritdoc/>
    public string Name => "sdxl";

    /// <inheritdoc/>
    public ImageFeatures Supports =>
        ImageFeatures.Lora | ImageFeatures.ControlNet | ImageFeatures.IpAdapter | ImageFeatures.Refiner
        | ImageFeatures.Img2Img | ImageFeatures.Inpaint | ImageFeatures.VariationSeed | ImageFeatures.SeamlessTiling
        | ImageFeatures.PromptScheduling;

    /// <inheritdoc/>
    /// <remarks>Ledger evidence in <c>PromptWeightingModeLedgerTests</c>; applied by
    /// <c>WeightedConditioning</c>, which blends each token's CLIP output against the empty-prompt baseline.</remarks>
    public Diffusion.Prompting.PromptWeightingMode PromptWeighting => Diffusion.Prompting.PromptWeightingMode.ComfyBlend;

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "sdxl", StringComparison.OrdinalIgnoreCase);

    /// <summary>SDXL's official sampling settings: 50 steps at CFG 5.0, 1024x1024 (diffusers <c>StableDiffusionXLPipeline.__call__</c>).</summary>
    public static ImageDefaults FamilyDefaults { get; } = new ImageDefaults { Steps = 50, CfgScale = 5.0f, Width = 1024, Height = 1024 };

    /// <inheritdoc/>
    public ImageDefaults Defaults => FamilyDefaults;

    /// <inheritdoc/>
    /// <inheritdoc/>
    public MemoryCapabilities MemorySupports => MemoryCapabilities.CfgParallel | MemoryCapabilities.ComponentPlacement;

    private static string ResolveComponent(ModelAsset asset, string component, CancellationToken cancel)
    {
        string target = ModelDownloader.TargetPath(asset);
        try
        {
            return ModelDownloader.EnsureSideModelAsync(asset, onProgress: null, cancel).GetAwaiter().GetResult();
        }
        catch (Exception ex)
        {
            Logs.Error($"[SdxlRecipe] {component} could not be resolved as a separate file.", ex);
            throw new InvalidOperationException(
                $"SDXL: this checkpoint does not bundle {component}, and it could not be resolved as a separate file. "
                + $"Looked for '{target}'; the download from HuggingFace '{asset.Repo}/{asset.RepoPath}' failed: {ex.Message}", ex);
        }
    }

    /// <summary>Guards against a file that opens but carries none of the expected tensors.</summary>
    private static Dictionary<string, Tensor> RequireWeights(Dictionary<string, Tensor> weights, string component, string path)
    {
        if (weights.Count == 0)
        {
            throw new InvalidOperationException($"SDXL: the {component} file '{path}' contains no usable {component} tensors.");
        }
        return weights;
    }

    public IRecipePipeline Construct(RecipeContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        // One container for either format, and it folds fp8/int8 companions before the converter sees them —
        // an SDXL fp8_scaled or GGUF UNet was previously invisible to this recipe.
        using HartsyInference.ModelAssets.Checkpoints.CheckpointSource loader =
            HartsyInference.ModelAssets.Checkpoints.CheckpointSource.Open(context.CheckpointPath);
        SdxlCheckpointConverter.ConvertedWeights converted = SdxlCheckpointConverter.Convert(
            new Dictionary<string, HartsyInference.Core.Tensors.Tensor>(loader.Weights, StringComparer.Ordinal));
        MergedLoraStack? loraStack = null;
        try
        {
            Dictionary<string, Tensor> unetWeights = WeightStaging.ToOwnedF32(converted.UNet);
            Dictionary<string, Tensor> clipLWeights = WeightStaging.ToOwnedF32(converted.ClipL);
            Dictionary<string, Tensor> clipGWeights = WeightStaging.ToOwnedF32(converted.ClipG);
            Dictionary<string, Tensor> vaeWeights = WeightStaging.ToOwnedF32(converted.Vae);
            if (vaeWeights.Count == 0)
            {
                // A checkpoint that omits its VAE takes the canonical SDXL VAE from its own file, as the SD3 recipe does.
                string path = ResolveComponent(SideModels.SdxlVae, "VAE", context.Cancel);
                (Dictionary<string, Tensor> staged, SafeTensorsLoader vaeLoader) = LoaderVaeUtils.LoadFluxVaeF32(path);
                using (vaeLoader)
                {
                    // The copy is taken before the file is released, as the checkpoint's own weights are.
                    vaeWeights = RequireWeights(WeightStaging.ToOwnedF32(staged), "VAE", path);
                }
                Logs.Info($"[SdxlRecipe] VAE resolved as a separate file: {path}.");
            }

            loraStack = RecipeLoraMerge.Apply(
                context,
                new LoraMergeTargets { Unet = unetWeights, ClipL = clipLWeights, ClipG = clipGWeights },
                "SdxlRecipe");

            ClipTextEncoder clipL = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipL);
            clipL.LoadWeights(clipLWeights, "text_model");

            ClipTextEncoder clipG = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipG);
            clipG.LoadWeights(clipGWeights, "text_model");

            UNet unet = new UNet(UNetConfig.SdxlBase);
            unet.LoadWeights(unetWeights);

            VaeDecoder vaeDecoder = new VaeDecoder(VaeConfig.Sdxl);
            vaeDecoder.LoadWeights(vaeWeights);

            VaeEncoder vaeEncoder = LoaderVaeUtils.BuildEncoder(VaeConfig.Sdxl, vaeWeights, "SdxlRecipe");

            SdxlPipeline pipeline = new SdxlPipeline(context.Backend, clipL, clipG, unet, vaeDecoder, vaeEncoder)
            {
                TextEncoderBackend = context.TextEncoderBackendOrDefault,
                VaeBackend = context.VaeBackendOrDefault,
                CfgParallelBackend = context.CfgParallelBackend,
            };
            Logs.Info("[SdxlRecipe] SDXL ready.");
            return new SdxlRecipePipeline(pipeline, context.Backend, clipL, clipG, loraStack);
        }
        catch (Exception ex)
        {
            Logs.Error("[SdxlRecipe] Construction failed.", ex);
            loraStack?.Dispose();
            throw;
        }
        finally
        {
            // Safe: ToOwnedF32 copied every weight out of the mmap.
        }
    }
}
