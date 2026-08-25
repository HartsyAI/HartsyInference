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
        | ImageFeatures.Img2Img | ImageFeatures.Inpaint | ImageFeatures.VariationSeed | ImageFeatures.SeamlessTiling;

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "sdxl", StringComparison.OrdinalIgnoreCase);

    /// <summary>SDXL's official sampling settings: 50 steps at CFG 5.0, 1024x1024 (diffusers <c>StableDiffusionXLPipeline.__call__</c>, mirrored by <c>GenerationDefaults.Sdxl</c>).</summary>
    public static ImageDefaults FamilyDefaults { get; } = new ImageDefaults { Steps = 50, CfgScale = 5.0f, Width = 1024, Height = 1024 };

    /// <inheritdoc/>
    public ImageDefaults Defaults => FamilyDefaults;

    /// <inheritdoc/>
    /// <inheritdoc/>
    public MemoryCapabilities MemorySupports => MemoryCapabilities.CfgParallel | MemoryCapabilities.ComponentPlacement;

    public IRecipePipeline Construct(RecipeContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        (SdxlCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            SdxlCheckpointConverter.LoadAndConvert(context.CheckpointPath);
        MergedLoraStack? loraStack = null;
        try
        {
            Dictionary<string, Tensor> unetWeights = WeightStaging.ToOwnedF32(converted.UNet);
            Dictionary<string, Tensor> clipLWeights = WeightStaging.ToOwnedF32(converted.ClipL);
            Dictionary<string, Tensor> clipGWeights = WeightStaging.ToOwnedF32(converted.ClipG);
            Dictionary<string, Tensor> vaeWeights = WeightStaging.ToOwnedF32(converted.Vae);

            loraStack = LoraApplier.BuildAndApply(
                LoraResolver.Resolve(context.Loras),
                context.Backend,
                unetWeights: unetWeights,
                clipLWeights: clipLWeights,
                clipGWeights: clipGWeights);

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
            loader.Dispose();
        }
    }
}
