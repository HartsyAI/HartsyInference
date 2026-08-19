using HartsyInference.Core.Logging;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Engine.Features;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Tokenizers;
using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Stable Diffusion 1.5 recipe: a single all-in-one LDM/CompVis <c>.safetensors</c> (the format every civitai SD1.5 finetune ships) carries the UNet, the CLIP-L text encoder, and the VAE. Lifted from the SwarmUI backend's <c>Sd15Loader</c>; nothing is resolved as a side model. Any requested LoRA stack is merged into the UNet/CLIP-L weights here (the pipeline is cached per stack), and the constructed components are driven through <see cref="Sd15RecipePipeline"/>.</summary>
public sealed class Sd15Recipe : IArchitectureRecipe
{
    /// <inheritdoc/>
    public string Name => "sd15";

    /// <inheritdoc/>
    public ImageFeatures Supports =>
        ImageFeatures.Lora | ImageFeatures.ControlNet | ImageFeatures.IpAdapter
        | ImageFeatures.Img2Img | ImageFeatures.Inpaint | ImageFeatures.VariationSeed | ImageFeatures.SeamlessTiling | ImageFeatures.Refiner;

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "sd15", StringComparison.OrdinalIgnoreCase);

    /// <summary>SD 1.5's official sampling settings: 50 steps at CFG 7.5 and a native 512x512 (diffusers <c>StableDiffusionPipeline.__call__</c>, mirrored by <c>GenerationDefaults.Sd15</c>).</summary>
    public static ImageDefaults FamilyDefaults { get; } = new ImageDefaults { Steps = 50, CfgScale = 7.5f, Width = 512, Height = 512 };

    /// <inheritdoc/>
    public ImageDefaults Defaults => FamilyDefaults;

    /// <inheritdoc/>
    public IRecipePipeline Construct(RecipeContext context)
    {
        (Sd15CheckpointConverter.ConvertedWeights converted, HartsyInference.ModelAssets.SafeTensors.SafeTensorsLoader loader) = Sd15CheckpointConverter.LoadAndConvert(context.CheckpointPath);
        if (converted.UNet.Count == 0 || converted.ClipL.Count == 0 || converted.Vae.Count == 0)
        {
            loader.Dispose();
            throw new InvalidOperationException("SD1.5 checkpoint is missing UNet/CLIP-L/VAE components. Is this a complete SD1.5 file?");
        }
        Logs.Info($"[Sd15Recipe] UNet keys={converted.UNet.Count}, CLIP-L keys={converted.ClipL.Count}, VAE keys={converted.Vae.Count}.");

        MergedLoraStack? loraStack = null;
        try
        {
            loraStack = LoraApplier.BuildAndApply(
                LoraResolver.Resolve(context.Loras),
                context.Backend,
                unetWeights: converted.UNet,
                clipLWeights: converted.ClipL);
        }
        catch (Exception ex)
        {
            Logs.Error("[Sd15Recipe] LoRA merge failed.", ex);
            loader.Dispose();
            throw;
        }

        UNet unet = new UNet(UNetConfig.Sd15);
        unet.LoadWeights(converted.UNet);

        ClipTextEncoder textEncoder = new ClipTextEncoder(ClipTextEncoderConfig.Sd15);
        textEncoder.LoadWeights(converted.ClipL, prefix: "text_model");

        VaeDecoder vaeDecoder = new VaeDecoder(VaeConfig.Sd15);
        vaeDecoder.LoadWeights(converted.Vae);

        VaeEncoder vaeEncoder = LoaderVaeUtils.BuildEncoder(VaeConfig.Sd15, converted.Vae, "Sd15Recipe");

        StableDiffusion15Pipeline pipeline = new StableDiffusion15Pipeline(context.Backend, textEncoder, unet, vaeDecoder, vaeEncoder);
        Logs.Info("[Sd15Recipe] SD1.5 ready.");
        return new Sd15RecipePipeline(pipeline, new ClipTokenizer(), loader, context.Backend, textEncoder, loraStack);
    }
}
