using HartsyInference.Core.Logging;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.Tokenizers;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Stable Diffusion 1.5 recipe: a single all-in-one LDM/CompVis <c>.safetensors</c> (the format every civitai SD1.5 finetune ships) carries the UNet, the CLIP-L text encoder, and the VAE. Lifted from the SwarmUI backend's <c>Sd15Loader</c>; nothing is resolved as a side model. Constructs the components and drives generation through <see cref="Sd15RecipePipeline"/>.</summary>
public sealed class Sd15Recipe : IArchitectureRecipe
{
    /// <inheritdoc/>
    public string Name => "sd15";

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "sd15", StringComparison.OrdinalIgnoreCase);

    /// <inheritdoc/>
    public IRecipePipeline Construct(RecipeContext context)
    {
        (Sd15CheckpointConverter.ConvertedWeights converted, HartsyInference.ModelHandler.SafeTensors.SafeTensorsLoader loader) = Sd15CheckpointConverter.LoadAndConvert(context.CheckpointPath);
        if (converted.UNet.Count == 0 || converted.ClipL.Count == 0 || converted.Vae.Count == 0)
        {
            loader.Dispose();
            throw new InvalidOperationException("SD1.5 checkpoint is missing UNet/CLIP-L/VAE components. Is this a complete SD1.5 file?");
        }
        Logs.Info($"[Sd15Recipe] UNet keys={converted.UNet.Count}, CLIP-L keys={converted.ClipL.Count}, VAE keys={converted.Vae.Count}.");

        UNet unet = new UNet(UNetConfig.Sd15);
        unet.LoadWeights(converted.UNet);

        ClipTextEncoder textEncoder = new ClipTextEncoder(ClipTextEncoderConfig.Sd15);
        textEncoder.LoadWeights(converted.ClipL, prefix: "text_model");

        VaeDecoder vaeDecoder = new VaeDecoder(VaeConfig.Sd15);
        vaeDecoder.LoadWeights(converted.Vae);

        VaeEncoder vaeEncoder = new VaeEncoder(VaeConfig.Sd15);
        vaeEncoder.LoadWeights(converted.Vae);

        StableDiffusion15Pipeline pipeline = new StableDiffusion15Pipeline(context.Backend, textEncoder, unet, vaeDecoder, vaeEncoder);
        Logs.Info("[Sd15Recipe] SD1.5 ready.");
        return new Sd15RecipePipeline(pipeline, new ClipTokenizer(), loader);
    }
}
