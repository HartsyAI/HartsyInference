using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Engine.Features;
namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Z-Image recipe (Tongyi Lab NextDiT): the checkpoint carries the transformer only; the Qwen3-4B text encoder and the Flux VAE are resolved as side models. Lifted from the SwarmUI backend's <c>ZImageLoader</c>; constructs the components and drives generation through <see cref="ZImageRecipePipeline"/>.</summary>
public sealed class ZImageRecipe : IArchitectureRecipe
{
    /// <inheritdoc/>
    public string Name => "zimage";

    /// <inheritdoc/>
    /// <remarks>Z-Image shares the Flux VAE; the encoder is constructed alongside the decoder and ZImagePipeline implements the masked path.</remarks>
    public ImageFeatures Supports => ImageFeatures.Img2Img | ImageFeatures.Inpaint;

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "zimage", StringComparison.OrdinalIgnoreCase);

    /// <summary>Z-Image Turbo's official sampling settings: 8 steps, guidance-free (CFG 1.0), 1024x1024 (<c>GenerationDefaults.ZImageTurbo</c>).</summary>
    public static ImageDefaults FamilyDefaults { get; } = new ImageDefaults { Steps = 8, CfgScale = 1.0f, Width = 1024, Height = 1024 };

    /// <inheritdoc/>
    public ImageDefaults Defaults => FamilyDefaults;

    /// <inheritdoc/>
    public IRecipePipeline Construct(RecipeContext context)
    {
        // Resolve the two side models synchronously (Construct is a sync seam): Qwen3-4B text encoder + Flux VAE.
        // TODO(E-IMG-4): honor a user-picked Qwen/VAE override from ImageRequest.Components/Extra instead of always
        // taking the canonical SideModels entry (the SwarmUI loader read input.Get(T2IParamTypes.QwenModel/VAE)).
        string qwenPath = ModelDownloader.EnsureSideModelAsync(SideModels.Qwen3_4B, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
        string vaePath = ModelDownloader.EnsureSideModelAsync(SideModels.FluxAe, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();

        // 1. Load + convert the Z-Image transformer (checkpoint carries only these weights).
        (ZImageCheckpointConverter.ConvertedWeights zConv, SafeTensorsLoader zLoader) = ZImageCheckpointConverter.LoadAndConvert(context.CheckpointPath);
        if (zConv.Transformer.Count == 0)
        {
            zLoader.Dispose();
            throw new InvalidOperationException("Z-Image checkpoint has no transformer weights.");
        }
        ZImageConfig zConfig = ZImageConfig.FromWeights(zConv.Transformer);
        Logs.Info($"[ZImageRecipe] Building transformer (SchedulerShift={zConfig.SchedulerShift}).");
        ZImageTransformer transformer = new ZImageTransformer(zConfig);
        transformer.LoadWeights(zConv.Transformer);

        // 2. Load the Qwen3-4B encoder + its embedded tokenizer.
        SafeTensorsLoader qwenLoader = new SafeTensorsLoader();
        qwenLoader.Load(qwenPath);
        IReadOnlyDictionary<string, Tensor> qwenWeights = qwenLoader.GetAllTensors();
        if (qwenWeights.Count == 0)
        {
            qwenLoader.Dispose();
            zLoader.Dispose();
            throw new InvalidOperationException($"Qwen3 model file '{qwenPath}' has no tensors.");
        }
        LlamaStyleEncoder qwen = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Qwen3_4B);
        qwen.LoadWeights(qwenWeights);
        Qwen3Tokenizer tokenizer = new Qwen3Tokenizer(maxLength: 256);

        // 3. Load the Flux VAE (Z-Image reuses it verbatim). BF16 on Ampere+ (F32-equivalent range, halves the
        //    full-res decode workspace), F32 otherwise — the SDXL-VAE precision policy.
        (Dictionary<string, Tensor> vaeWeights, SafeTensorsLoader vaeLoader) = LoaderVaeUtils.LoadFluxVaeF32(vaePath);
        if (vaeWeights.Count == 0)
        {
            vaeLoader.Dispose();
            qwenLoader.Dispose();
            zLoader.Dispose();
            throw new InvalidOperationException($"VAE file '{vaePath}' has no usable VAE tensors.");
        }
        vaeWeights = VaePrecisionHelper.CastVaeWeights(vaeWeights, VaePrecisionHelper.PreferredVaeDtype(context.Backend));
        VaeDecoder vae = new VaeDecoder(VaeConfig.ZImage);
        vae.LoadWeights(vaeWeights);
        VaeEncoder vaeEncoder = LoaderVaeUtils.BuildEncoder(VaeConfig.ZImage, vaeWeights, "ZImageRecipe");

        ZImagePipeline pipeline = new ZImagePipeline(context.Backend, transformer, vae, vaeEncoder, zConfig);
        Logs.Info("[ZImageRecipe] Z-Image ready.");
        return new ZImageRecipePipeline(pipeline, qwen, tokenizer, context.Backend, zLoader, qwenLoader, vaeLoader);
    }
}
