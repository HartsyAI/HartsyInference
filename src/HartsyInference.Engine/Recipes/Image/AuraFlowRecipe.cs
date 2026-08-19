using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Engine.Features;
namespace HartsyInference.Engine.Recipes.Image;

/// <summary>AuraFlow v0.2 / v0.3 recipe (fal/AuraFlow, MMDiT + single-DiT hybrid). The single-file checkpoint bundles the transformer, the Pile-T5-XL text encoder, and the SDXL-family VAE under one safetensors; nothing is resolved as a side model. Lifted from the SwarmUI backend's <c>AuraFlowLoader</c>; constructs the components and drives generation through <see cref="AuraFlowRecipePipeline"/>.</summary>
public sealed class AuraFlowRecipe : IArchitectureRecipe
{
    /// <inheritdoc/>
    public string Name => "auraflow";


    /// <inheritdoc/>
    /// <remarks>AuraFlow reuses the SDXL VAE; the encoder half is built alongside the decoder.</remarks>
    public ImageFeatures Supports => ImageFeatures.Img2Img | ImageFeatures.Inpaint | ImageFeatures.SeamlessTiling | ImageFeatures.VariationSeed;
    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "auraflow", StringComparison.OrdinalIgnoreCase);

    /// <summary>AuraFlow's official sampling settings: 50 steps at guidance 3.5, 1024x1024 (diffusers <c>AuraFlowPipeline.__call__</c>, mirrored by <c>GenerationDefaults.AuraFlow</c>).</summary>
    public static ImageDefaults FamilyDefaults { get; } = new ImageDefaults { Steps = 50, CfgScale = 3.5f, Width = 1024, Height = 1024 };

    /// <inheritdoc/>
    public ImageDefaults Defaults => FamilyDefaults;

    /// <inheritdoc/>
    public IRecipePipeline Construct(RecipeContext context)
    {
        (AuraFlowCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader mainLoader) = AuraFlowCheckpointConverter.LoadAndConvert(context.CheckpointPath);
        Logs.Info($"[AuraFlowRecipe] Converted: {converted.Transformer.Count} transformer / {converted.T5.Count} T5 / {converted.Vae.Count} VAE keys.");
        if (converted.Transformer.Count == 0 || converted.T5.Count == 0 || converted.Vae.Count == 0)
        {
            mainLoader.Dispose();
            throw new InvalidOperationException("AuraFlow checkpoint is missing one of transformer / T5 / VAE. AuraFlow expects a complete bundled file (the v0.3 fal-released format).");
        }

        AuraFlowConfig config = AuraFlowConfig.V03;
        Logs.Info($"[AuraFlowRecipe] Building transformer ({config.NumDoubleBlocks} double + {config.NumSingleBlocks} single, V03 preset).");
        AuraFlowTransformer transformer = new AuraFlowTransformer(config);
        transformer.LoadWeights(converted.Transformer);

        T5TextEncoder t5 = new T5TextEncoder(T5TextEncoderConfig.PileT5Xl);
        t5.LoadWeights(converted.T5);

        // AuraFlow reuses the SDXL VAE — same F16-overflow problem. BF16 on Ampere+ (F32-equivalent range),
        // F32 otherwise. Never F16, which overflows the SDXL VAE resnets. (Inlined VaePrecisionHelper policy.)
        DType vaeDtype = VaePrecisionHelper.PreferredVaeDtype(context.Backend);
        Dictionary<string, Tensor> vaeWeights = VaePrecisionHelper.CastVaeWeights(converted.Vae, vaeDtype);
        VaeDecoder vae = new VaeDecoder(VaeConfig.AuraFlow);
        vae.LoadWeights(vaeWeights);
        VaeEncoder? vaeEncoder = LoaderVaeUtils.TryBuildEncoder(VaeConfig.AuraFlow, vaeWeights, "AuraFlowRecipe");

        // Pile-T5-XL needs its OWN SentencePiece (same 32128 vocab size as Google T5 v1.1 but different
        // token-ID assignments) — the embedded Google-T5 spiece denoises into a coherent image but not the
        // prompted one, since every token id maps to the wrong piece.
        string spiecePath = ModelDownloader.EnsureSideModelAsync(SideModels.PileT5XlSpiece, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
        T5Tokenizer tokenizer = new T5Tokenizer(spiecePath, maxLength: 256);

        AuraFlowPipeline pipeline = new AuraFlowPipeline(context.Backend, t5, transformer, vae, vaeEncoder, config);
        Logs.Info("[AuraFlowRecipe] AuraFlow ready.");
        return new AuraFlowRecipePipeline(pipeline, tokenizer, mainLoader);
    }
}
