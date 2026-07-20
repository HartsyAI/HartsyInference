using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>AuraFlow v0.2 / v0.3 recipe (fal/AuraFlow, MMDiT + single-DiT hybrid). The single-file checkpoint bundles the transformer, the Pile-T5-XL text encoder, and the SDXL-family VAE under one safetensors; nothing is resolved as a side model. Lifted from the SwarmUI backend's <c>AuraFlowLoader</c>; constructs the components and drives generation through <see cref="AuraFlowRecipePipeline"/>.</summary>
public sealed class AuraFlowRecipe : IArchitectureRecipe
{
    /// <inheritdoc/>
    public string Name => "auraflow";

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "auraflow", StringComparison.OrdinalIgnoreCase);

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
        DType vaeDtype = context.Backend.Capabilities.SupportsBF16 ? DType.BF16 : DType.F32;
        Dictionary<string, Tensor> vaeWeights = CastWeights(converted.Vae, vaeDtype);
        VaeDecoder vae = new VaeDecoder(VaeConfig.AuraFlow);
        vae.LoadWeights(vaeWeights);

        // TODO(E-IMG-5): AuraFlow needs the Pile-T5-XL SentencePiece (same 32128 vocab as Google T5 v1.1 but
        // different token-ID assignments); the SwarmUI loader downloaded pile_t5xl_spiece.model. The embedded
        // Google-T5 spiece used here still denoises into a coherent image but not the prompted one — resolving
        // the Pile-T5 spiece is required for real-weight parity.
        T5Tokenizer tokenizer = new T5Tokenizer(maxLength: 256);

        AuraFlowPipeline pipeline = new AuraFlowPipeline(context.Backend, t5, transformer, vae, config);
        Logs.Info("[AuraFlowRecipe] AuraFlow ready.");
        return new AuraFlowRecipePipeline(pipeline, tokenizer, mainLoader);
    }

    /// <summary>Casts a VAE weight dictionary to <paramref name="target"/>, leaving tensors already at that dtype untouched (the SDXL-VAE BF16/F32 precision policy — never F16).</summary>
    private static Dictionary<string, Tensor> CastWeights(Dictionary<string, Tensor> weights, DType target)
    {
        Dictionary<string, Tensor> result = new Dictionary<string, Tensor>(weights.Count);
        foreach (KeyValuePair<string, Tensor> kv in weights)
        {
            result[kv.Key] = kv.Value.DType == target ? kv.Value : kv.Value.CastTo(target);
        }
        return result;
    }
}
