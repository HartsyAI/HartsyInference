using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Memory;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Engine.Features;
namespace HartsyInference.Engine.Recipes.Image;

/// <summary>AuraFlow v0.2 / v0.3 recipe (fal/AuraFlow, MMDiT + single-DiT hybrid). The single-file checkpoint bundles the transformer, the Pile-T5-XL text encoder, and the SDXL-family VAE under one safetensors; nothing is resolved as a side model. Lifted from the SwarmUI backend's <c>AuraFlowLoader</c>; constructs the components and drives generation through <see cref="AuraFlowRecipePipeline"/>.</summary>
public sealed class AuraFlowRecipe : IArchitectureRecipe
{
    /// <inheritdoc/>
    public string Name => "auraflow";


    /// <inheritdoc/>
    /// <remarks>AuraFlow reuses the SDXL VAE; the encoder half is built alongside the decoder.
    /// <para><see cref="ImageFeatures.Lora"/> added 2026-08-20. <see cref="HartsyInference.Diffusion.Models.Denoisers.AuraFlowTransformer"/> names its blocks <c>joint_transformer_blocks.{i}</c> / <c>single_transformer_blocks.{i}</c>; the first of those is a root the bare-root LoRA detector only started recognizing in the same change.</para></remarks>
    public ImageFeatures Supports => ImageFeatures.Img2Img | ImageFeatures.Inpaint | ImageFeatures.SeamlessTiling | ImageFeatures.VariationSeed | ImageFeatures.Refiner | ImageFeatures.Lora;

    /// <inheritdoc/>
    /// <remarks>Ledger evidence in <c>PromptWeightingModeLedgerTests</c>: <c>supported_models.py:661</c> →
    /// <c>aura_t5.AuraT5Tokenizer</c> → <c>PT5XlTokenizer</c>, which does NOT disable weights — so they survive
    /// tokenization and ComfyUI interpolates the encoder OUTPUT away from the empty-prompt baseline.</remarks>
    public Diffusion.Prompting.PromptWeightingMode PromptWeighting =>
        Diffusion.Prompting.PromptWeightingMode.ComfyBlend;
    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "auraflow", StringComparison.OrdinalIgnoreCase);

    /// <summary>AuraFlow's official sampling settings: 50 steps at guidance 3.5, 1024x1024 (diffusers <c>AuraFlowPipeline.__call__</c>, mirrored by <c>GenerationDefaults.AuraFlow</c>).</summary>
    public static ImageDefaults FamilyDefaults { get; } = new ImageDefaults { Steps = 50, CfgScale = 3.5f, Width = 1024, Height = 1024 };

    /// <inheritdoc/>
    public ImageDefaults Defaults => FamilyDefaults;

    /// <inheritdoc/>
    public IRecipePipeline Construct(RecipeContext context)
    {
        IDisposable? checkpoint = null;
        try
        {
            // One container for either format: an AuraFlow GGUF is a repack of this same bundled file and keeps
            // its key names, so nothing below needs to know which one arrived.
            CheckpointSource source = CheckpointSource.Open(context.CheckpointPath);
            checkpoint = source;
            AuraFlowCheckpointConverter.ConvertedWeights converted = AuraFlowCheckpointConverter.Convert(source.Weights);
            Logs.Info($"[AuraFlowRecipe] Converted: {converted.Transformer.Count} transformer / {converted.T5.Count} T5 / {converted.Vae.Count} VAE keys.");
            if (converted.Transformer.Count == 0 || converted.T5.Count == 0 || converted.Vae.Count == 0)
            {
                throw new InvalidOperationException("AuraFlow checkpoint is missing one of transformer / T5 / VAE. AuraFlow expects a complete bundled file (the v0.3 fal-released format).");
            }
            // Any quant this run's devices have no packed-weight kernel for widens here rather than failing inside
            // the first GEMM. Tracked immediately so a failure further down frees the widened copies.
            QuantizedWeightPolicy.PreparedWeights prepared =
                QuantizedWeightPolicy.PrepareForBackends(converted.Transformer, context.TransformerBackends);
            checkpoint = new CompositeDisposable(source, prepared);

            AuraFlowConfig config = AuraFlowConfig.V03;
            Logs.Info($"[AuraFlowRecipe] Building transformer ({config.NumDoubleBlocks} double + {config.NumSingleBlocks} single, V03 preset).");
            AuraFlowTransformer transformer = new AuraFlowTransformer(config);
            // Merge any requested LoRAs BEFORE LoadWeights — device caches are identity-keyed, so merging
            // after would leave layers serving the pre-merge tensors (the Sd3Recipe ordering rule).
            MergedLoraStack? loraStack = RecipeLoraMerge.Apply(
                context,
                new LoraMergeTargets { Transformer = converted.Transformer },
                "AuraFlowRecipe");
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
            // TODO: Pile-T5's special ids likely differ from the ones this tokenizer hardcodes. ComfyUI's
            // `aura_t5.py:9/:14` declares `special_tokens={"end": 2, "pad": 1}` and `pad_token=1`, while
            // `T5Tokenizer` fixes EOS=1/PAD=0 for every family it serves — so AuraFlow may be terminating and
            // padding with the wrong pieces on the plain encode. Unverified against the spiece vocab; it fails as
            // plausible output rather than an error, and a fix moves every existing AuraFlow generation.
            T5Tokenizer tokenizer = new T5Tokenizer(spiecePath, maxLength: 256);

            AuraFlowPipeline pipeline = new AuraFlowPipeline(context.Backend, t5, transformer, vae, vaeEncoder, config);
            Logs.Info("[AuraFlowRecipe] AuraFlow ready.");
            return new AuraFlowRecipePipeline(pipeline, tokenizer, checkpoint, loraStack);
        }
        catch (Exception ex)
        {
            Logs.Error("[AuraFlowRecipe] Construction failed.", ex);
            checkpoint?.Dispose();
            throw;
        }
    }
}
