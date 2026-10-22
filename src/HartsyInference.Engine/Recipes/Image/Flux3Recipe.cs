using HartsyInference.Core.Logging;
using HartsyInference.Core.Memory;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Variants;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Tokenizers;
using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>FLUX.3 Klein recipe (Black Forest Labs, ~5.6B per-mode + joint DiT; text-to-image, editing with up to 10
/// references, native layout boxes). The checkpoint is the transformer in ComfyUI's single-file layout (BF16, int8
/// ConvRot or NVFP4); the Qwen3-VL-4B text encoder (<see cref="SideModels.Qwen3VL_4B"/>, the file SwarmUI loads for this
/// class; BFL's int8 ConvRot copy works as a component override) and the Flux.2 VAE
/// (<see cref="SideModels.Flux2Vae"/>) resolve as side models. Full vs step-distilled is structural (see
/// <see cref="Flux3Variants"/>). Drives through <see cref="Flux3RecipePipeline"/>.</summary>
public sealed class Flux3Recipe : IArchitectureRecipe
{
    /// <summary>The full build's sampling settings (ComfyUI template): 25 steps at true-CFG 3.5, 1024².</summary>
    public static ImageDefaults FamilyDefaults { get; } = new ImageDefaults { Steps = 25, CfgScale = 3.5f, Width = 1024, Height = 1024 };

    /// <summary>The step-distilled build's settings: 16 steps (4–16) at CFG 1 (SwarmUI <c>Model Support.md</c>, Flux 3 Image 'sd').</summary>
    public static ImageDefaults DistilledDefaults { get; } = new ImageDefaults { Revision = 2, Steps = 16, CfgScale = 1.0f, Width = 1024, Height = 1024 };

    /// <inheritdoc/>
    public string Name => "flux3";

    /// <inheritdoc/>
    /// <remarks>Regional prompts become the model's own layout table (<see cref="LayoutPromptBuilder"/>) rather than an
    /// attention bias; references and the init image are both VAE-encoded, so edits need no vision tower.</remarks>
    public ImageFeatures Supports => ImageFeatures.Img2Img | ImageFeatures.Inpaint | ImageFeatures.RefEdit | ImageFeatures.Regional
        | ImageFeatures.SeamlessTiling | ImageFeatures.VariationSeed | ImageFeatures.Lora;

    /// <inheritdoc/>
    /// <remarks>Ten references, plus an init image that is denoised; an init image in Reference mode is one of the ten.
    /// References stand alone.</remarks>
    public ImageInputLimits InputLimits => new ImageInputLimits(Flux3Config.ReferenceImageLimit + 1, ReferencesRequireInitImage: false,
        MaxReferences: Flux3Config.ReferenceImageLimit);

    /// <inheritdoc/>
    /// <remarks>Ledger evidence in <c>PromptWeightingModeLedgerTests</c>: ComfyUI's <c>Qwen3VLTokenizer</c> passes
    /// <c>disable_weights=True</c>, so the prompt is encoded at weight 1 and each token's cond row is scaled afterwards.
    /// Nothing is trimmed off the encoder output, so the alignment offset is 0.</remarks>
    public Diffusion.Prompting.PromptWeightingMode PromptWeighting => Diffusion.Prompting.PromptWeightingMode.CondScale;

    /// <inheritdoc/>
    public ImageDefaults Defaults => FamilyDefaults;

    /// <inheritdoc/>
    public ModelVariantCatalog? Variants => Flux3Variants.Catalog;

    /// <inheritdoc/>
    public ImageDefaults DefaultsFor(ResolvedModelVariant? variant) =>
        variant is not null && variant.Is(Flux3Variants.Distilled) ? DistilledDefaults : FamilyDefaults;

    /// <inheritdoc/>
    public bool Matches(string familyId) =>
        string.Equals(familyId, "flux3", StringComparison.OrdinalIgnoreCase)
        || string.Equals(familyId, "flux3-klein", StringComparison.OrdinalIgnoreCase);

    /// <inheritdoc/>
    public IRecipePipeline Construct(RecipeContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        string fileName = Path.GetFileName(context.CheckpointPath);
        ResolvedModelVariant variant = context.ResolveVariant(Flux3Variants.Catalog);
        List<IDisposable> owned = new List<IDisposable>();
        try
        {
            CheckpointSource source = CheckpointSource.Open(context.CheckpointPath, CheckpointOpenOptions.ForNvfp4Consumer(context.Backend));
            owned.Add(source);
            Dictionary<string, Tensor> converted = Flux3CheckpointConverter.ConvertTransformer(source.Weights);
            owned.Add(QuantizedWeightPolicy.PrepareForBackends(converted, context.TransformerBackends));
            Flux3Config config = Flux3Config.FromWeights(converted);
            Logs.Info($"[Flux3Recipe] {variant.Variant.DisplayName} ({variant.Source}): {fileName}, hidden={config.HiddenSize}, "
                + $"{config.Depth} per-mode + {config.DepthSingleBlocks} joint blocks, flow map={config.FlowMap}.");

            Flux3Transformer transformer = new Flux3Transformer(config);
            owned.Add(transformer);
            // Merged before LoadWeights: device caches are identity-keyed, so a later merge would never be seen.
            MergedLoraStack? loraStack = RecipeLoraMerge.Apply(context, new LoraMergeTargets { Transformer = converted }, "Flux3Recipe");
            if (loraStack is not null) owned.Add(loraStack);
            transformer.LoadWeights(converted);

            string encoderPath = ModelFileLocator.RequireOrEnsure(context.Components?.Qwen, SideModels.Qwen3VL_4B,
                "Qwen3-VL-4B text encoder", context.Cancel, "text_encoders", "clip");
            (Dictionary<string, Tensor> encoderWeights, CheckpointSource encoderSource) = ComponentLoader.Load(
                encoderPath, "Flux3Recipe", CheckpointConvertUtils.RemapQwenLanguageKey, applyFp8Dequant: true);
            owned.Add(encoderSource);
            LlamaStyleEncoder encoder = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Qwen3_VL_4B);
            owned.Add(encoder);
            encoder.LoadWeights(encoderWeights);

            string vaePath = ModelFileLocator.RequireOrEnsure(context.Components?.Vae, SideModels.Flux2Vae, "Flux.2 VAE",
                context.Cancel, "VAE", "vae");
            Flux2VaeComponents vae = Flux2VaeComponents.Load(vaePath, "Flux3Recipe");
            owned.Add(vae);

            Flux3Pipeline pipeline = new Flux3Pipeline(context.Backend, encoder, transformer, vae.Decoder, vae.Encoder, vae.BnMean, vae.BnVar);
            owned.Add(pipeline);
            Qwen3Tokenizer tokenizer = new Qwen3Tokenizer();
            owned.Add(tokenizer);
            Logs.Info("[Flux3Recipe] FLUX.3 Klein ready.");
            return new Flux3RecipePipeline(pipeline, tokenizer, config, variant, owned);
        }
        catch (Exception ex)
        {
            Logs.Error("[Flux3Recipe] Construction failed.", ex);
            new CompositeDisposable([.. Enumerable.Reverse(owned)]).Dispose();
            throw;
        }
    }
}
