using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Variants;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Fibo 2.0 recipe (Bria AI, CC-BY-NC-4.0): a single-stream DiT on the Qwen3-VL-4B text encoder with gist tokens and
/// tanh-gated injection, over the FLUX.2 32-channel VAE. The checkpoint is the transformer; the text encoder and VAE resolve as side
/// models. Text-to-image, img2img, mask inpaint and LoRA (merged into the F16 weights before load; bf16 builds only); glyph text images, reference-image editing and refiners are not implemented.</summary>
public sealed class Fibo2Recipe : IArchitectureRecipe
{
    /// <summary>ComfyUI's Fibo 2.0 shift 1.15 is a Flux-type exponent: <c>sigma(t) = e^1.15 / (e^1.15 + 1/t - 1)</c>, which is
    /// the scheduler's <c>s·t / (1 + (s - 1)·t)</c> with <c>s = e^1.15</c>, not <c>s = 1.15</c>.</summary>
    private static readonly double FluxTimeShift = Math.Exp(1.15);

    /// <summary>Fibo 2.0 dev's sampling settings, from the Bria model card (briaai/Fibo-2.0-dev: 50 steps, guidance 5.0, 1024×1024 or
    /// ~1 MP in multiples of 16). The 1.15 shift is ComfyUI's.</summary>
    public static ImageDefaults FamilyDefaults { get; } = new ImageDefaults
    {
        Steps = 50, CfgScale = 5.0f, Width = 1024, Height = 1024, SigmaShift = FluxTimeShift,
    };

    /// <summary>Edit-turbo's sampling settings: 4 steps and guidance 1.0 (no CFG) per the Bria model card (briaai/Fibo-2.0-edit-turbo),
    /// 1024×704 from ComfyUI's edit-turbo workflow.</summary>
    public static ImageDefaults EditTurboDefaults { get; } = new ImageDefaults
    {
        Steps = 4, CfgScale = 1.0f, Width = 1024, Height = 704, SigmaShift = FluxTimeShift,
    };

    /// <inheritdoc/>
    public string Name => "fibo2";

    /// <inheritdoc/>
    /// <remarks>Text-to-image, img2img, mask inpaint (a latent blend over the init image; Bria's own conditioned masked edit is
    /// reference editing and is separate) and LoRA. Reference editing, regions and refiners are declared off until they are implemented.</remarks>
    public ImageFeatures Supports => ImageFeatures.Img2Img | ImageFeatures.Inpaint | ImageFeatures.Lora;

    /// <inheritdoc/>
    public ImageDefaults Defaults => FamilyDefaults;

    /// <inheritdoc/>
    public ModelVariantCatalog? Variants => Fibo2Variants.Catalog;

    /// <inheritdoc/>
    public ImageDefaults DefaultsFor(ResolvedModelVariant? variant) =>
        variant is not null && variant.Is(Fibo2Variants.EditTurbo) ? EditTurboDefaults : FamilyDefaults;

    /// <inheritdoc/>
    /// <remarks>Matches the catalog id <c>fibo2-dev</c> as well, since the registry resolves a catalog entry by its id.</remarks>
    public bool Matches(string familyId) =>
        string.Equals(familyId, "fibo2", StringComparison.OrdinalIgnoreCase)
        || string.Equals(familyId, "fibo2-dev", StringComparison.OrdinalIgnoreCase)
        || string.Equals(familyId, "fibo2-dev-int8", StringComparison.OrdinalIgnoreCase)
        || string.Equals(familyId, "fibo2-edit-turbo-int8", StringComparison.OrdinalIgnoreCase);

    /// <inheritdoc/>
    public IRecipePipeline Construct(RecipeContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        ResolvedModelVariant variant = context.ResolveVariant(Fibo2Variants.Catalog);
        List<IDisposable> owned = [];
        try
        {
            CheckpointSource source = CheckpointSource.Open(context.CheckpointPath);
            owned.Add(source);
            Dictionary<string, Tensor> dit = Fibo2WeightPreparation.HalfPrecisionWeights(source.Weights, owned);
            bool quantized = Fibo2WeightPreparation.HasPackedWeights(dit);
            // A packed int8 weight no executing device can read widens here, not inside the first GEMM.
            Fibo2WeightPreparation.WidenUnsupported(dit, context.TransformerBackends, DType.F16, owned);
            Fibo2Config config = Fibo2Config.FromWeights(dit, Fibo2Config.Checkpoint);
            if (config.InjectionLayers.Length != Fibo2TextBundles.InjectionLayerCount)
            {
                throw new InvalidDataException($"Fibo 2.0 checkpoint has {config.InjectionLayers.Length} injection layers "
                    + $"({string.Join(",", config.InjectionLayers)}) but the text bundles feed {Fibo2TextBundles.InjectionLayerCount}.");
            }
            Logs.Info($"[Fibo2Recipe] {variant.Variant.DisplayName} ({variant.Source}): dim={config.Dim}, {config.NumLayers} layers, "
                + $"injection={string.Join(",", config.InjectionLayers)}.");

            MergeLoras(context, dit, owned);

            Fibo2Transformer transformer = new(config);
            owned.Add(transformer);
            transformer.LoadWeights(dit);

            string encoderPath = ModelFileLocator.RequireOrEnsure(context.Components?.Qwen,
                quantized ? SideModels.Fibo2Qwen3VL4BInt8 : SideModels.Fibo2Qwen3VL4B,
                "Qwen3-VL-4B text encoder (Fibo 2.0)", context.Cancel, "text_encoders", "clip");
            (Dictionary<string, Tensor> encoderWeights, CheckpointSource encoderSource) = ComponentLoader.Load(
                encoderPath, "Fibo2Recipe", CheckpointConvertUtils.RemapQwenLanguageKey, applyFp8Dequant: true);
            owned.Add(encoderSource);
            Fibo2WeightPreparation.WidenUnsupported(encoderWeights, [context.TextEncoderBackendOrDefault], DType.BF16, owned);
            LlamaStyleEncoderConfig encoderConfig = Fibo2TextEncoderConfig.FromWeights(encoderWeights);
            Fibo2TextEncoderConfig.ValidateFor(encoderConfig, config);
            LlamaStyleEncoder encoder = new(encoderConfig);
            owned.Add(encoder);
            encoder.LoadWeights(encoderWeights);

            string vaePath = ModelFileLocator.RequireOrEnsure(context.Components?.Vae, SideModels.Fibo2Vae, "FLUX.2 VAE (Fibo 2.0)",
                context.Cancel, "VAE", "vae");
            Flux2VaeComponents vae = Flux2VaeComponents.Load(vaePath, "Fibo2Recipe");
            owned.Add(vae);

            Fibo2Pipeline pipeline = new(context.Backend, encoder, transformer, vae.Decoder, vae.BnMean, vae.BnVar, config,
                vaeEncoder: vae.Encoder);
            owned.Add(pipeline);
            Qwen3Tokenizer tokenizer = new();
            owned.Add(tokenizer);
            ImageDefaults defaults = DefaultsFor(variant);
            Logs.Info("[Fibo2Recipe] Fibo 2.0 ready.");
            return new Fibo2RecipePipeline(pipeline, tokenizer, defaults, variant, owned);
        }
        catch (Exception ex)
        {
            Logs.Error("[Fibo2Recipe] Construction failed.", ex);
            for (int i = owned.Count - 1; i >= 0; i--)
                owned[i].Dispose();
            throw;
        }
    }

    /// <summary>Merges the request's LoRA stack into <paramref name="dit"/> before the transformer binds the weights (device caches are
    /// identity-keyed). The weights are already F16 here, so each merged weight is the exact F16 base plus the F32 delta rounded once.
    /// The stack owns the merged tensors and joins <paramref name="owned"/>; every F16 copy the merge superseded is freed at once, since
    /// otherwise the whole merged set would sit beside its originals. A LoRA that matches no weight is refused by the merge.</summary>
    private static void MergeLoras(RecipeContext context, Dictionary<string, Tensor> dit, List<IDisposable> owned)
    {
        if (context.Loras is null)
            return;
        Dictionary<string, Tensor> before = new(dit, StringComparer.Ordinal);
        MergedLoraStack? stack = RecipeLoraMerge.Apply(context, new LoraMergeTargets { Transformer = dit }, "Fibo2Recipe");
        if (stack is null)
            return;
        owned.Add(stack);
        foreach (KeyValuePair<string, Tensor> pair in before)
        {
            Tensor merged = dit[pair.Key];
            if (ReferenceEquals(merged, pair.Value) || !pair.Value.OwnsMemory)
                continue;
            int index = owned.FindIndex(o => ReferenceEquals(o, pair.Value));
            if (index < 0)
                continue;
            owned.RemoveAt(index);
            pair.Value.Dispose();
        }
    }
}
