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
/// models. Text-to-image only: glyph text images, reference-image editing, LoRA and refiners are not implemented.</summary>
public sealed class Fibo2Recipe : IArchitectureRecipe
{
    /// <summary>ComfyUI's Fibo 2.0 shift 1.15 is a Flux-type exponent: <c>sigma(t) = e^1.15 / (e^1.15 + 1/t - 1)</c>, which is
    /// the scheduler's <c>s·t / (1 + (s - 1)·t)</c> with <c>s = e^1.15</c>, not <c>s = 1.15</c>.</summary>
    private static readonly double FluxTimeShift = Math.Exp(1.15);

    /// <summary>Fibo 2.0 dev's sampling settings. Steps and CFG come from the diffusers v1 pipeline example (50 steps, guidance 5);
    /// the 2.0 dev defaults are not documented yet, so revisit them against the Bria model card. The 1.15 shift is ComfyUI's.</summary>
    public static ImageDefaults FamilyDefaults { get; } = new ImageDefaults
    {
        Steps = 50, CfgScale = 5.0f, Width = 1024, Height = 1024, SigmaShift = FluxTimeShift,
    };

    /// <summary>Edit-turbo's sampling settings from ComfyUI's edit-turbo workflow: 4 steps, 1024×704. The workflow sets no CFG, so
    /// guidance is off (1.0), which is an assumption for the distilled build.</summary>
    public static ImageDefaults EditTurboDefaults { get; } = new ImageDefaults
    {
        Steps = 4, CfgScale = 1.0f, Width = 1024, Height = 704, SigmaShift = FluxTimeShift,
    };

    /// <inheritdoc/>
    public string Name => "fibo2";

    /// <inheritdoc/>
    /// <remarks>Text-to-image only. Img2img, reference editing, regions, LoRA and refiners are declared off until they are implemented.</remarks>
    public ImageFeatures Supports => ImageFeatures.None;

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
        || string.Equals(familyId, "fibo2-dev", StringComparison.OrdinalIgnoreCase);

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
            Dictionary<string, Tensor> dit = HalfPrecisionWeights(source.Weights, owned);
            Fibo2Config config = Fibo2Config.FromWeights(dit, Fibo2Config.Checkpoint);
            if (config.InjectionLayers.Length != Fibo2TextBundles.InjectionLayerCount)
            {
                throw new InvalidDataException($"Fibo 2.0 checkpoint has {config.InjectionLayers.Length} injection layers "
                    + $"({string.Join(",", config.InjectionLayers)}) but the text bundles feed {Fibo2TextBundles.InjectionLayerCount}.");
            }
            Logs.Info($"[Fibo2Recipe] {variant.Variant.DisplayName} ({variant.Source}): dim={config.Dim}, {config.NumLayers} layers, "
                + $"injection={string.Join(",", config.InjectionLayers)}.");

            Fibo2Transformer transformer = new(config);
            owned.Add(transformer);
            transformer.LoadWeights(dit);

            string encoderPath = ModelFileLocator.RequireOrEnsure(context.Components?.Qwen, SideModels.Fibo2Qwen3VL4B,
                "Qwen3-VL-4B text encoder (Fibo 2.0)", context.Cancel, "text_encoders", "clip");
            (Dictionary<string, Tensor> encoderWeights, CheckpointSource encoderSource) = ComponentLoader.Load(
                encoderPath, "Fibo2Recipe", CheckpointConvertUtils.RemapQwenLanguageKey, applyFp8Dequant: true);
            owned.Add(encoderSource);
            LlamaStyleEncoderConfig encoderConfig = Fibo2TextEncoderConfig.FromWeights(encoderWeights);
            Fibo2TextEncoderConfig.ValidateFor(encoderConfig, config);
            LlamaStyleEncoder encoder = new(encoderConfig);
            owned.Add(encoder);
            encoder.LoadWeights(encoderWeights);

            string vaePath = ModelFileLocator.RequireOrEnsure(context.Components?.Vae, SideModels.Fibo2Vae, "FLUX.2 VAE (Fibo 2.0)",
                context.Cancel, "VAE", "vae");
            Flux2VaeComponents vae = Flux2VaeComponents.Load(vaePath, "Fibo2Recipe");
            owned.Add(vae);

            Fibo2Pipeline pipeline = new(context.Backend, encoder, transformer, vae.Decoder, vae.BnMean, vae.BnVar, config);
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

    /// <summary>Casts BF16 weights to F16, which the CPU and GPU kernels take directly, as the Flux.2 recipe does. The copies join
    /// <paramref name="owned"/> so they outlive the transformer that borrows them; other dtypes pass through.</summary>
    /// <exception cref="NotSupportedException">A quantized tensor is present; the int8 builds are not wired yet.</exception>
    private static Dictionary<string, Tensor> HalfPrecisionWeights(IReadOnlyDictionary<string, Tensor> source, List<IDisposable> owned)
    {
        Dictionary<string, Tensor> weights = new(source.Count, StringComparer.Ordinal);
        foreach (KeyValuePair<string, Tensor> pair in source)
        {
            Tensor tensor = pair.Value;
            bool dense = tensor.DType == DType.F32 || tensor.DType == DType.F16 || tensor.DType == DType.BF16;
            if (!dense)
            {
                throw new NotSupportedException(
                    $"Fibo 2.0 tensor '{pair.Key}' is {tensor.DType}. Quantized Fibo 2.0 checkpoints (int8 ConvRot) are not wired yet; "
                    + "use the bf16 build.");
            }
            if (tensor.DType == DType.BF16)
            {
                Tensor half = tensor.CastTo(DType.F16);
                owned.Add(half);
                weights[pair.Key] = half;
            }
            else
            {
                weights[pair.Key] = tensor;
            }
        }
        return weights;
    }
}
