using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
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

/// <summary>OmniGen 2 recipe (VectorSpaceLab, 4B MMDiT, Apache-2.0). The checkpoint is the fp16 transformer; the Qwen2.5-VL-3B text encoder (<see cref="SideModels.Qwen2_5_VL_3B"/>) and the FLUX.1 16-channel VAE (<see cref="SideModels.FluxAe"/>) resolve as side models. Lifted from the SwarmUI backend's <c>OmniGen2Loader</c>; the text encoder lives OUTSIDE the pipeline, so <see cref="OmniGen2RecipePipeline"/> live-encodes the caption and calls the embeddings overload.</summary>
public sealed class OmniGen2Recipe : IArchitectureRecipe
{
    /// <inheritdoc/>
    public string Name => "omnigen2";


    /// <inheritdoc/>
    /// <remarks>Reference editing only: OmniGen2 conditions on VAE-encoded reference latents with dual
    /// text/image guidance, so there is no denoise-strength knob to honour.</remarks>
    public ImageFeatures Supports => ImageFeatures.RefEdit | ImageFeatures.SeamlessTiling | ImageFeatures.VariationSeed | ImageFeatures.Refiner | ImageFeatures.Lora;
    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "omnigen2", StringComparison.OrdinalIgnoreCase);

    /// <summary>OmniGen2's official sampling settings: 28 steps at text-guidance 4.0, 1024x1024 (<c>GenerationDefaults.OmniGen2</c>).</summary>
    public static ImageDefaults FamilyDefaults { get; } = new ImageDefaults { Steps = 28, CfgScale = 4.0f, Width = 1024, Height = 1024 };

    /// <inheritdoc/>
    public ImageDefaults Defaults => FamilyDefaults;

    /// <inheritdoc/>
    public IRecipePipeline Construct(RecipeContext context)
    {
        // TODO(E-IMG-4/5): the reference-image edit path (Init Image / Prompt Images → VAE ref latents + dual
        // text/image guidance via OmniGen2Pipeline.EditFromEmbeddings) is deferred, as are user text-encoder / VAE
        // overrides from ImageRequest.Components (the loader read T2IParamTypes.QwenModel / T2IParamTypes.VAE).
        List<SafeTensorsLoader> loaders = new List<SafeTensorsLoader>();
        try
        {
            (OmniGen2CheckpointConverter.ConvertedWeights converted, SafeTensorsLoader transformerLoader) = OmniGen2CheckpointConverter.LoadAndConvert(context.CheckpointPath);
            loaders.Add(transformerLoader);
            if (converted.Transformer.Count == 0)
            {
                throw new InvalidOperationException($"OmniGen2 checkpoint '{Path.GetFileName(context.CheckpointPath)}' contains no transformer weights.");
            }
            Logs.Info($"[OmniGen2Recipe] Parsed checkpoint: {converted.Transformer.Count} transformer tensors.");

            // BF16 (not F16, not F32): F16 overflows under CFG (NaN → all-black at cfg >= 5), F32 doubles the
            // footprint to ~16 GB. BF16 keeps 8 GB with F32's exponent range — the validated e2e recipe.
            OmniGen2Config config = OmniGen2Config.V1;
            OmniGen2Transformer transformer = new OmniGen2Transformer(config);
            // Merge any requested LoRAs BEFORE LoadWeights — device caches are identity-keyed, so merging
            // after would leave layers serving the pre-merge tensors (the Sd3Recipe ordering rule).
            MergedLoraStack? loraStack = LoraApplier.BuildAndApply(
                LoraResolver.Resolve(context.Loras), context.Backend, transformerWeights: converted.Transformer);
            transformer.LoadWeights(VaePrecisionHelper.CastWeights(converted.Transformer, [DType.F16, DType.F32], DType.BF16));

            string encoderPath = ModelDownloader.EnsureSideModelAsync(SideModels.Qwen2_5_VL_3B, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
            SafeTensorsLoader teLoader = new SafeTensorsLoader();
            teLoader.Load(encoderPath);
            loaders.Add(teLoader);
            LlamaStyleEncoder textEncoder = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Qwen2_5_VL_3B);
            textEncoder.LoadWeights(teLoader.GetAllTensors());

            string vaePath = ModelDownloader.EnsureSideModelAsync(SideModels.FluxAe, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
            (Dictionary<string, Tensor> vaeWeights, SafeTensorsLoader vaeLoader) = LoaderVaeUtils.LoadFluxVaeF32(vaePath);
            loaders.Add(vaeLoader);
            // BF16 on Ampere+ (F32-equivalent range, halves the full-res decode workspace), F32 otherwise —
            // the SDXL-VAE precision policy; LoadFluxVaeF32 force-upcasts to F32, this recovers BF16 where safe.
            vaeWeights = VaePrecisionHelper.CastVaeWeights(vaeWeights, VaePrecisionHelper.PreferredVaeDtype(context.Backend));
            VaeDecoder vae = new VaeDecoder(VaeConfig.Flux);
            vae.LoadWeights(vaeWeights);
            // The encoder half is what the deferred reference-image edit path needs; constructing with it keeps
            // OmniGen2Pipeline.EditFromEmbeddings reachable once E-IMG-4 lands.
            VaeEncoder vaeEncoder = LoaderVaeUtils.BuildEncoder(VaeConfig.Flux, vaeWeights, "OmniGen2Recipe");

            OmniGen2Pipeline pipeline = new OmniGen2Pipeline(context.Backend, transformer, vae, vaeEncoder, config);
            Qwen3Tokenizer tokenizer = new Qwen3Tokenizer(maxLength: 512);
            Logs.Info("[OmniGen2Recipe] OmniGen 2 ready (Qwen2.5-VL-3B live encode, FLUX.1 VAE).");
            return new OmniGen2RecipePipeline(pipeline, tokenizer, textEncoder, transformer, context.Backend, loaders, loraStack);
        }
        catch (Exception ex)
        {
            Logs.Error("[OmniGen2Recipe] Construction failed.", ex);
            foreach (SafeTensorsLoader loader in loaders)
            {
                loader.Dispose();
            }
            throw;
        }
    }
}
