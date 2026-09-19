using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Engine.Features;
namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Boogu-Image recipe (<c>boogu-project/Boogu-Image</c>, 10B): an OmniGen2/Lumina-2 lineage DiT (8 dual-stream + 32 single-stream blocks, GQA 28:7) conditioned on Qwen3-VL-8B (<see cref="SideModels.Qwen3VL_8B"/>) and decoded by the FLUX.1 VAE (<see cref="SideModels.FluxAe"/>). Lifted from the SwarmUI backend's <c>BooguImageLoader</c>; constructs the components and drives generation through <see cref="BooguImageRecipePipeline"/>.</summary>
public sealed class BooguImageRecipe : IArchitectureRecipe
{
    /// <summary>Minimum free VRAM to attempt a CUDA load — 10B DiT (fp8 ~10 GB) + Qwen3-VL-8B + VAE + headroom.</summary>
    private const double MinRequiredVramGb = 16.0;

    /// <inheritdoc/>
    public string Name => "boogu";


    /// <inheritdoc/>
    /// <remarks>Reference editing at text-only guidance. Steerable image guidance needs the Qwen3-VL vision
    /// tower for the text-and-image-dropped embedding, which is still deferred.
    /// <para><see cref="ImageFeatures.Lora"/> added 2026-08-20. <see cref="HartsyInference.Diffusion.Models.Denoisers.BooguImageTransformer"/> names its stacks <c>double_stream_layers.{i}</c>, <c>noise_refiner.{i}</c>, <c>context_refiner.{i}</c> and <c>ref_image_refiner.{i}</c> — none of which the bare-root LoRA detector recognized before the same change.</para></remarks>
    public ImageFeatures Supports => ImageFeatures.RefEdit | ImageFeatures.SeamlessTiling | ImageFeatures.VariationSeed | ImageFeatures.Refiner | ImageFeatures.Lora;

    /// <inheritdoc/>
    /// <remarks>Ledger evidence in <c>PromptWeightingModeLedgerTests</c>: <c>supported_models.py:1927</c> →
    /// <c>boogu.BooguTokenizer</c>, a <c>Qwen3VLTokenizer</c> subclass, which disables weights
    /// (<c>qwen3vl.py:187</c>).</remarks>
    public Diffusion.Prompting.PromptWeightingMode PromptWeighting =>
        Diffusion.Prompting.PromptWeightingMode.CondScale;
    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "boogu", StringComparison.OrdinalIgnoreCase);

    /// <summary>Boogu's sampling settings: 25 steps at text-guidance 3.5, 1024x1024 — the step count is the lifted loader's, and 3.5 sits in the 2-5 band the base model is documented to work in (Turbo wants 1.0).</summary>
    public static ImageDefaults FamilyDefaults { get; } = new ImageDefaults { Steps = 25, CfgScale = 3.5f, Width = 1024, Height = 1024 };

    /// <inheritdoc/>
    public ImageDefaults Defaults => FamilyDefaults;

    /// <inheritdoc/>
    public IRecipePipeline Construct(RecipeContext context)
    {
        // Asked of the backend, not of its class: any GPU backend that reports memory can answer this, and the
        // preflight is about how much VRAM there is rather than about which vendor supplies it.
        (long freeBytes, long totalBytes) = context.Backend.GetVramInfo();
        if (totalBytes > 0)
        {
            double freeGb = freeBytes / (1024.0 * 1024.0 * 1024.0);
            if (freeGb < MinRequiredVramGb)
            {
                throw new InvalidOperationException(
                    $"Boogu-Image needs ≥{MinRequiredVramGb:F0} GB free VRAM (10B DiT + Qwen3-VL-8B + VAE); " +
                    $"this GPU has {freeGb:F1} GB free of {totalBytes / (1024.0 * 1024.0 * 1024.0):F1} GB total.");
            }
        }
        else
        {
            Logs.Warning("[BooguImageRecipe] Non-CUDA backend — a 10B DiT per step will be extremely slow.");
        }

        // TODO(E-IMG-4): honor user-picked Qwen3-VL / VAE overrides from ImageRequest.Components (the SwarmUI loader
        // header-probed input.Get(T2IParamTypes.QwenModel/VAE) for a 4096-dim Qwen3-VL and a 16-channel FLUX.1 ae).
        string tePath = ModelDownloader.EnsureSideModelAsync(SideModels.Qwen3VL_8B, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
        string vaePath = ModelDownloader.EnsureSideModelAsync(SideModels.FluxAe, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();

        List<IDisposable> loaders = new List<IDisposable>();
        try
        {
            (Dictionary<string, Tensor> transformerW, CheckpointSource transformerSource) =
                ComponentLoader.Load(context.CheckpointPath, "BooguImageRecipe", CheckpointConvertUtils.StripTransformerPrefix, applyFp8Dequant: true);
            loaders.Add(transformerSource);
            // Any quant this run's devices have no packed-weight kernel for widens here rather than failing inside
            // the first GEMM, minutes into a generation.
            loaders.Add(QuantizedWeightPolicy.PrepareForBackends(transformerW, context.TransformerBackends));

            // TODO(E-IMG-4): the reference-image edit path also loads the Qwen3-VL vision tower (visual.* keys) and
            // wires a Qwen3VlMultimodalEncoder. Text-to-image only here, so the language tower alone is loaded.
            (Dictionary<string, Tensor> teW, CheckpointSource teSource) =
                ComponentLoader.Load(tePath, "BooguImageRecipe", CheckpointConvertUtils.RemapQwenLanguageKey, applyFp8Dequant: true);
            loaders.Add(teSource);

            // The auto-downloaded flux_ae.safetensors ships BFL-native LDM keys; ConvertVaeKey remaps LDM → diffusers
            // and passes already-diffusers keys through unchanged (a raw load throws on mid_block.resnets.0).
            (Dictionary<string, Tensor> vaeW, SafeTensorsLoader vaeL) = LoaderVaeUtils.LoadFluxVaeF32(vaePath);
            loaders.Add(vaeL);
            // BF16 on Ampere+ (F32-equivalent range, halves the full-res decode workspace), F32 otherwise —
            // the SDXL-VAE precision policy; LoadFluxVaeF32 force-upcasts to F32, this recovers BF16 where safe.
            vaeW = VaePrecisionHelper.CastVaeWeights(vaeW, VaePrecisionHelper.PreferredVaeDtype(context.Backend));

            BooguImageConfig config = BooguImageConfig.V01;
            BooguImageTransformer transformer = new BooguImageTransformer(config);
            // Merge any requested LoRAs BEFORE LoadWeights — device caches are identity-keyed, so merging
            // after would leave layers serving the pre-merge tensors (the Sd3Recipe ordering rule).
            MergedLoraStack? loraStack = RecipeLoraMerge.Apply(
                context,
                new LoraMergeTargets { Transformer = transformerW },
                "BooguImageRecipe");
            transformer.LoadWeights(transformerW);

            LlamaStyleEncoder textEncoder = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Qwen3_VL_8B);
            textEncoder.LoadWeights(teW);

            VaeDecoder vaeDecoder = new VaeDecoder(VaeConfig.Flux);
            vaeDecoder.LoadWeights(vaeW);
            VaeEncoder vaeEncoder = LoaderVaeUtils.BuildEncoder(VaeConfig.Flux, vaeW, "BooguImageRecipe");

            Qwen3Tokenizer tokenizer = new Qwen3Tokenizer(maxLength: 4096);
            BooguImagePipeline pipeline = new BooguImagePipeline(context.Backend, transformer, vaeDecoder, vaeEncoder, config);
            Logs.Info("[BooguImageRecipe] Boogu-Image ready (text-to-image).");
            return new BooguImageRecipePipeline(pipeline, tokenizer, textEncoder, transformer, context.Backend, loaders, loraStack);
        }
        catch (Exception ex)
        {
            Logs.Error("[BooguImageRecipe] Construction failed.", ex);
            foreach (IDisposable loader in loaders)
            {
                loader.Dispose();
            }
            throw;
        }
    }
}
