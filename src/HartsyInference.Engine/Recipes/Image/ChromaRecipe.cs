using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Core.Memory;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Placement;
namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Chroma recipe (Lodestone Rock's 8.9B Flux derivative: T5-only, no CLIP/pooled conditioning, joint-attention DiT, same VAE as Flux.1). Lifted from the SwarmUI backend's <c>ChromaLoader</c>; the checkpoint is the DiT, the T5-XXL text encoder and Flux VAE are resolved as side models. Constructs and drives through <see cref="ChromaRecipePipeline"/>.</summary>
public sealed class ChromaRecipe : IArchitectureRecipe
{
    /// <inheritdoc/>
    public string Name => "chroma";


    /// <inheritdoc/>
    /// <remarks>Chroma reuses the Flux.1 VAE; the encoder half is built alongside the decoder and ChromaPipeline implements the packed-latent masked path.</remarks>
    public ImageFeatures Supports => ImageFeatures.Img2Img | ImageFeatures.Inpaint | ImageFeatures.SeamlessTiling | ImageFeatures.VariationSeed | ImageFeatures.Refiner | ImageFeatures.Lora;

    /// <inheritdoc/>
    /// <remarks>Ledger evidence in <c>PromptWeightingModeLedgerTests</c>: <c>supported_models.py:1797</c> →
    /// <c>pixart_t5.PixArtTokenizer</c> → <c>T5XXLTokenizer</c>, which does not disable weights.</remarks>
    public Diffusion.Prompting.PromptWeightingMode PromptWeighting =>
        Diffusion.Prompting.PromptWeightingMode.ComfyBlend;
    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "chroma", StringComparison.OrdinalIgnoreCase);

    /// <summary>Chroma's official sampling settings: 35 steps at guidance 5.0, 1024x1024 (<c>ChromaConfig.DefaultSteps</c>/<c>ChromaConfig.DefaultCfgScale</c>).</summary>
    public static ImageDefaults FamilyDefaults { get; } = new ImageDefaults { Steps = 35, CfgScale = 5.0f, Width = 1024, Height = 1024 };

    /// <inheritdoc/>
    public ImageDefaults Defaults => FamilyDefaults;

    /// <inheritdoc/>
    /// <inheritdoc/>
    public MemoryCapabilities MemorySupports => MemoryCapabilities.DitSharding | MemoryCapabilities.ComponentPlacement;

    public IRecipePipeline Construct(RecipeContext context)
    {
        // Resolve the two side models synchronously (Construct is a sync seam): T5-XXL (encoder-only fp8) + Flux VAE.
        // TODO(E-IMG-4): honor a user-picked T5/VAE override from ImageRequest.Components/Extra instead of always
        // taking the canonical SideModels entry (the SwarmUI loader read input.Get(T2IParamTypes.T5XXLModel/VAE)).
        string t5Path = ModelDownloader.EnsureSideModelAsync(SideModels.T5XxlEnconly, onProgress: null, context.Cancel).GetAwaiter().GetResult();
        string vaePath = ModelDownloader.EnsureSideModelAsync(SideModels.FluxAe, onProgress: null, context.Cancel).GetAwaiter().GetResult();

        List<IDisposable> loaders = new List<IDisposable>();
        IDisposable? checkpoint = null;
        try
        {
            // 1. Load + convert the Chroma transformer. One container for either format: a Chroma GGUF is a repack of
            // this same file under the same BFL key names, so the converter cannot tell them apart.
            CheckpointSource source = CheckpointSource.Open(context.CheckpointPath);
            checkpoint = source;
            ChromaCheckpointConverter.ConvertedWeights zConv = ChromaCheckpointConverter.Convert(source.Weights);
            if (zConv.Transformer.Count == 0)
            {
                throw new InvalidOperationException("Chroma checkpoint has no recognized transformer weights after conversion.");
            }
            // Any quant this run's devices have no packed-weight kernel for widens here rather than failing inside
            // the first GEMM, minutes into a generation. Tracked immediately so a failure further down frees the
            // widened copies rather than leaving them to the finalizer.
            QuantizedWeightPolicy.PreparedWeights prepared =
                QuantizedWeightPolicy.PrepareForBackends(zConv.Transformer, context.TransformerBackends);
            checkpoint = new CompositeDisposable(source, prepared);

            ChromaConfig config = ChromaConfig.V1;
            Logs.Info($"[ChromaRecipe] Building transformer ({config.HiddenSize} hidden, {config.Depth} double / {config.DepthSingleBlocks} single).");
            ChromaTransformer transformer = new ChromaTransformer(config);
            // Merge any requested LoRAs BEFORE LoadWeights — device caches are identity-keyed, so merging
            // after would leave layers serving the pre-merge tensors (the Sd3Recipe ordering rule).
            MergedLoraStack? loraStack = RecipeLoraMerge.Apply(
                context,
                new LoraMergeTargets { Transformer = zConv.Transformer },
                "ChromaRecipe");
            transformer.LoadWeights(zConv.Transformer);

            // DiT sharding split point — byte-weighted: Chroma's 19 double blocks are ~2× its 38 single blocks,
            // so a count-proportional split would misallocate by GBs. Computed post-load (needs live free VRAM).
            int ditShardSplitBlock = 0;
            if (context.DitShardBackend is not null)
            {
                ditShardSplitBlock = DitShardPlanner.SplitBlockByBytes(
                    context.Backend, context.DitShardBackend, transformer.BlockCount,
                    transformer.EnumerateBlockRangeWeights, transformer.EnumerateSharedWeights());
                Logs.Info($"[ChromaRecipe] DiT sharding enabled: blocks [0,{ditShardSplitBlock}) on the primary "
                    + $"backend, [{ditShardSplitBlock},{transformer.BlockCount}) on the shard backend "
                    + "(sequential dual-pass CFG; the step graph is disabled while sharded).");
            }

            // 2. Load T5-XXL + its embedded tokenizer.
            SafeTensorsLoader t5Loader = new SafeTensorsLoader();
            t5Loader.Load(t5Path);
            loaders.Add(t5Loader);
            Dictionary<string, Tensor> t5Weights = LoaderPrefixUtils.StripT5XxlPrefix(t5Loader.GetAllTensors());
            if (t5Weights.Count == 0)
            {
                throw new InvalidOperationException($"T5 model file '{t5Path}' has no usable T5 tensors.");
            }
            T5TextEncoder t5 = new T5TextEncoder(T5TextEncoderConfig.Xxl);
            t5.LoadWeights(t5Weights);
            T5Tokenizer tokenizer = new T5Tokenizer(maxLength: 512);

            // 3. Load the Flux VAE (Chroma reuses it verbatim).
            (Dictionary<string, Tensor> vaeWeights, SafeTensorsLoader vaeLoader) = LoaderVaeUtils.LoadFluxVaeF32(vaePath);
            loaders.Add(vaeLoader);
            if (vaeWeights.Count == 0)
            {
                throw new InvalidOperationException($"VAE file '{vaePath}' has no usable VAE tensors.");
            }
            // BF16 on Ampere+ (F32-equivalent range, halves the full-res decode workspace), F32 otherwise —
            // the SDXL-VAE precision policy; LoadFluxVaeF32 force-upcasts to F32, this recovers BF16 where safe.
            vaeWeights = VaePrecisionHelper.CastVaeWeights(vaeWeights, VaePrecisionHelper.PreferredVaeDtype(context.Backend));
            VaeDecoder vae = new VaeDecoder(VaeConfig.Chroma);
            vae.LoadWeights(vaeWeights);

            VaeEncoder? vaeEncoder = LoaderVaeUtils.TryBuildEncoder(VaeConfig.Chroma, vaeWeights, "ChromaRecipe");
            ChromaPipeline pipeline = new ChromaPipeline(context.Backend, t5, transformer, vae, vaeEncoder, config)
            {
                TextEncoderBackend = context.TextEncoderBackendOrDefault,
                VaeBackend = context.VaeBackendOrDefault,
                DitShardBackend = context.DitShardBackend,
                DitShardSplitBlock = ditShardSplitBlock,
            };
            Logs.Info("[ChromaRecipe] Chroma ready.");
            return new ChromaRecipePipeline(pipeline, tokenizer, checkpoint, loaders, loraStack);
        }
        catch (Exception ex)
        {
            Logs.Error("[ChromaRecipe] Construction failed.", ex);
            foreach (IDisposable loader in loaders)
            {
                loader.Dispose();
            }
            checkpoint?.Dispose();
            throw;
        }
    }
}
