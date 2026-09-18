using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Core.Memory;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

using HartsyInference.Engine.Features;
namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Zeta-Chroma recipe (<c>lodestones/Zeta-Chroma</c>) — the pixel-space, VAE-free Chroma variant that swaps T5 for Qwen3-4B caption conditioning (<see cref="SideModels.Qwen3_4B"/>, the same encoder path as <see cref="ZImageRecipe"/>) and predicts x0 directly in pixel space. No CLIP, no VAE. Lifted from the SwarmUI backend's <c>ZetaChromaLoader</c> and driven through <see cref="ZetaChromaRecipePipeline"/>.</summary>
public sealed class ZetaChromaRecipe : IArchitectureRecipe
{
    /// <inheritdoc/>
    public string Name => "zeta-chroma";

    /// <inheritdoc/>
    /// <remarks>Zeta-Chroma is pixel-space — the source image IS the clean sample, so no VAE encoder is involved.</remarks>
    public ImageFeatures Supports => ImageFeatures.Img2Img | ImageFeatures.Inpaint | ImageFeatures.SeamlessTiling | ImageFeatures.VariationSeed | ImageFeatures.Refiner | ImageFeatures.Lora;

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "zeta-chroma", StringComparison.OrdinalIgnoreCase);

    /// <summary>Zeta-Chroma's official sampling settings: 50 steps at guidance 3.0, 1024x1024 (<c>ZetaChromaConfig.DefaultSteps</c>/<c>DefaultCfgScale</c>).</summary>
    public static ImageDefaults FamilyDefaults { get; } = new ImageDefaults { Steps = 50, CfgScale = 3.0f, Width = 1024, Height = 1024 };

    /// <inheritdoc/>
    public ImageDefaults Defaults => FamilyDefaults;

    /// <inheritdoc/>
    public IRecipePipeline Construct(RecipeContext context)
    {
        // TODO(E-IMG-4): honor a user-picked Qwen override from ImageRequest.Components (the SwarmUI loader read
        // input.Get(T2IParamTypes.QwenModel)); this always takes the canonical SideModels entry.
        string qwenPath = ModelDownloader.EnsureSideModelAsync(SideModels.Qwen3_4B, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();

        List<IDisposable> loaders = new List<IDisposable>();
        IDisposable? checkpoint = null;
        try
        {
            // 1. Load + convert (Z-Image-derived converter), through the one container so a GGUF or fp8_scaled
            // repack of the same file reaches the converter identically.
            CheckpointSource source = CheckpointSource.Open(context.CheckpointPath);
            checkpoint = source;
            ZImageCheckpointConverter.ConvertedWeights conv = ZetaChromaCheckpointConverter.Convert(source.Weights);
            if (conv.Transformer.Count == 0)
            {
                throw new InvalidOperationException("Zeta-Chroma checkpoint has no recognized transformer weights after conversion.");
            }
            // Any quant this run's devices have no packed-weight kernel for widens here rather than failing inside
            // the first GEMM. Tracked immediately so a failure further down frees the widened copies.
            QuantizedWeightPolicy.PreparedWeights prepared =
                QuantizedWeightPolicy.PrepareForBackends(conv.Transformer, context.TransformerBackends);
            checkpoint = new CompositeDisposable(source, prepared);

            ZetaChromaConfig config = ZetaChromaConfig.FromWeights(conv.Transformer);
            Logs.Info($"[ZetaChromaRecipe] Architecture: pixel-space, patch={config.PatchSize}, x0-prediction.");
            ZetaChromaTransformer transformer = new ZetaChromaTransformer(config);
            // Merge any requested LoRAs BEFORE LoadWeights — device caches are identity-keyed, so merging
            // after would leave layers serving the pre-merge tensors (the Sd3Recipe ordering rule).
            MergedLoraStack? loraStack = RecipeLoraMerge.Apply(
                context,
                new LoraMergeTargets { Transformer = conv.Transformer },
                "ZetaChromaRecipe");
            transformer.LoadWeights(conv.Transformer);

            // 2. Qwen3-4B caption encoder — its weights are uploaded and freed per generation, like Z-Image.
            SafeTensorsLoader qwenLoader = new SafeTensorsLoader();
            qwenLoader.Load(qwenPath);
            loaders.Add(qwenLoader);
            IReadOnlyDictionary<string, Tensor> qwenWeights = qwenLoader.GetAllTensors();
            if (qwenWeights.Count == 0)
            {
                throw new InvalidOperationException($"Qwen3 model file '{qwenPath}' has no tensors.");
            }
            LlamaStyleEncoder qwen = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Qwen3_4B);
            qwen.LoadWeights(qwenWeights);
            Qwen3Tokenizer tokenizer = new Qwen3Tokenizer(maxLength: 256);

            ZetaChromaPipeline pipeline = new ZetaChromaPipeline(context.Backend, transformer, config);
            Logs.Info("[ZetaChromaRecipe] Zeta-Chroma ready (mid-pretraining checkpoint — output is validation-gated).");
            return new ZetaChromaRecipePipeline(pipeline, config, qwen, tokenizer, context.Backend, checkpoint, loaders, loraStack);
        }
        catch (Exception ex)
        {
            Logs.Error("[ZetaChromaRecipe] Construction failed.", ex);
            foreach (IDisposable loader in loaders)
            {
                loader.Dispose();
            }
            checkpoint?.Dispose();
            throw;
        }
    }
}
