using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Chroma Radiance recipe (<c>lodestones/Chroma-Radiance</c>) — the pixel-space, VAE-free Chroma variant: same T5-XXL conditioning and checkpoint converter as <see cref="ChromaRecipe"/>, but the transformer carries a NeRF-style pixel decoder head (<c>nerf_image_embedder.*</c>) and outputs RGB directly, so there is NO VAE to resolve. Lifted from the SwarmUI backend's <c>ChromaRadianceLoader</c> and driven through <see cref="ChromaRadianceRecipePipeline"/>.</summary>
public sealed class ChromaRadianceRecipe : IArchitectureRecipe
{
    /// <inheritdoc/>
    public string Name => "chroma-radiance";

    /// <inheritdoc/>
    /// <remarks>Chroma Radiance is pixel-space (NeRF decode head), so img2img noises the source pixels directly with no VAE.</remarks>
    public ImageFeatures Supports => ImageFeatures.Img2Img | ImageFeatures.Inpaint;

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "chroma-radiance", StringComparison.OrdinalIgnoreCase);

    /// <summary>Chroma Radiance's official sampling settings: 50 steps at guidance 3.5, 1024x1024 (<c>ChromaRadianceConfig.DefaultSteps</c>/<c>DefaultCfgScale</c>).</summary>
    public static ImageDefaults FamilyDefaults { get; } = new ImageDefaults { Steps = 50, CfgScale = 3.5f, Width = 1024, Height = 1024 };

    /// <inheritdoc/>
    public ImageDefaults Defaults => FamilyDefaults;

    /// <inheritdoc/>
    public IRecipePipeline Construct(RecipeContext context)
    {
        // TODO(E-IMG-4): honor a user-picked T5 override from ImageRequest.Components (the SwarmUI loader read
        // input.Get(T2IParamTypes.T5XXLModel)); this always takes the canonical SideModels entry.
        string t5Path = ModelDownloader.EnsureSideModelAsync(SideModels.T5XxlEnconly, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();

        // 1. Load + convert (same converter as Chroma; the radiance keys pass through).
        (ChromaCheckpointConverter.ConvertedWeights conv, SafeTensorsLoader loader) = ChromaCheckpointConverter.LoadAndConvert(context.CheckpointPath);
        if (conv.Transformer.Count == 0)
        {
            loader.Dispose();
            throw new InvalidOperationException("Chroma Radiance checkpoint has no recognized transformer weights after conversion.");
        }
        if (!ChromaRadianceConfig.IsRadiance(conv.Transformer))
        {
            loader.Dispose();
            throw new InvalidOperationException(
                "Checkpoint converted but has no Radiance pixel-decoder keys (nerf_image_embedder.*). " +
                "This looks like a vanilla Chroma checkpoint — load it as a 'chroma' model instead.");
        }
        ChromaRadianceConfig config = ChromaRadianceConfig.FromWeights(conv.Transformer);
        Logs.Info($"[ChromaRadianceRecipe] Architecture: pixel-space, patch={config.PatchSize}, nerf hidden={config.NerfHidden}.");
        ChromaRadianceTransformer transformer = new ChromaRadianceTransformer(config);
        transformer.LoadWeights(conv.Transformer);

        // 2. T5-XXL + its embedded tokenizer. No VAE — Radiance is pixel-space.
        SafeTensorsLoader t5Loader = new SafeTensorsLoader();
        t5Loader.Load(t5Path);
        Dictionary<string, Tensor> t5Weights = NormalizeT5(t5Loader.GetAllTensors());
        if (t5Weights.Count == 0)
        {
            t5Loader.Dispose();
            loader.Dispose();
            throw new InvalidOperationException($"T5 model file '{t5Path}' has no usable T5 tensors.");
        }
        T5TextEncoder t5 = new T5TextEncoder(T5TextEncoderConfig.Xxl);
        t5.LoadWeights(t5Weights);
        T5Tokenizer tokenizer = new T5Tokenizer(maxLength: 512);

        ChromaRadiancePipeline pipeline = new ChromaRadiancePipeline(context.Backend, t5, transformer, config);
        Logs.Info("[ChromaRadianceRecipe] Chroma Radiance ready (mid-pretraining checkpoint — output is validation-gated).");
        return new ChromaRadianceRecipePipeline(pipeline, config, tokenizer, loader, t5Loader);
    }

    /// <summary>Strips Comfy's <c>text_encoders.t5xxl.transformer.</c> prefix from a standalone T5-XXL safetensors file so the encoder finds its keys.</summary>
    private static Dictionary<string, Tensor> NormalizeT5(IReadOnlyDictionary<string, Tensor> raw)
    {
        Dictionary<string, Tensor> result = new Dictionary<string, Tensor>(raw.Count);
        const string ComfyPrefix = "text_encoders.t5xxl.transformer.";
        foreach (KeyValuePair<string, Tensor> kv in raw)
        {
            if (kv.Key.StartsWith(ComfyPrefix, StringComparison.Ordinal))
            {
                result[kv.Key[ComfyPrefix.Length..]] = kv.Value;
            }
            else
            {
                result[kv.Key] = kv.Value;
            }
        }
        return result;
    }
}
