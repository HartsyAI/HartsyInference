using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Engine.Features;
namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Lance (image) recipe — ByteDance's 3B unified multimodal model, which ships as a FOLDER checkpoint (model.safetensors or shards + llm_config.json + the Qwen2 tokenizer files). The Wan2.2 48-channel VAE (<see cref="SideModels.Wan22Vae"/>) resolves as a side model. Lifted from the SwarmUI backend's <c>LanceLoader</c> (image half only; the video variant is a separate family) and driven through <see cref="LanceImageRecipePipeline"/>.</summary>
public sealed class LanceImageRecipe : IArchitectureRecipe
{
    /// <inheritdoc/>
    public string Name => "lance-image";


    /// <inheritdoc/>
    /// <remarks>Img2img only, deliberately: LanceImagePipeline throws on a mask because the Wan2.2 VAE's 16x
    /// downscale leaves one mask cell per 16x16-pixel block, too coarse for the blend-on-vanilla inpaint.</remarks>
    public ImageFeatures Supports => ImageFeatures.Img2Img;
    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "lance-image", StringComparison.OrdinalIgnoreCase);

    /// <summary>Lance's official sampling settings: 30 steps at text-CFG 4.0, 1024x1024 (<c>LanceConfig.NumTimesteps</c>/<c>CfgTextScale</c>).</summary>
    public static ImageDefaults FamilyDefaults { get; } = new ImageDefaults { Steps = 30, CfgScale = 4.0f, Width = 1024, Height = 1024 };

    /// <inheritdoc/>
    public ImageDefaults Defaults => FamilyDefaults;

    /// <inheritdoc/>
    public IRecipePipeline Construct(RecipeContext context)
    {
        // TODO(E-IMG-4): honor a user-picked VAE override from ImageRequest.Components (the SwarmUI loader read
        // input.Get(T2IParamTypes.VAE)); this always takes the canonical Wan2.2 VAE.
        string checkpointFolder = ResolveLanceFolder(context.CheckpointPath)
            ?? throw new DirectoryNotFoundException($"Lance checkpoint folder not found: {context.CheckpointPath}");
        string vaePath = ModelDownloader.EnsureSideModelAsync(SideModels.Wan22Vae, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();

        // 1. Transformer (sharded-folder aware converter). The ViT keys are dropped — the understanding path is unused.
        (LanceCheckpointConverter.ConvertedWeights conv, IReadOnlyList<SafeTensorsLoader> loaders) = LanceCheckpointConverter.LoadAndConvert(checkpointFolder);
        List<SafeTensorsLoader> owned = new List<SafeTensorsLoader>(loaders);
        try
        {
            if (conv.Transformer.Count == 0)
            {
                throw new InvalidOperationException($"Lance checkpoint '{checkpointFolder}' has no language_model transformer weights.");
            }
            Logs.Info($"[LanceImageRecipe] Converted {conv.Transformer.Count} transformer keys ({conv.Vit.Count} ViT keys ignored).");
            LanceConfig config = LanceConfig.Image;
            LanceTransformer transformer = new LanceTransformer(config);
            transformer.LoadWeights(conv.Transformer);

            // 2. Wan2.2 VAE decoder — computes in F32 (the Wan VAE resnets overflow at F16).
            (Dictionary<string, Tensor> vaeWeightsRaw, IReadOnlyList<SafeTensorsLoader> vaeLoaders) = LanceCheckpointConverter.LoadVae(vaePath);
            owned.AddRange(vaeLoaders);
            Dictionary<string, Tensor> vaeWeights = VaePrecisionHelper.CastVaeWeights(vaeWeightsRaw, DType.F32);
            Wan22VaeDecoder vae = new Wan22VaeDecoder();
            vae.LoadWeights(vaeWeights);
            // Encoder half from the same staged dict (keys encoder.* + top-level conv1.*); null on a decode-only file.
            Wan22VaeEncoder? vaeEncoder = null;
            if (vaeWeights.ContainsKey("encoder.conv1.weight"))
            {
                vaeEncoder = new Wan22VaeEncoder();
                vaeEncoder.LoadWeights(vaeWeights);
            }
            else
            {
                Logs.Info("[LanceImageRecipe] Wan2.2 VAE is decode-only — img2img will be refused for this checkpoint.");
            }

            // 3. Byte-level BPE straight out of the checkpoint's tokenizer.json (exact pre-tokenizer Split regex); the
            //    two-file Qwen2Tokenizer path mis-splits space+punct sequences and is a fallback only.
            ILlmTokenizer tokenizer;
            string tokenizerJsonPath = Path.Combine(checkpointFolder, "tokenizer.json");
            if (File.Exists(tokenizerJsonPath))
            {
                using FileStream tokFs = File.OpenRead(tokenizerJsonPath);
                tokenizer = HfTokenizerJson.LoadByteLevelBpe(tokFs);
            }
            else
            {
                Logs.Warning("[LanceImageRecipe] Checkpoint folder has no tokenizer.json — falling back to the embedded Qwen2 tokenizer (ids may differ around punctuation).");
                tokenizer = new Qwen2Tokenizer();
            }

            LancePromptTemplate template = LancePromptTemplate.Create(tokenizer.EncodeOrdinary, config, video: false);
            LanceImagePipeline pipeline = new LanceImagePipeline(context.Backend, transformer, vae, vaeEncoder, config, template);
            Logs.Info("[LanceImageRecipe] Lance ready (text-to-image).");
            return new LanceImageRecipePipeline(pipeline, config, transformer, tokenizer, owned);
        }
        catch (Exception ex)
        {
            Logs.Error("[LanceImageRecipe] Construction failed.", ex);
            foreach (SafeTensorsLoader loader in owned)
            {
                loader.Dispose();
            }
            throw;
        }
    }

    /// <summary>Maps a checkpoint path (the .safetensors file or the checkpoint folder itself) to the Lance folder, or null when neither exists.</summary>
    private static string? ResolveLanceFolder(string rawPath)
    {
        if (string.IsNullOrWhiteSpace(rawPath))
        {
            return null;
        }
        if (Directory.Exists(rawPath))
        {
            return rawPath;
        }
        if (File.Exists(rawPath))
        {
            return Path.GetDirectoryName(rawPath)?.Replace('\\', '/');
        }
        return null;
    }
}
