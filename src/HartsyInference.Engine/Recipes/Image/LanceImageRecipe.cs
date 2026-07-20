using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Lance (image) recipe — ByteDance's 3B unified multimodal model, which ships as a FOLDER checkpoint (model.safetensors or shards + llm_config.json + the Qwen2 tokenizer files). The Wan2.2 48-channel VAE (<see cref="SideModels.Wan22Vae"/>) resolves as a side model. Lifted from the SwarmUI backend's <c>LanceLoader</c> (image half only; the video variant is a separate family) and driven through <see cref="LanceImageRecipePipeline"/>.</summary>
public sealed class LanceImageRecipe : IArchitectureRecipe
{
    /// <inheritdoc/>
    public string Name => "lance-image";

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "lance-image", StringComparison.OrdinalIgnoreCase);

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
            Dictionary<string, Tensor> vaeWeights = CastWeights(vaeWeightsRaw, DType.F32);
            Wan22VaeDecoder vae = new Wan22VaeDecoder();
            vae.LoadWeights(vaeWeights);

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
            LanceImagePipeline pipeline = new LanceImagePipeline(context.Backend, transformer, vae, config, template);
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

    /// <summary>Casts a VAE weight dictionary to <paramref name="target"/>, leaving tensors already at that dtype untouched (inlined VAE precision policy).</summary>
    private static Dictionary<string, Tensor> CastWeights(Dictionary<string, Tensor> weights, DType target)
    {
        Dictionary<string, Tensor> result = new Dictionary<string, Tensor>(weights.Count);
        foreach (KeyValuePair<string, Tensor> kv in weights)
        {
            result[kv.Key] = kv.Value.DType == target ? kv.Value : kv.Value.CastTo(target);
        }
        return result;
    }
}
