using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Video.Pipelines;

using HartsyInference.Engine.Features;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>Lance (video) recipe — the ByteDance Lance_3B_Video unified multimodal DiT, which ships as a FOLDER
/// checkpoint (model.safetensors or shards + llm_config.json + the Qwen2 tokenizer files). The Wan2.2 48-channel VAE
/// (<see cref="SideModels.Wan22Vae"/>) resolves as a side model. Lifted from the SwarmUI backend's
/// <c>LanceLoader</c> (video half; the image half is <c>LanceImageRecipe</c>) and driven through
/// <see cref="LanceVideoRecipePipeline"/>.</summary>
public sealed class LanceVideoRecipe : IVideoRecipe
{
    /// <inheritdoc/>
    public string Name => "lance-video";

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "lance-video", StringComparison.OrdinalIgnoreCase);

    /// <summary>Lance Video's official sampling settings: 30 steps at text-CFG 4.0, 512x512, 25 frames
    /// (<c>LanceConfig.NumTimesteps</c>/<c>CfgTextScale</c>; <c>LanceVideoGenerationTests</c> verified 512x512
    /// coherent, using a shorter 9-frame clip only to keep the smoke test fast).</summary>
    public VideoDefaults Defaults { get; } = new VideoDefaults { Steps = 30, CfgScale = 4.0f, Width = 512, Height = 512, Frames = 25 };

    /// <inheritdoc/>
    /// <remarks><see cref="VideoFeatures.Lora"/> added 2026-08-20 — the first conditioning this family declares at all (it previously inherited <see cref="IVideoRecipe"/>'s <c>None</c>). Image-to-video is tracked separately: the Wan 2.2 VAE this family loads already has a verified encoder half, so it is wiring rather than a port.</remarks>
    public VideoFeatures Supports => VideoFeatures.Lora;

    /// <inheritdoc/>
    public IVideoRecipePipeline Construct(RecipeContext context)
    {
        // TODO(E-IMG-4/5): a user-picked VAE override from VideoRequest.Components (the SwarmUI loader read
        // input.Get(T2IParamTypes.VAE)) is deferred; this always takes the canonical Wan2.2 VAE.
        string checkpointFolder = ResolveLanceFolder(context.CheckpointPath)
            ?? throw new DirectoryNotFoundException($"Lance checkpoint folder not found: {context.CheckpointPath}");
        string vaePath = ModelDownloader.EnsureSideModelAsync(SideModels.Wan22Vae, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();

        (LanceCheckpointConverter.ConvertedWeights conv, IReadOnlyList<SafeTensorsLoader> loaders) = LanceCheckpointConverter.LoadAndConvert(checkpointFolder);
        List<SafeTensorsLoader> owned = new List<SafeTensorsLoader>(loaders);
        try
        {
            if (conv.Transformer.Count == 0)
            {
                throw new InvalidOperationException($"Lance checkpoint '{checkpointFolder}' has no language_model transformer weights.");
            }
            Logs.Info($"[LanceVideoRecipe] Converted {conv.Transformer.Count} transformer keys ({conv.Vit.Count} ViT keys ignored).");
            LanceConfig config = LanceConfig.Video;
            LanceTransformer transformer = new LanceTransformer(config);
            // Merge any requested LoRAs BEFORE LoadWeights — device caches are identity-keyed, so merging
            // after would leave layers serving the pre-merge tensors (the Sd3Recipe ordering rule).
            MergedLoraStack? loraStack = LoraApplier.BuildAndApply(
                LoraResolver.Resolve(context.Loras), context.Backend, transformerWeights: conv.Transformer);
            transformer.LoadWeights(conv.Transformer);

            // Wan2.2 VAE decoder — computes in F32 (the Wan VAE resnets overflow at F16).
            (Dictionary<string, Tensor> vaeWeightsRaw, IReadOnlyList<SafeTensorsLoader> vaeLoaders) = LanceCheckpointConverter.LoadVae(vaePath);
            owned.AddRange(vaeLoaders);
            Wan22VaeDecoder vae = new Wan22VaeDecoder();
            vae.LoadWeights(VaePrecisionHelper.CastVaeWeights(vaeWeightsRaw, DType.F32));

            // Byte-level BPE straight out of the checkpoint's tokenizer.json (exact pre-tokenizer Split regex); the
            // two-file Qwen2Tokenizer path mis-splits space+punct sequences and is a fallback only.
            ILlmTokenizer tokenizer;
            string tokenizerJsonPath = Path.Combine(checkpointFolder, "tokenizer.json");
            if (File.Exists(tokenizerJsonPath))
            {
                using FileStream tokFs = File.OpenRead(tokenizerJsonPath);
                tokenizer = HfTokenizerJson.LoadByteLevelBpe(tokFs);
            }
            else
            {
                Logs.Warning("[LanceVideoRecipe] Checkpoint folder has no tokenizer.json — falling back to the embedded Qwen2 tokenizer (ids may differ around punctuation).");
                tokenizer = new Qwen2Tokenizer();
            }

            LancePromptTemplate template = LancePromptTemplate.Create(tokenizer.EncodeOrdinary, config, video: true);
            LanceVideoPipeline pipeline = new LanceVideoPipeline(context.Backend, transformer, vae, config, template);
            Logs.Info("[LanceVideoRecipe] Lance ready (text-to-video).");
            return new LanceVideoRecipePipeline(pipeline, config, transformer, tokenizer, owned, loraStack);
        }
        catch (Exception ex)
        {
            Logs.Error("[LanceVideoRecipe] Construction failed.", ex);
            foreach (SafeTensorsLoader loader in owned)
            {
                loader.Dispose();
            }
            throw;
        }
    }

    /// <summary>Maps a checkpoint path (the .safetensors file or the folder itself) to the Lance folder, or null when neither exists.</summary>
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
