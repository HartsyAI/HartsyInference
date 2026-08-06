using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Video.Pipelines;

using HartsyInference.Engine.Features;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>Kandinsky 5.0 T2V-Lite recipe (kandinskylab, ~2B video DiT): the checkpoint is the diffusers
/// <c>Kandinsky-5.0-T2V-Lite-*-Diffusers</c> folder — a <c>transformer/</c> shard directory plus a bundled
/// <c>vae/</c> shard that is the shared <see cref="HunyuanVideoVaeDecoder"/> in diffusers naming (per
/// <c>Kandinsky5VideoGenerationTests</c>) — while the dual text stack (Qwen2.5-VL-7B sequence embeddings via
/// <see cref="SideModels.Qwen2_5_VL_7B"/> + CLIP-L pooled via <see cref="SideModels.ClipL"/>) resolves as side
/// models, reusing the exact live-encode path <see cref="Image.Kandinsky5Recipe"/> verified for T2I
/// (<see cref="Kandinsky5TextEncoding"/>). Unlike the T2I checkpoint test's pre-computed embeddings, this recipe
/// encodes the request's own prompt live.</summary>
public sealed class Kandinsky5VideoRecipe : IVideoRecipe
{
    /// <inheritdoc/>
    public string Name => "kandinsky5-video";


    /// <inheritdoc/>
    /// <remarks>Start-frame image-to-video: the first frame is VAE-encoded and pinned as latent frame 0, with
    /// only frames 1.. denoising. No end-frame conditioning in this family.</remarks>
    public VideoFeatures Supports => VideoFeatures.InitImage;
    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "kandinsky5-video", StringComparison.OrdinalIgnoreCase);

    /// <summary>Kandinsky-5 Video's official sampling settings: 50 steps at guidance 5.0, 512x512, 25 frames @ 24fps
    /// (<c>Kandinsky5VideoPipeline</c> doc comment; <c>Kandinsky5VideoGenerationTests</c> verified 512x512
    /// coherent output at this cadence).</summary>
    public VideoDefaults Defaults { get; } = new VideoDefaults { Steps = 50, CfgScale = 5.0f, Width = 512, Height = 512, Frames = 25, Fps = 24 };

    /// <inheritdoc/>
    public IVideoRecipePipeline Construct(RecipeContext context)
    {
        // TODO(E-IMG-4/5): a VideoRequest.Components Qwen2.5-VL / CLIP-L / VAE override is still deferred.
        string transformerDir = ResolveTransformerDir(context.CheckpointPath);
        string vaeDir = Path.Combine(Path.GetDirectoryName(transformerDir.TrimEnd('/', '\\'))!, "vae");
        List<SafeTensorsLoader> loaders = new List<SafeTensorsLoader>();
        try
        {
            Logs.Info($"[Kandinsky5VideoRecipe] Loading video transformer: {transformerDir}.");
            (Kandinsky5CheckpointConverter.ConvertedWeights converted, List<SafeTensorsLoader> shardLoaders) = Kandinsky5CheckpointConverter.LoadVideoTransformer(transformerDir);
            loaders.AddRange(shardLoaders);
            if (converted.Transformer.Count == 0)
            {
                throw new InvalidOperationException($"Kandinsky 5 Video checkpoint '{transformerDir}' has no transformer weights.");
            }

            // BF16 -> F16 (native F16 GEMM), matching the T2I recipe and Kandinsky5VideoGenerationTests.
            Kandinsky5Config config = Kandinsky5Config.VideoLite2B;
            Dictionary<string, Tensor> transformerWeights = new Dictionary<string, Tensor>(converted.Transformer.Count);
            foreach (KeyValuePair<string, Tensor> kv in converted.Transformer)
            {
                transformerWeights[kv.Key] = kv.Value.DType == DType.BF16 ? kv.Value.CastTo(DType.F16) : kv.Value;
            }
            Kandinsky5Transformer transformer = new Kandinsky5Transformer(config);
            transformer.LoadWeights(transformerWeights);

            Logs.Info($"[Kandinsky5VideoRecipe] Loading HunyuanVideo VAE (diffusers naming): {vaeDir}.");
            (Dictionary<string, Tensor> vaeWeightsRaw, List<SafeTensorsLoader> vaeLoaders) = Kandinsky5CheckpointConverter.LoadHunyuanVideoVae(vaeDir);
            loaders.AddRange(vaeLoaders);
            Dictionary<string, Tensor> vaeWeights = new Dictionary<string, Tensor>(vaeWeightsRaw.Count);
            foreach (KeyValuePair<string, Tensor> kv in vaeWeightsRaw)
            {
                vaeWeights[kv.Key] = kv.Value.DType == DType.BF16 ? kv.Value.CastTo(DType.F16) : kv.Value;
            }
            HunyuanVideoVaeDecoder vae = new HunyuanVideoVaeDecoder();
            vae.LoadWeights(vaeWeights);

            string qwenPath = ModelDownloader.EnsureSideModelAsync(SideModels.Qwen2_5_VL_7B, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
            SafeTensorsLoader qwenLoader = new SafeTensorsLoader();
            qwenLoader.Load(qwenPath);
            loaders.Add(qwenLoader);
            LlamaStyleEncoder qwen = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Qwen2_5_VL_7B);
            qwen.LoadWeights(qwenLoader.GetAllTensors());

            string clipPath = ModelDownloader.EnsureSideModelAsync(SideModels.ClipL, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
            SafeTensorsLoader clipLoader = new SafeTensorsLoader();
            clipLoader.Load(clipPath);
            loaders.Add(clipLoader);
            ClipTextEncoder clipL = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipL);
            clipL.LoadWeights(Kandinsky5TextEncoding.ConvertClipLFromStandalone(clipLoader.GetAllTensors()), prefix: "text_model");

            // Encoder half from the same vae/ shard — this is what turns on the I2V path (EncodeFirstFrame).
            HunyuanVideoVaeEncoder? vaeEncoder = null;
            if (vaeWeights.ContainsKey("encoder.conv_in.conv.weight") || vaeWeights.ContainsKey("encoder.conv_in.weight"))
            {
                vaeEncoder = new HunyuanVideoVaeEncoder();
                vaeEncoder.LoadWeights(vaeWeights);
            }
            else
            {
                Logs.Info("[Kandinsky5VideoRecipe] VAE shard is decode-only — image-to-video will be refused for this checkpoint.");
            }

            Kandinsky5VideoPipeline pipeline = new Kandinsky5VideoPipeline(context.Backend, transformer, vae, config, vaeEncoder);
            Logs.Info("[Kandinsky5VideoRecipe] Kandinsky 5 T2V-Lite ready (Qwen2.5-VL + CLIP-L live encode).");
            return new Kandinsky5VideoRecipePipeline(pipeline, qwen, clipL, new Qwen2Tokenizer(), new ClipTokenizer(), context.Backend, transformer, loaders);
        }
        catch (Exception ex)
        {
            Logs.Error("[Kandinsky5VideoRecipe] Construction failed.", ex);
            foreach (SafeTensorsLoader loader in loaders)
            {
                loader.Dispose();
            }
            throw;
        }
    }

    /// <summary>Maps the resolved checkpoint path to the diffusers <c>transformer/</c> directory
    /// <see cref="Kandinsky5CheckpointConverter.LoadVideoTransformer"/> expects: the catalog's primary asset
    /// resolves to the transformer shard FILE inside that folder, so a file path degrades to its parent
    /// directory; a directory is used as-is (or its own <c>transformer</c> subfolder, when it is the checkpoint
    /// root rather than the transformer folder itself).</summary>
    private static string ResolveTransformerDir(string rawPath)
    {
        if (string.IsNullOrWhiteSpace(rawPath))
        {
            throw new ArgumentException("Kandinsky 5 Video checkpoint path is empty.", nameof(rawPath));
        }
        if (File.Exists(rawPath))
        {
            return Path.GetDirectoryName(rawPath)?.Replace('\\', '/')
                ?? throw new DirectoryNotFoundException($"Kandinsky 5 Video transformer file has no parent directory: {rawPath}");
        }
        if (Directory.Exists(rawPath))
        {
            if (string.Equals(Path.GetFileName(rawPath.TrimEnd('/', '\\')), "transformer", StringComparison.OrdinalIgnoreCase))
            {
                return rawPath;
            }
            string nested = Path.Combine(rawPath, "transformer");
            return Directory.Exists(nested) ? nested : rawPath;
        }
        throw new DirectoryNotFoundException($"Kandinsky 5 Video checkpoint not found: {rawPath}");
    }
}
