using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.CheckpointConverters.Utils;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;
using HartsyInference.Video.Pipelines;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>Wan-Animate recipe (character animation) — the Wan backbone plus a driving-pose latent
/// (<c>pose_patch_embedding</c>) and a face pathway (<c>motion_encoder</c> → <c>face_encoder</c> →
/// <c>face_adapter</c>). Config is weight-derived via <see cref="WanConfigDetector"/> (the layout the engine's Animate
/// harness validated). Lifted from the SwarmUI backend's <c>WanAnimateLoader</c>: umT5-XXL
/// (<see cref="SideModels.Umt5Xxl"/>), the z=16 Wan2.1 VAE (<see cref="SideModels.Wan21Vae"/>), and CLIP-ViT-H
/// (<see cref="SideModels.ClipVisionH14"/>) when the checkpoint ships the i2v image embedder.</summary>
public sealed class WanAnimateRecipe : IVideoRecipe
{
    /// <inheritdoc/>
    public string Name => "wan-animate";

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "wan-animate", StringComparison.OrdinalIgnoreCase);

    /// <inheritdoc/>
    public IVideoRecipePipeline Construct(RecipeContext context)
    {
        // TODO(E-IMG-4/5): LoRA and VideoRequest.Components overrides for the umT5 / VAE / CLIP-Vision picks are deferred.
        string umt5Path = ModelDownloader.EnsureSideModelAsync(SideModels.Umt5Xxl, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
        string vaePath = ModelDownloader.EnsureSideModelAsync(SideModels.Wan21Vae, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();

        (WanVideoCheckpointConverter.ConvertedWeights conv, SafeTensorsLoader ditLoader) = WanVideoCheckpointConverter.LoadAndConvert(context.CheckpointPath);
        List<SafeTensorsLoader> loaders = new List<SafeTensorsLoader> { ditLoader };
        try
        {
            if (!conv.Transformer.ContainsKey("pose_patch_embedding.weight"))
            {
                throw new InvalidOperationException(
                    $"'{context.CheckpointPath}' is routed as Wan-Animate but has no 'pose_patch_embedding' weights after conversion — "
                    + "it may be a plain Wan checkpoint, or a layout the converter doesn't map yet.");
            }
            WanVideoConfig config = WanConfigDetector.Detect(conv.Transformer);
            bool hasClipEmbedder = conv.Transformer.ContainsKey("condition_embedder.image_embedder.norm1.weight");
            Logs.Info($"[WanAnimateRecipe] Converted {conv.Transformer.Count} transformer keys (Animate: pose + face/motion pathway, inner {config.InnerDim}{(hasClipEmbedder ? ", i2v CLIP" : "")}).");

            WanAnimateTransformer transformer = new WanAnimateTransformer(config);
            transformer.LoadWeights(conv.Transformer);

            (Dictionary<string, Tensor> vaeWeightsRaw, IReadOnlyList<SafeTensorsLoader> vaeLoaders) = LanceCheckpointConverter.LoadVae(vaePath);
            loaders.AddRange(vaeLoaders);
            Dictionary<string, Tensor> vaeWeights = VideoRecipeUtils.CastWeights(vaeWeightsRaw, DType.F32);
            Wan21VaeDecoder vaeDecoder = new Wan21VaeDecoder();
            vaeDecoder.LoadWeights(vaeWeights);
            Wan21VaeEncoder vaeEncoder = new Wan21VaeEncoder();
            vaeEncoder.LoadWeights(vaeWeights);

            ClipVisionEncoder? clipVision = null;
            if (hasClipEmbedder)
            {
                string clipPath = ModelDownloader.EnsureSideModelAsync(SideModels.ClipVisionH14, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
                SafeTensorsLoader clipLoader = new SafeTensorsLoader();
                clipLoader.Load(clipPath);
                loaders.Add(clipLoader);
                clipVision = new ClipVisionEncoder(ClipVisionEncoderConfig.ViTH14);
                clipVision.LoadWeights(clipLoader.GetAllTensors());
            }

            SafeTensorsLoader umt5Loader = new SafeTensorsLoader();
            umt5Loader.Load(umt5Path);
            loaders.Add(umt5Loader);
            Dictionary<string, Tensor> umt5Weights = CheckpointConvertUtils.ApplyFp8ScaledDequant(umt5Loader.GetAllTensors());
            T5TextEncoder umt5 = new T5TextEncoder(T5TextEncoderConfig.Umt5Xxl);
            umt5.LoadWeights(umt5Weights);
            T5Tokenizer tokenizer = T5Tokenizer.CreateUmt5(maxLength: WanVideoRecipe.TokenLength);

            WanAnimatePipeline pipeline = new WanAnimatePipeline(context.Backend, transformer, vaeDecoder, vaeEncoder, config);
            Logs.Info("[WanAnimateRecipe] Wan-Animate ready (driving-video).");
            return new WanAnimateRecipePipeline(context.Backend, pipeline, config, tokenizer, umt5, transformer, vaeEncoder, clipVision, loaders);
        }
        catch (Exception ex)
        {
            Logs.Error("[WanAnimateRecipe] Construction failed.", ex);
            foreach (SafeTensorsLoader loader in loaders)
            {
                loader.Dispose();
            }
            throw;
        }
    }
}
