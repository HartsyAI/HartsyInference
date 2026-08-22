using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Video.Pipelines;
using HartsyInference.Vision.Clip;

using HartsyInference.Engine.Features;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>Wan-Animate-2 recipe — the Wan2.1 I2V-14B backbone driven by a raw video through a second token stream
/// rather than by a pose/face pathway. The checkpoint adds no module of its own, so it is key-for-key an I2V-14B one
/// and is recognised only by <c>__metadata__["config"].transformer.model_type == "animate2"</c>; see
/// <see cref="WanVideoRecipe.DetectVariant"/>. Side models are the family's usual set: umT5-XXL
/// (<see cref="SideModels.Umt5Xxl"/>), the z=16 Wan2.1 VAE (<see cref="SideModels.Wan21Vae"/>) and CLIP-ViT-H
/// (<see cref="SideModels.ClipVisionH14"/>), which is <b>required</b> here — both token streams carry a 257-token
/// image context.</summary>
public sealed class WanAnimate2Recipe : IVideoRecipe
{
    /// <inheritdoc/>
    public string Name => "wan-animate-2";

    /// <inheritdoc/>
    /// <remarks>A driving video and a reference character image, and nothing else. Unlike V1 there is no pose, face,
    /// background or mask branch to override — <see cref="WanAnimate2RecipePipeline"/> refuses those clips by name
    /// rather than accepting and dropping them.</remarks>
    public VideoFeatures Supports => VideoFeatures.InitImage | VideoFeatures.DrivingVideo | VideoFeatures.Lora;

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "wan-animate-2", StringComparison.OrdinalIgnoreCase);

    /// <summary>The base build's operative settings (README / demo / gradio): 40 steps at guidance 3.0 over an
    /// 81-frame clip at 24 fps. The distillation build wants 10 steps at guidance 1.0 instead; only its
    /// <c>log_scale</c> is routed automatically (<see cref="WanAnimate2Transformer.ResolveLogScale"/>) — the step
    /// count and guidance stay the caller's, because a filename is too weak a signal to override them on.</summary>
    public VideoDefaults Defaults { get; } = new VideoDefaults { Steps = 40, CfgScale = 3.0f, Frames = 81, Fps = 24 };

    /// <inheritdoc/>
    public IVideoRecipePipeline Construct(RecipeContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        PlacementSupport.WarnIfIgnored("WanAnimate2Recipe", context, blockStreaming: true);
        string umt5Path = ModelDownloader.EnsureSideModelAsync(SideModels.Umt5Xxl, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
        string vaePath = ModelDownloader.EnsureSideModelAsync(SideModels.Wan21Vae, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();

        (WanVideoCheckpointConverter.ConvertedWeights conv, SafeTensorsLoader ditLoader) = WanVideoCheckpointConverter.LoadAndConvert(context.CheckpointPath);
        List<SafeTensorsLoader> loaders = new List<SafeTensorsLoader> { ditLoader };
        ModelAssets.Lora.LoraStack? loraStack = null;
        try
        {
            if (!conv.IsAnimate2)
            {
                throw new InvalidOperationException(
                    $"'{context.CheckpointPath}' is routed as Wan-Animate-2 but its metadata does not declare "
                    + "model_type 'animate2' — the weights alone cannot distinguish it from a plain Wan2.1 I2V-14B checkpoint.");
            }
            if (!conv.Transformer.ContainsKey("condition_embedder.image_embedder.norm1.weight"))
            {
                throw new InvalidOperationException(
                    $"'{context.CheckpointPath}' has no i2v CLIP image embedder after conversion — Wan-Animate-2 prepends a "
                    + "257-token image context to BOTH streams and cannot run without it.");
            }
            // log_scale is a config fact the weights cannot carry: the base and distillation builds are key-for-key
            // identical and both declare only model_type 'animate2'. Without this the distillation build silently
            // ran the base build's 0 and lost the score bias upstream trains it with.
            WanVideoConfig config = WanConfigDetector.Detect(conv.Transformer, isAnimate2: true) with
            {
                Animate2LogScale = WanAnimate2Transformer.ResolveLogScale(context.CheckpointPath),
            };
            if (config.InChannels != 36)
            {
                throw new InvalidOperationException(
                    $"Wan-Animate-2 needs a 36-channel patch embedding (latent 16 | mask 4 | cond 16); '{context.CheckpointPath}' has {config.InChannels}.");
            }
            Logs.Info($"[WanAnimate2Recipe] Converted {conv.Transformer.Count} transformer keys "
                + $"({config.NumLayers} blocks, inner {config.InnerDim}, log_scale {config.Animate2LogScale}).");

            // Merge BEFORE LoadWeights (identity-keyed device caches — same ordering as WanVideoRecipe).
            loraStack = Features.LoraApplier.BuildAndApply(
                Features.LoraResolver.Resolve(context.Loras),
                context.Backend,
                transformerWeights: conv.Transformer);

            WanAnimate2Transformer transformer = new WanAnimate2Transformer(config);
            transformer.LoadWeights(conv.Transformer);

            (Dictionary<string, Tensor> vaeWeightsRaw, IReadOnlyList<SafeTensorsLoader> vaeLoaders) = LanceCheckpointConverter.LoadVae(vaePath);
            loaders.AddRange(vaeLoaders);
            Dictionary<string, Tensor> vaeWeights = VaePrecisionHelper.CastVaeWeights(vaeWeightsRaw, DType.F32);
            Wan21VaeDecoder vaeDecoder = new Wan21VaeDecoder();
            vaeDecoder.LoadWeights(vaeWeights);
            Wan21VaeEncoder vaeEncoder = new Wan21VaeEncoder();
            vaeEncoder.LoadWeights(vaeWeights);

            string clipPath = ModelDownloader.EnsureSideModelAsync(SideModels.ClipVisionH14, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
            SafeTensorsLoader clipLoader = new SafeTensorsLoader();
            clipLoader.Load(clipPath);
            loaders.Add(clipLoader);
            ClipVisionEncoder clipVision = new ClipVisionEncoder(ClipVisionEncoderConfig.ViTH14);
            clipVision.LoadWeights(clipLoader.GetAllTensors());

            SafeTensorsLoader umt5Loader = new SafeTensorsLoader();
            umt5Loader.Load(umt5Path);
            loaders.Add(umt5Loader);
            Dictionary<string, Tensor> umt5Weights = CheckpointConvertUtils.ApplyFp8ScaledDequant(umt5Loader.GetAllTensors());
            T5TextEncoder umt5 = new T5TextEncoder(T5TextEncoderConfig.Umt5Xxl);
            umt5.LoadWeights(umt5Weights);
            T5Tokenizer tokenizer = T5Tokenizer.CreateUmt5(maxLength: WanVideoRecipe.TokenLength);

            WanAnimate2Pipeline pipeline = new WanAnimate2Pipeline(context.Backend, transformer, vaeDecoder, vaeEncoder, config);
            Logs.Info("[WanAnimate2Recipe] Wan-Animate-2 ready (raw driving video, no pose/face preprocessing).");
            return new WanAnimate2RecipePipeline(context.Backend, pipeline, config, tokenizer, umt5, transformer, vaeEncoder, clipVision, loaders, loraStack);
        }
        catch (Exception ex)
        {
            Logs.Error("[WanAnimate2Recipe] Construction failed.", ex);
            foreach (SafeTensorsLoader loader in loaders)
            {
                loader.Dispose();
            }
            loraStack?.Dispose();
            throw;
        }
    }
}
