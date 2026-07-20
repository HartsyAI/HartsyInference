using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.CheckpointConverters.Utils;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;
using HartsyInference.Video.Pipelines;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>Wan2.2-S2V recipe (speech-to-video) — the Wan backbone plus an audio injector
/// (<c>audio_injector.*</c>) and a causal audio encoder (<c>casual_audio_encoder.*</c>) over stacked Wav2Vec2
/// features. Config comes from <see cref="WanConfigDetector"/> (the parity-proven S2V layout, including the
/// non-uniform audio-inject block indices). Lifted from the SwarmUI backend's <c>WanS2VLoader</c>: umT5-XXL
/// (<see cref="SideModels.Umt5Xxl"/>), the z=16 Wan2.1 VAE (<see cref="SideModels.Wan21Vae"/>), and the Wav2Vec2
/// front-end (<see cref="SideModels.Wav2Vec2Large"/> / <see cref="SideModels.Wav2Vec2Base"/>, picked from the
/// weight-derived audio feature dim).</summary>
public sealed class WanS2VRecipe : IVideoRecipe
{
    /// <inheritdoc/>
    public string Name => "wan-s2v";

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "wan-s2v", StringComparison.OrdinalIgnoreCase);

    /// <inheritdoc/>
    public IVideoRecipePipeline Construct(RecipeContext context)
    {
        // TODO(E-IMG-4/5): LoRA and VideoRequest.Components overrides for the umT5 / VAE picks are deferred.
        string umt5Path = ModelDownloader.EnsureSideModelAsync(SideModels.Umt5Xxl, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
        string vaePath = ModelDownloader.EnsureSideModelAsync(SideModels.Wan21Vae, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();

        (WanVideoCheckpointConverter.ConvertedWeights conv, SafeTensorsLoader ditLoader) = WanVideoCheckpointConverter.LoadAndConvert(context.CheckpointPath);
        List<SafeTensorsLoader> loaders = new List<SafeTensorsLoader> { ditLoader };
        try
        {
            Dictionary<string, Tensor> weights = conv.Transformer;
            WanVideoConfig config = WanConfigDetector.Detect(weights);
            if (!config.HasAudioConditioning)
            {
                throw new InvalidOperationException(
                    $"'{context.CheckpointPath}' is routed as Wan S2V but the converted checkpoint has no "
                    + "'casual_audio_encoder.*'/'audio_injector.*' weights — it may be a plain Wan checkpoint, or a layout the converter doesn't map yet.");
            }
            int audioDim = config.AudioDim;
            Logs.Info($"[WanS2VRecipe] Converted {weights.Count} keys (S2V audio injector x{config.AudioInjectLayers.Length}, audioDim {audioDim}, "
                + $"{config.AudioLayers} harvested layers, {config.AudioTokens} tok/frame, cfg {config.GuidanceScale}).");

            WanS2VTransformer transformer = new WanS2VTransformer(config);
            transformer.LoadWeights(weights);
            WanS2VAudioEncoder audioEncoder = new WanS2VAudioEncoder(config.AudioLayers, config.AudioDim, config.InnerDim, config.AudioTokens);
            audioEncoder.LoadWeights(weights);

            (Dictionary<string, Tensor> vaeWeightsRaw, IReadOnlyList<SafeTensorsLoader> vaeLoaders) = LanceCheckpointConverter.LoadVae(vaePath);
            loaders.AddRange(vaeLoaders);
            Dictionary<string, Tensor> vaeWeights = VideoRecipeUtils.CastWeights(vaeWeightsRaw, DType.F32);
            Wan21VaeDecoder vaeDecoder = new Wan21VaeDecoder();
            vaeDecoder.LoadWeights(vaeWeights);
            Wan21VaeEncoder vaeEncoder = new Wan21VaeEncoder();
            vaeEncoder.LoadWeights(vaeWeights);

            Wav2Vec2EncoderConfig w2vConfig = audioDim >= 1024 ? Wav2Vec2EncoderConfig.Large : Wav2Vec2EncoderConfig.Base;
            ModelAsset w2vAsset = audioDim >= 1024 ? SideModels.Wav2Vec2Large : SideModels.Wav2Vec2Base;
            string w2vPath = ModelDownloader.EnsureSideModelAsync(w2vAsset, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
            SafeTensorsLoader w2vLoader = new SafeTensorsLoader();
            w2vLoader.Load(w2vPath);
            loaders.Add(w2vLoader);
            Wav2Vec2Encoder wav2vec2 = new Wav2Vec2Encoder(w2vConfig);
            // The Comfy-Org audio-encoder file prefixes every key with "wav2vec2." (plus an unused lm_head).
            Dictionary<string, Tensor> wavWeights = new Dictionary<string, Tensor>();
            foreach (KeyValuePair<string, Tensor> kv in w2vLoader.GetAllTensors())
            {
                if (kv.Key.StartsWith("wav2vec2.", StringComparison.Ordinal))
                {
                    wavWeights[kv.Key["wav2vec2.".Length..]] = kv.Value;
                }
            }
            if (wavWeights.Count == 0)
            {
                wavWeights = new Dictionary<string, Tensor>(w2vLoader.GetAllTensors());
            }
            wav2vec2.LoadWeights(wavWeights);

            SafeTensorsLoader umt5Loader = new SafeTensorsLoader();
            umt5Loader.Load(umt5Path);
            loaders.Add(umt5Loader);
            Dictionary<string, Tensor> umt5Weights = CheckpointConvertUtils.ApplyFp8ScaledDequant(umt5Loader.GetAllTensors());
            T5TextEncoder umt5 = new T5TextEncoder(T5TextEncoderConfig.Umt5Xxl);
            umt5.LoadWeights(umt5Weights);
            T5Tokenizer tokenizer = T5Tokenizer.CreateUmt5(maxLength: WanVideoRecipe.TokenLength);

            WanS2VPipeline pipeline = new WanS2VPipeline(context.Backend, transformer, audioEncoder, vaeDecoder, config, encoder: vaeEncoder);
            Logs.Info("[WanS2VRecipe] Wan S2V ready (audio+text, reference-identity capable).");
            return new WanS2VRecipePipeline(context.Backend, pipeline, config, tokenizer, umt5, transformer, audioEncoder, wav2vec2, vaeEncoder, loaders);
        }
        catch (Exception ex)
        {
            Logs.Error("[WanS2VRecipe] Construction failed.", ex);
            foreach (SafeTensorsLoader loader in loaders)
            {
                loader.Dispose();
            }
            throw;
        }
    }
}
