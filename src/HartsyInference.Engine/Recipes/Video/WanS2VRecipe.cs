using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Video.Pipelines;

using HartsyInference.Engine.Features;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>Wan2.2-S2V recipe (speech-to-video) — the Wan backbone plus an audio injector (<c>audio_injector.*</c>) and a causal audio encoder (<c>casual_audio_encoder.*</c>) over stacked Wav2Vec2 features. Config comes from <see cref="WanConfigDetector"/> (the parity-proven S2V layout, including the non-uniform audio-inject block indices). Lifted from the SwarmUI backend's <c>WanS2VLoader</c>: umT5-XXL (<see cref="SideModels.Umt5Xxl"/>), the z=16 Wan2.1 VAE (<see cref="SideModels.Wan21Vae"/>), and the Wav2Vec2 front-end (<see cref="SideModels.Wav2Vec2Large"/> / <see cref="SideModels.Wav2Vec2Base"/>, picked from the weight-derived audio feature dim).</summary>
public sealed class WanS2VRecipe : IVideoRecipe
{
    /// <inheritdoc/>
    public string Name => "wan-s2v";


    /// <inheritdoc/>
    /// <remarks>Wan-S2V turns the init image into appended identity reference tokens.</remarks>
    /// <remarks><see cref="VideoFeatures.Lora"/> added 2026-08-20, matching the plain Wan family. The merge runs against the shared <c>weights</c> dict before BOTH the transformer and the audio encoder load from it, so an S2V LoRA touching either lands.</remarks>
    /// <remarks><see cref="VideoFeatures.DrivingAudio"/> added 2026-09-17, with the bit itself. S2V's driving speech
    /// arrives in <c>VideoRequest.VideoAudioReference</c> — the field that now classifies as this feature — and
    /// <c>WanS2VRecipePipeline.Generate</c> throws without it, so leaving it undeclared had the generic planner
    /// refuse the family's own mandatory input before construction.</remarks>
    /// <inheritdoc/>
    /// <remarks>Ledger evidence in <c>PromptWeightingModeLedgerTests</c>: every Wan variant class
    /// resolves to <c>wan.WanT5Tokenizer</c> → <c>UMT5XXlTokenizer</c>, which does NOT disable weights,
    /// so the mechanism is the encoder-output blend rather than a cond scale. Wan S2V is a separate
    /// recipe class from <c>WanVideoRecipe</c> with its own prompt path, which is why it needed its own
    /// wiring rather than inheriting one. Its audio conditioning is untouched — only the text stream weights.</remarks>
    public Diffusion.Prompting.PromptWeightingMode PromptWeighting =>
        Diffusion.Prompting.PromptWeightingMode.ComfyBlend;

    public VideoFeatures Supports => VideoFeatures.InitImage | VideoFeatures.Lora | VideoFeatures.DrivingAudio;
    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "wan-s2v", StringComparison.OrdinalIgnoreCase);

    /// <summary>Wan S2V's official sampling settings: 50 steps at guidance 5.0 (<c>WanVideoConfig.NumInferenceSteps</c>/<c>GuidanceScale</c>).</summary>
    public VideoDefaults Defaults { get; } = new VideoDefaults { Steps = 50, CfgScale = 5.0f };

    /// <inheritdoc/>
    public IVideoRecipePipeline Construct(RecipeContext context)
    {
        // TODO(E-IMG-4/5): LoRA and VideoRequest.Components overrides for the umT5 / VAE picks are deferred.
        string umt5Path = ModelDownloader.EnsureSideModelAsync(SideModels.Umt5Xxl, onProgress: null, context.Cancel).GetAwaiter().GetResult();
        string vaePath = ModelDownloader.EnsureSideModelAsync(SideModels.Wan21Vae, onProgress: null, context.Cancel).GetAwaiter().GetResult();

        // Side-model loaders and the checkpoint share one bag: the container is format-agnostic, so what it hands
        // back is an IDisposable rather than a SafeTensorsLoader.
        List<IDisposable> loaders = new List<IDisposable>();
        MergedLoraStack? loraStack = null;
        try
        {
            // One container for either format: a Wan GGUF is a repack of this same file and keeps its tensor
            // names, so nothing below needs to know which one arrived.
            CheckpointSource source = CheckpointSource.Open(context.CheckpointPath);
            loaders.Add(source);
            WanVideoCheckpointConverter.ConvertedWeights conv =
                WanVideoCheckpointConverter.Convert(source.Weights, source.Header.Metadata);
            // Any quant this backend has no packed-weight kernel for widens here rather than failing inside the
            // first GEMM, minutes into a generation. Tracked immediately so a failure further down frees the
            // widened copies rather than leaving them to the finalizer.
            loaders.Add(QuantizedWeightPolicy.PrepareForBackends(conv.Transformer, context.TransformerBackends));
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
            // Merge any requested LoRAs BEFORE LoadWeights — device caches are identity-keyed, so merging
            // after would leave layers serving the pre-merge tensors (the Sd3Recipe ordering rule).
            loraStack = RecipeLoraMerge.Apply(context, new LoraMergeTargets { Transformer = weights }, "WanS2VRecipe");
            transformer.LoadWeights(weights);
            WanS2VAudioEncoder audioEncoder = new WanS2VAudioEncoder(config.AudioLayers, config.AudioDim, config.InnerDim, config.AudioTokens);
            audioEncoder.LoadWeights(weights);

            (IWanVaeDecoder vaeDecoder, IWanVaeEncoder vaeEncoder) = VideoRecipeUtils.LoadWanVae(vaePath, isWan21: true, loaders);

            Wav2Vec2EncoderConfig w2vConfig = audioDim >= 1024 ? Wav2Vec2EncoderConfig.Large : Wav2Vec2EncoderConfig.Base;
            ModelAsset w2vAsset = audioDim >= 1024 ? SideModels.Wav2Vec2Large : SideModels.Wav2Vec2Base;
            string w2vPath = ModelDownloader.EnsureSideModelAsync(w2vAsset, onProgress: null, context.Cancel).GetAwaiter().GetResult();
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

            (T5TextEncoder umt5, T5Tokenizer tokenizer) = VideoRecipeUtils.LoadUmt5(umt5Path, loaders);

            WanS2VPipeline pipeline = new WanS2VPipeline(context.Backend, transformer, audioEncoder, vaeDecoder, config, encoder: vaeEncoder);
            Logs.Info("[WanS2VRecipe] Wan S2V ready (audio+text, reference-identity capable).");
            return new WanS2VRecipePipeline(context.Backend, pipeline, config, tokenizer, umt5, transformer, audioEncoder, wav2vec2, vaeEncoder, loaders, loraStack);
        }
        catch (Exception ex)
        {
            Logs.Error("[WanS2VRecipe] Construction failed.", ex);
            foreach (IDisposable loader in loaders)
            {
                loader.Dispose();
            }
            loraStack?.Dispose();
            throw;
        }
    }
}
