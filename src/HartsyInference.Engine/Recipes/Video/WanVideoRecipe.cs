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
using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;

using HartsyInference.Engine.Features;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>Wan-Video recipe (Wan-AI, umT5-conditioned text/image-to-video) and the entry point for the whole Wan family: the SwarmUI compat classes <c>wan-22-5b</c> / <c>wan-21-1_3b</c> / <c>wan-21-14b</c> are shared by the plain T2V/I2V backbone and the VACE / Animate / S2V conditioning variants, so — exactly like the extension's <c>WanModelVariants.Detect</c> — this recipe sniffs the checkpoint header and hands off to <see cref="WanVaceRecipe"/>, <see cref="WanAnimateRecipe"/>, or <see cref="WanS2VRecipe"/> when it sees their signature weights. Lifted from the SwarmUI backend's <c>WanVideoLoader</c>: umT5-XXL (<see cref="SideModels.Umt5Xxl"/>), the z=48 Wan2.2 VAE (<see cref="SideModels.Wan22Vae"/>) or z=16 Wan2.1 VAE (<see cref="SideModels.Wan21Vae"/>), and CLIP-ViT-H (<see cref="SideModels.ClipVisionH14"/>) for Wan2.1 I2V.</summary>
public sealed class WanVideoRecipe : IVideoRecipe
{
    /// <summary>Wan2.2 TI2V-5B compat class (z=48 VAE).</summary>
    public const string Wan22_5BCompatClassId = "wan-22-5b";

    /// <summary>Wan2.1 1.3B compat class.</summary>
    public const string Wan21_1_3BCompatClassId = "wan-21-1_3b";

    /// <summary>Wan2.1 / Wan2.2 14B compat class (shared by T2V, CLIP-I2V, VACE, Animate and S2V checkpoints).</summary>
    public const string Wan21_14BCompatClassId = "wan-21-14b";

    /// <summary>Wan's umT5 context length (matches diffusers' 512-token encode).</summary>
    internal const int TokenLength = 512;

    /// <summary>Upstream's <c>wan_shared_cfg.sample_neg_prompt</c> (<c>wan/configs/shared_config.py</c>), used by every Wan family when the caller gives no negative prompt.</summary>
    internal const string DefaultNegativePrompt =
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
        + "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走";

    private readonly string _familyId;

    /// <summary>Binds the recipe to one Wan family id. The catalog slug "wan" derives the config from the weights; a compat-class id ("wan-22-5b" / "wan-21-1_3b" / "wan-21-14b") selects the matching preset instead.</summary>
    public WanVideoRecipe(string familyId = "wan") => _familyId = familyId;

    /// <inheritdoc/>
    public string Name => _familyId;


    /// <inheritdoc/>
    /// <remarks><b>Updated 2026-08-11 (Tier 3.3 — real end-frame wiring):</b> <see cref="WanVideoRecipePipeline"/>'s
    /// non-concat path now VAE-encodes <see cref="VideoRequest.VideoEndFrame"/> into a <c>lastFrameLatent</c>
    /// exactly like <see cref="VideoRequest.InitImage"/>'s <c>firstFrameLatent</c> (see
    /// <see cref="WanVideoPipeline.RunDenoise"/>'s symmetric per-frame-timestep-pin mechanism), so
    /// <see cref="VideoFeatures.EndFrame"/> is no longer a lie for <c>wan-22-5b</c> — real-weight verified against
    /// the locally-available TI2V-5B checkpoint (see the extension backlog memory / plan for the verification
    /// artifact). <c>wan-21_1_3b</c> shares the identical non-concat code path (same <see cref="ResolveConfig"/>
    /// branch shape as Ti2V5B) so the mechanism should cover it too, but is left narrowed here deliberately — no
    /// local 1.3B checkpoint exists to actually run and look at, and this backlog's hard rule is real-checkpoint
    /// verification, not "should work by symmetry." Revisit once a 1.3B checkpoint is available to test.
    /// <c>wan-21-14b</c> stays ambiguous at the family level (T2V vs. concat-I2V is a checkpoint property — see
    /// <see cref="SupportsFor"/>) and keeps claiming both; the generic "wan" catalog slug (weight-derived config,
    /// no compat class) does too, for the same reason.</remarks>
    public VideoFeatures Supports =>
        (_familyId is Wan21_1_3BCompatClassId ? VideoFeatures.InitImage
            : VideoFeatures.InitImage | VideoFeatures.EndFrame) | VideoFeatures.Lora;

    /// <summary>The features for a CONCRETE checkpoint: VACE/Animate/S2V share Wan's compat classes and are only detected by sniffing the header, so the family-level <see cref="Supports"/> alone would wrongly refuse (e.g.) a driving video on an Animate checkpoint loaded under <c>wan-21-14b</c>. Falls back to the family answer when the file cannot be peeked.</summary>
    /// <remarks>Does NOT yet narrow the <c>wan-21-14b</c> T2V-vs-concat-I2V ambiguity — that needs the in-channels of <c>patch_embedding.weight</c>, which <see cref="ConstructBase"/> reads off the CONVERTED weight dict (post <see cref="WanVideoCheckpointConverter.LoadAndConvert"/>), not the raw checkpoint's own key names. Wan ships both single-file and diffusers-shard layouts with different raw prefixes, so a cheap raw-header peek here (mirroring <see cref="VideoRecipeUtils.PeekSafeTensorKeys"/>) risks silently misclassifying a checkpoint whose prefix the peek doesn't recognize — worse than the current over-claim, which at least fails loudly as a silent no-op the caller can be told about rather than a wrong refusal. Left for the real end-frame wiring (tracked in the extension's TODO backlog), which needs the converted weights loaded anyway.</remarks>
    public VideoFeatures SupportsFor(string? checkpointPath)
    {
        if (string.IsNullOrWhiteSpace(checkpointPath))
        {
            return Supports;
        }
        try
        {
            return DetectVariant(checkpointPath) switch
            {
                WanVariant.Vace => new WanVaceRecipe(_familyId).Supports,
                WanVariant.Animate => new WanAnimateRecipe().Supports,
                WanVariant.Animate2 => new WanAnimate2Recipe().Supports,
                WanVariant.S2V => new WanS2VRecipe().Supports,
                _ => Supports,
            };
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or UnauthorizedAccessException)
        {
            Logs.Warning($"[WanVideoRecipe] Could not peek '{checkpointPath}' for variant-aware features; using family defaults. {ex.Message}");
            return Supports;
        }
    }

    /// <summary>The sampling defaults for a CONCRETE checkpoint. Same reason as <see cref="SupportsFor"/>: the variants share Wan's compat classes, so a VACE/Animate/S2V checkpoint resolves under <c>wan-21-14b</c> and would otherwise be handed the plain-Wan defaults — Animate's 20 steps at guidance 1.0 never applied, and the request fell through to <c>WanVideoConfig</c>'s 50/5.0 instead.</summary>
    public VideoDefaults DefaultsFor(string? checkpointPath)
    {
        if (string.IsNullOrWhiteSpace(checkpointPath))
        {
            return Defaults;
        }
        try
        {
            return DetectVariant(checkpointPath) switch
            {
                WanVariant.Vace => new WanVaceRecipe(_familyId).Defaults,
                WanVariant.Animate => new WanAnimateRecipe().Defaults,
                WanVariant.Animate2 => new WanAnimate2Recipe().Defaults,
                WanVariant.S2V => new WanS2VRecipe().Defaults,
                _ => Defaults,
            };
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or UnauthorizedAccessException)
        {
            Logs.Warning($"[WanVideoRecipe] Could not peek '{checkpointPath}' for variant-aware defaults; using family defaults. {ex.Message}");
            return Defaults;
        }
    }
    /// <summary>The sampler/schedule selection a CONCRETE checkpoint accepts. Same reason as <see cref="SupportsFor"/>
    /// and <see cref="DefaultsFor"/>: Animate and Animate-2 share <c>wan-21-14b</c> with the solver-owned plain
    /// backbone, so a host querying capabilities under the compat class id alone would report no selectable
    /// sampler for a checkpoint that, once constructed, accepts UniPC/DPM++2M. Vace and S2V do not narrow their own
    /// sampling support (both stay solver-owned like the family), so only Animate and Animate2 are special-cased.</summary>
    public SamplingCapabilities.SamplingSupport SamplingSupportFor(string? checkpointPath)
    {
        if (string.IsNullOrWhiteSpace(checkpointPath))
        {
            return SamplingCapabilities.ForVideo(_familyId);
        }
        try
        {
            return DetectVariant(checkpointPath) switch
            {
                WanVariant.Animate => SamplingCapabilities.ForVideo("wan-animate"),
                WanVariant.Animate2 => SamplingCapabilities.ForVideo("wan-animate-2"),
                _ => SamplingCapabilities.ForVideo(_familyId),
            };
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or UnauthorizedAccessException)
        {
            Logs.Warning($"[WanVideoRecipe] Could not peek '{checkpointPath}' for variant-aware sampling support; using family defaults. {ex.Message}");
            return SamplingCapabilities.ForVideo(_familyId);
        }
    }

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, _familyId, StringComparison.OrdinalIgnoreCase);

    /// <summary>Wan's official sampling settings: 50 steps at guidance 5.0, 832x480 ("480p"), 33 frames — the resolution/frame-count every <c>WanVariant_Gpu_E2E</c> 480p test verifies coherent at (<c>WanVideoConfig.NumInferenceSteps</c>/<c>GuidanceScale</c>).</summary>
    public VideoDefaults Defaults { get; } = new VideoDefaults { Steps = 50, CfgScale = 5.0f, Width = 832, Height = 480, Frames = 33 };

    /// <inheritdoc/>
    public IVideoRecipePipeline Construct(RecipeContext context)
    {
        // TODO(E-IMG-4/5): VideoRequest.Components overrides for the umT5 / VAE / CLIP-Vision picks are deferred.
        WanVariant variant = DetectVariant(context.CheckpointPath);
        if (variant != WanVariant.Base)
        {
            if (context.VideoSwapModelPath is not null)
            {
                throw new NotSupportedException(
                    $"Video Swap Model is only supported on plain Wan T2V/I2V checkpoints — the loaded checkpoint is a Wan '{variant}' variant.");
            }
            Logs.Info($"[WanVideoRecipe] Checkpoint is a Wan '{variant}' variant — delegating to its recipe.");
            return variant switch
            {
                WanVariant.Vace => new WanVaceRecipe(_familyId).Construct(context),
                WanVariant.Animate => new WanAnimateRecipe().Construct(context),
                WanVariant.Animate2 => new WanAnimate2Recipe().Construct(context),
                WanVariant.S2V => new WanS2VRecipe().Construct(context),
                _ => throw new InvalidOperationException($"Unhandled Wan variant '{variant}'."),
            };
        }
        return ConstructBase(context, _familyId);
    }

    /// <summary>Builds the plain T2V / I2V / TI2V pipeline. <paramref name="familyId"/> selects the config preset when it is one of the three Wan compat classes; otherwise the config is derived from the converted weights.</summary>
    internal IVideoRecipePipeline ConstructBase(RecipeContext context, string? familyId)
    {
        string umt5Path = ModelDownloader.EnsureSideModelAsync(SideModels.Umt5Xxl, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
        (WanVideoCheckpointConverter.ConvertedWeights conv, SafeTensorsLoader ditLoader) = WanVideoCheckpointConverter.LoadAndConvert(context.CheckpointPath);
        List<SafeTensorsLoader> loaders = new List<SafeTensorsLoader> { ditLoader };
        MergedLoraStack? loraStack = null;
        try
        {
            if (conv.Transformer.Count == 0)
            {
                throw new InvalidOperationException($"Wan checkpoint '{context.CheckpointPath}' has no recognized transformer weights after conversion.");
            }
            bool isClipI2V = conv.Transformer.ContainsKey("condition_embedder.image_embedder.norm1.weight");
            int inChannels = conv.Transformer.TryGetValue("patch_embedding.weight", out Tensor? patchEmbed) ? (int)patchEmbed.Shape[1] : 0;
            WanVideoConfig config = ResolveConfig(familyId, isClipI2V, inChannels, conv.Transformer);
            bool isWan21 = config.VaeLatentChannels < 48;
            string mode = isClipI2V ? "CLIP-I2V" : config.InChannels > config.VaeLatentChannels ? "concat-I2V" : "T2V/TI2V";
            Logs.Info($"[WanVideoRecipe] Converted {conv.Transformer.Count} transformer keys ({mode}, in {inChannels}, inner {config.InnerDim}).");

            // Wan 2.2 A14B dual-expert pair: the boundary is resolved BEFORE the transformers are built (both share
            // the config). WanVideoPipeline.SwapToExpert keeps GPU residency sequential — one expert on-device at a
            // time — and already declares CFG-parallel incompatible with a logged fallback.
            WanVideoTransformer? transformer2 = null;
            if (context.VideoSwapModelPath is not null)
            {
                bool isConcatI2V = !isClipI2V && config.InChannels > config.VaeLatentChannels;
                config = config with { BoundaryRatio = ResolveBoundary(context.VideoSwapPercent, config, isConcatI2V) };
            }

            // Merge BEFORE LoadWeights, not after: the merge swaps dictionary entries, and device caches are
            // identity-keyed, so a tensor already captured by a layer would keep serving its stale copy
            // (same ordering constraint as MiniMaxH3Recipe/Sd3Recipe's LoRA merges). One resolved stack is reused
            // for the low-noise expert below (when a MoE pair is configured) rather than re-loading the LoRA file
            // a second time — Wan 2.2's dual-expert split shares one LoRA selection across both experts.
            loraStack = LoraApplier.BuildAndApply(
                LoraResolver.Resolve(context.Loras),
                context.Backend,
                transformerWeights: conv.Transformer);

            WanVideoTransformer transformer = new WanVideoTransformer(config);
            transformer.LoadWeights(conv.Transformer);

            if (context.VideoSwapModelPath is not null)
            {
                string swapPath = File.Exists(context.VideoSwapModelPath) ? context.VideoSwapModelPath
                    : ModelFileLocator.Require(context.VideoSwapModelPath, "Video swap model", "Stable-Diffusion", "diffusion_models", "unet");
                WanVariant swapVariant = DetectVariant(swapPath);
                if (swapVariant != WanVariant.Base)
                {
                    throw new NotSupportedException(
                        $"Wan low-noise expert '{swapPath}' is a Wan '{swapVariant}' variant — the Wan 2.2 expert pair needs plain T2V/I2V checkpoints.");
                }
                Logs.Info($"[WanVideoRecipe] Loading Wan low-noise expert: {swapPath} (boundary {config.BoundaryRatio:0.###}).");
                (WanVideoCheckpointConverter.ConvertedWeights convLow, SafeTensorsLoader lowLoader) = WanVideoCheckpointConverter.LoadAndConvert(swapPath);
                loaders.Add(lowLoader);
                if (convLow.Transformer.Count == 0)
                {
                    throw new InvalidOperationException($"Wan low-noise expert '{swapPath}' has no recognized transformer weights after conversion.");
                }
                WanVideoConfig lowConfig = WanConfigDetector.Detect(convLow.Transformer);
                if (lowConfig.InnerDim != config.InnerDim || lowConfig.NumLayers != config.NumLayers)
                {
                    throw new InvalidOperationException(
                        $"Wan low-noise expert '{swapPath}' does not architecturally match the base checkpoint "
                        + $"(inner {lowConfig.InnerDim} vs {config.InnerDim}, layers {lowConfig.NumLayers} vs {config.NumLayers}) — "
                        + "an A14B expert pair must be two builds of the same architecture.");
                }
                // Same LoRA selection, merged into the low-noise expert's own weight dict — same pre-LoadWeights
                // ordering constraint as the primary expert above.
                loraStack?.ApplyTo(convLow.Transformer, HartsyInference.ModelAssets.Lora.LoraTarget.Transformer, context.Backend);

                transformer2 = new WanVideoTransformer(config);
                transformer2.LoadWeights(convLow.Transformer);
            }

            string vaePath = ModelDownloader.EnsureSideModelAsync(isWan21 ? SideModels.Wan21Vae : SideModels.Wan22Vae, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
            (IWanVaeDecoder vaeDecoder, IWanVaeEncoder vaeEncoder) = VideoRecipeUtils.LoadWanVae(vaePath, isWan21, loaders);

            ClipVisionEncoder? clipVision = null;
            if (isClipI2V)
            {
                string clipPath = ModelDownloader.EnsureSideModelAsync(SideModels.ClipVisionH14, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
                SafeTensorsLoader clipLoader = new SafeTensorsLoader();
                clipLoader.Load(clipPath);
                loaders.Add(clipLoader);
                clipVision = new ClipVisionEncoder(ClipVisionEncoderConfig.ViTH14);
                clipVision.LoadWeights(clipLoader.GetAllTensors());
                Logs.Info("[WanVideoRecipe] CLIP-ViT-H image encoder resolved as side model (Wan2.1 I2V).");
            }

            (T5TextEncoder umt5, T5Tokenizer tokenizer) = VideoRecipeUtils.LoadUmt5(umt5Path, loaders);

            WanVideoPipeline pipeline = new WanVideoPipeline(context.Backend, transformer, vaeDecoder, config, vaeEncoder, transformer2)
            {
                CfgParallelBackend = context.CfgParallelBackend,
                CpBackends = context.CpBackends,
                VaeBackend = context.VaeBackendOrDefault,
            };
            Logs.Info($"[WanVideoRecipe] Wan ready ({mode}{(transformer2 is not null ? ", MoE expert pair" : "")}).");
            return new WanVideoRecipePipeline(context.Backend, context.TextEncoderBackendOrDefault, context.VaeBackendOrDefault,
                pipeline, config, isClipI2V, tokenizer, umt5, transformer, vaeEncoder, clipVision, loaders, transformer2, loraStack);
        }
        catch (Exception ex)
        {
            Logs.Error("[WanVideoRecipe] Construction failed.", ex);
            loraStack?.Dispose();
            foreach (SafeTensorsLoader loader in loaders)
            {
                loader.Dispose();
            }
            throw;
        }
    }

    /// <summary>Resolves the MoE timestep boundary. Null <paramref name="percent"/> → the preset's own boundary when it has one, else Wan 2.2's official 0.875 (T2V) / 0.9 (I2V). An explicit p — the fraction of steps given to the low-noise expert — warps through the shifted flow schedule (<c>boundary = s·p/(1+(s−1)·p)</c>, the same warp the UniPC sigmas use; p=0.5 at shift 8 ≈ 0.889). The warp needs <see cref="WanVideoConfig.FlowShift"/>, which only the engine knows — transports must pass the raw fraction, never a pre-warped boundary.</summary>
    internal static float ResolveBoundary(double? percent, WanVideoConfig config, bool isConcatI2V)
    {
        if (percent is null)
        {
            return config.BoundaryRatio > 0f ? config.BoundaryRatio : isConcatI2V ? 0.9f : 0.875f;
        }
        float shift = config.FlowShift > 0f ? config.FlowShift : 8f;
        float frac = Math.Clamp((float)percent.Value, 0.01f, 0.99f);
        return shift * frac / (1f + (shift - 1f) * frac);
    }

    /// <summary>Maps a Wan compat class (+ the DiT's CLIP-image-embedder presence and patch-embed in_channels) to the engine config preset; falls back to the weight-derived <see cref="WanConfigDetector"/> when the caller supplied the coarse catalog slug rather than a compat class.</summary>
    private static WanVideoConfig ResolveConfig(string? familyId, bool isClipI2V, int inChannels, IReadOnlyDictionary<string, Tensor> weights)
    {
        if (string.Equals(familyId, Wan21_1_3BCompatClassId, StringComparison.OrdinalIgnoreCase))
        {
            return WanVideoConfig.T2V_1_3B;
        }
        if (string.Equals(familyId, Wan21_14BCompatClassId, StringComparison.OrdinalIgnoreCase))
        {
            return inChannels == 36 ? (isClipI2V ? WanVideoConfig.I2V_14B_480p : WanVideoConfig.I2V_A14B) : WanVideoConfig.T2V_14B;
        }
        if (string.Equals(familyId, Wan22_5BCompatClassId, StringComparison.OrdinalIgnoreCase))
        {
            return WanVideoConfig.Ti2V5B;
        }
        WanVideoConfig detected = WanConfigDetector.Detect(weights);
        Logs.Info($"[WanVideoRecipe] Weight-derived config: {WanConfigDetector.Describe(detected)}");
        return detected;
    }

    /// <summary>The Wan conditioning variants that share a compat class.</summary>
    internal enum WanVariant
    {
        /// <summary>Plain T2V / I2V / TI2V backbone.</summary>
        Base,

        /// <summary>VACE control branch (<c>vace_patch_embedding</c> / <c>vace_blocks.*</c>).</summary>
        Vace,

        /// <summary>Wan-Animate pose + face pathway.</summary>
        Animate,

        /// <summary>Wan-Animate-2 driving-video stream. Carries no module of its own, so it is detectable only from the file's <c>__metadata__</c>.</summary>
        Animate2,

        /// <summary>Wan2.2-S2V audio injector.</summary>
        S2V,
    }

    /// <summary>Classifies a Wan checkpoint from its safetensors header keys. The extension took VACE from SwarmUI's model-class id; with no host classifier here the VACE branch sniffs its own signature weights, which are as unique to the variant as the Animate/S2V ones.</summary>
    internal static WanVariant DetectVariant(string checkpointPath)
    {
        IReadOnlySet<string> keys = VideoRecipeUtils.PeekSafeTensorKeys(checkpointPath);
        // First, and by metadata: Animate-2's weights are indistinguishable from a plain I2V-14B's, so every
        // key-based arm below would classify it as Base.
        if (WanVideoCheckpointConverter.IsAnimate2Metadata(VideoRecipeUtils.PeekSafeTensorMetadata(checkpointPath)))
        {
            return WanVariant.Animate2;
        }
        foreach (string key in keys)
        {
            if (key.Contains("vace_patch_embedding", StringComparison.Ordinal) || key.Contains("vace_blocks", StringComparison.Ordinal))
            {
                return WanVariant.Vace;
            }
        }
        foreach (string key in keys)
        {
            if (key.Contains("pose_patch_embedding", StringComparison.Ordinal)
                || key.Contains("motion_encoder", StringComparison.Ordinal)
                || key.Contains("face_adapter", StringComparison.Ordinal))
            {
                return WanVariant.Animate;
            }
        }
        foreach (string key in keys)
        {
            if (key.Contains("audio_encoder", StringComparison.Ordinal) || key.Contains("audio_injector", StringComparison.Ordinal))
            {
                return WanVariant.S2V;
            }
        }
        return WanVariant.Base;
    }
}
