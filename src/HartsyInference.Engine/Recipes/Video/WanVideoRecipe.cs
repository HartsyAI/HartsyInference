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

using HartsyInference.Engine.Features;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>Wan-Video recipe (Wan-AI, umT5-conditioned text/image-to-video) and the entry point for the whole Wan
/// family: the SwarmUI compat classes <c>wan-22-5b</c> / <c>wan-21-1_3b</c> / <c>wan-21-14b</c> are shared by the
/// plain T2V/I2V backbone and the VACE / Animate / S2V conditioning variants, so — exactly like the extension's
/// <c>WanModelVariants.Detect</c> — this recipe sniffs the checkpoint header and hands off to
/// <see cref="WanVaceRecipe"/>, <see cref="WanAnimateRecipe"/>, or <see cref="WanS2VRecipe"/> when it sees their
/// signature weights. Lifted from the SwarmUI backend's <c>WanVideoLoader</c>: umT5-XXL
/// (<see cref="SideModels.Umt5Xxl"/>), the z=48 Wan2.2 VAE (<see cref="SideModels.Wan22Vae"/>) or z=16 Wan2.1 VAE
/// (<see cref="SideModels.Wan21Vae"/>), and CLIP-ViT-H (<see cref="SideModels.ClipVisionH14"/>) for Wan2.1 I2V.</summary>
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

    private readonly string _familyId;

    /// <summary>Binds the recipe to one Wan family id. The catalog slug "wan" derives the config from the weights;
    /// a compat-class id ("wan-22-5b" / "wan-21-1_3b" / "wan-21-14b") selects the matching preset instead.</summary>
    public WanVideoRecipe(string familyId = "wan") => _familyId = familyId;

    /// <inheritdoc/>
    public string Name => _familyId;

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, _familyId, StringComparison.OrdinalIgnoreCase);

    /// <summary>Wan's official sampling settings: 50 steps at guidance 5.0, 832x480 ("480p"), 33 frames — the
    /// resolution/frame-count every <c>WanVariant_Gpu_E2E</c> 480p test verifies coherent at
    /// (<c>WanVideoConfig.NumInferenceSteps</c>/<c>GuidanceScale</c>).</summary>
    public VideoDefaults Defaults { get; } = new VideoDefaults { Steps = 50, CfgScale = 5.0f, Width = 832, Height = 480, Frames = 33 };

    /// <inheritdoc/>
    public IVideoRecipePipeline Construct(RecipeContext context)
    {
        // TODO(E-IMG-4/5): LoRA (the extension's GenerateWithLoras merges the stack into BOTH MoE experts), the
        // Wan2.2 A14B low-noise expert pair (Refiner Model / Video Swap Model slots + the boundary warp), and
        // VideoRequest.Components overrides for the umT5 / VAE / CLIP-Vision picks are deferred — this is the
        // single-expert, canonical-side-model path.
        WanVariant variant = DetectVariant(context.CheckpointPath);
        if (variant != WanVariant.Base)
        {
            Logs.Info($"[WanVideoRecipe] Checkpoint is a Wan '{variant}' variant — delegating to its recipe.");
            return variant switch
            {
                WanVariant.Vace => new WanVaceRecipe(_familyId).Construct(context),
                WanVariant.Animate => new WanAnimateRecipe().Construct(context),
                WanVariant.S2V => new WanS2VRecipe().Construct(context),
                _ => throw new InvalidOperationException($"Unhandled Wan variant '{variant}'."),
            };
        }
        return ConstructBase(context, _familyId);
    }

    /// <summary>Builds the plain T2V / I2V / TI2V pipeline. <paramref name="familyId"/> selects the config preset when
    /// it is one of the three Wan compat classes; otherwise the config is derived from the converted weights.</summary>
    internal IVideoRecipePipeline ConstructBase(RecipeContext context, string? familyId)
    {
        string umt5Path = ModelDownloader.EnsureSideModelAsync(SideModels.Umt5Xxl, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
        (WanVideoCheckpointConverter.ConvertedWeights conv, SafeTensorsLoader ditLoader) = WanVideoCheckpointConverter.LoadAndConvert(context.CheckpointPath);
        List<SafeTensorsLoader> loaders = new List<SafeTensorsLoader> { ditLoader };
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

            WanVideoTransformer transformer = new WanVideoTransformer(config);
            transformer.LoadWeights(conv.Transformer);

            string vaePath = ModelDownloader.EnsureSideModelAsync(isWan21 ? SideModels.Wan21Vae : SideModels.Wan22Vae, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
            (Dictionary<string, Tensor> vaeWeightsRaw, IReadOnlyList<SafeTensorsLoader> vaeLoaders) = LanceCheckpointConverter.LoadVae(vaePath);
            loaders.AddRange(vaeLoaders);
            Dictionary<string, Tensor> vaeWeights = VaePrecisionHelper.CastVaeWeights(vaeWeightsRaw, DType.F32);
            IWanVaeDecoder vaeDecoder;
            IWanVaeEncoder vaeEncoder;
            if (isWan21)
            {
                Wan21VaeDecoder decoder = new Wan21VaeDecoder();
                decoder.LoadWeights(vaeWeights);
                vaeDecoder = decoder;
                Wan21VaeEncoder encoder = new Wan21VaeEncoder();
                encoder.LoadWeights(vaeWeights);
                vaeEncoder = encoder;
            }
            else
            {
                Wan22VaeDecoder decoder = new Wan22VaeDecoder();
                decoder.LoadWeights(vaeWeights);
                vaeDecoder = decoder;
                Wan22VaeEncoder encoder = new Wan22VaeEncoder();
                encoder.LoadWeights(vaeWeights);
                vaeEncoder = encoder;
            }

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

            SafeTensorsLoader umt5Loader = new SafeTensorsLoader();
            umt5Loader.Load(umt5Path);
            loaders.Add(umt5Loader);
            Dictionary<string, Tensor> umt5Weights = CheckpointConvertUtils.ApplyFp8ScaledDequant(umt5Loader.GetAllTensors());
            T5TextEncoder umt5 = new T5TextEncoder(T5TextEncoderConfig.Umt5Xxl);
            umt5.LoadWeights(umt5Weights);
            T5Tokenizer tokenizer = T5Tokenizer.CreateUmt5(maxLength: TokenLength);

            WanVideoPipeline pipeline = new WanVideoPipeline(context.Backend, transformer, vaeDecoder, config, vaeEncoder)
            {
                CfgParallelBackend = context.CfgParallelBackend,
            };
            Logs.Info($"[WanVideoRecipe] Wan ready ({mode}).");
            return new WanVideoRecipePipeline(context.Backend, context.TextEncoderBackendOrDefault, pipeline, config,
                isClipI2V, tokenizer, umt5, transformer, vaeEncoder, clipVision, loaders);
        }
        catch (Exception ex)
        {
            Logs.Error("[WanVideoRecipe] Construction failed.", ex);
            foreach (SafeTensorsLoader loader in loaders)
            {
                loader.Dispose();
            }
            throw;
        }
    }

    /// <summary>Maps a Wan compat class (+ the DiT's CLIP-image-embedder presence and patch-embed in_channels) to the
    /// engine config preset; falls back to the weight-derived <see cref="WanConfigDetector"/> when the caller supplied
    /// the coarse catalog slug rather than a compat class.</summary>
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

        /// <summary>Wan2.2-S2V audio injector.</summary>
        S2V,
    }

    /// <summary>Classifies a Wan checkpoint from its safetensors header keys. The extension took VACE from SwarmUI's
    /// model-class id; with no host classifier here the VACE branch sniffs its own signature weights, which are as
    /// unique to the variant as the Animate/S2V ones.</summary>
    internal static WanVariant DetectVariant(string checkpointPath)
    {
        IReadOnlySet<string> keys = VideoRecipeUtils.PeekSafeTensorKeys(checkpointPath);
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
