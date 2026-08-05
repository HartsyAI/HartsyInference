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

/// <summary>LTX-Video recipe (Lightricks; catalog slug "ltx-video", SwarmUI compat class
/// <c>lightricks-ltx-video</c>). Targets the single-file checkpoints that bundle DiT + VAE — the variant (0.9 2B base,
/// 0.9.5 2B with the timestep-conditioned VAE, 0.9.7/0.9.8 13B) is detected from the DiT layer count and the VAE key
/// shape. Lifted from the SwarmUI backend's <c>LtxVideoLoader</c>; the one side model is plain T5-XXL
/// (<see cref="SideModels.T5XxlEnconly"/> — LTX uses standard T5, not Wan's umT5).</summary>
public sealed class LtxVideoRecipe : IVideoRecipe
{
    /// <summary>LTX's T5 context length (diffusers uses 128 tokens).</summary>
    internal const int TokenLength = 128;

    /// <inheritdoc/>
    public string Name => "ltx-video";

    /// <inheritdoc/>
    public bool Matches(string familyId) =>
        string.Equals(familyId, "ltx-video", StringComparison.OrdinalIgnoreCase)
        || string.Equals(familyId, "lightricks-ltx-video", StringComparison.OrdinalIgnoreCase);

    /// <summary>LTX-Video's official sampling settings: 50 steps at guidance 3.0, 704x480, 97 frames
    /// (<c>LtxVideoConfig.NumInferenceSteps</c>/<c>GuidanceScale</c>; frame count matches the pipeline's own
    /// <c>modelDefault</c> in <see cref="LtxVideoRecipePipeline.Generate"/>).</summary>
    public VideoDefaults Defaults { get; } = new VideoDefaults { Steps = 50, CfgScale = 3.0f, Width = 704, Height = 480, Frames = 97 };

    /// <inheritdoc/>
    public IVideoRecipePipeline Construct(RecipeContext context)
    {
        // TODO(E-IMG-4/5): LoRA, image-to-video conditioning, and a VideoRequest.Components T5 override are deferred —
        // this is the vanilla single-file text-to-video path with the canonical T5-XXL side model.
        string t5Path = ModelDownloader.EnsureSideModelAsync(SideModels.T5XxlEnconly, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
        (LtxVideoCheckpointConverter.ConvertedWeights conv, SafeTensorsLoader ckptLoader) = LtxVideoCheckpointConverter.LoadAndConvert(context.CheckpointPath);
        List<SafeTensorsLoader> loaders = new List<SafeTensorsLoader> { ckptLoader };
        try
        {
            if (conv.Transformer.Count == 0)
            {
                throw new InvalidOperationException($"LTX checkpoint '{context.CheckpointPath}' has no recognized transformer weights after conversion.");
            }
            if (conv.Vae.Count == 0)
            {
                throw new InvalidOperationException(
                    $"LTX checkpoint '{context.CheckpointPath}' has no bundled VAE weights — a full single-file LTX-Video checkpoint (DiT + VAE) is required.");
            }
            Logs.Info($"[LtxVideoRecipe] Converted {conv.Transformer.Count} DiT keys, {conv.Vae.Count} VAE keys.");

            int maxBlock = MaxTransformerBlockIndex(conv.Transformer.Keys);
            string nameLc = Path.GetFileName(context.CheckpointPath).ToLowerInvariant();
            bool is13B = maxBlock >= 28 || nameLc.Contains("13b", StringComparison.Ordinal)
                || nameLc.Contains("0.9.7", StringComparison.Ordinal) || nameLc.Contains("0.9.8", StringComparison.Ordinal);
            bool timestepVae = is13B || LtxVideoCheckpointConverter.IsTimestepVae(ckptLoader.GetAllTensors().Keys) || nameLc.Contains("0.9.5", StringComparison.Ordinal);
            LtxVideoConfig config = is13B ? LtxVideoConfig.V097 : timestepVae ? LtxVideoConfig.V095 : LtxVideoConfig.V09;
            Logs.Info($"[LtxVideoRecipe] Variant: {(is13B ? "0.9.7/13B" : timestepVae ? "0.9.5" : "0.9")} ({maxBlock + 1} DiT layers, {(timestepVae ? "timestep-conditioned" : "base")} VAE).");

            LtxVideoTransformer transformer = new LtxVideoTransformer(config);
            transformer.LoadWeights(conv.Transformer);
            LtxVideoVaeDecoder vae = timestepVae
                ? new LtxVideoVaeDecoder(blockOutChannels: [256, 512, 1024], spatioTemporalScaling: [true, true, true],
                    layersPerBlock: [5, 5, 5, 5], patchSize: 4, isCausal: false, timestepConditioned: true,
                    upsampleFactor: [2, 2, 2], upsampleResidual: [true, true, true])
                : new LtxVideoVaeDecoder();
            vae.LoadWeights(VaePrecisionHelper.CastVaeWeights(conv.Vae, DType.F32));

            // 13B fp8 weights stay fp8-resident (~13 GB); caching their F16 casts would roughly double VRAM and OOM a
            // 24 GB card — dequant transiently per GEMM instead (the verified 13B recipe, matching the Wan fp8 path).
            if (is13B)
            {
                RecipeBackendFlags.DisableCacheWeightCasts(context, "LtxVideoRecipe");
            }

            SafeTensorsLoader t5Loader = new SafeTensorsLoader();
            t5Loader.Load(t5Path);
            loaders.Add(t5Loader);
            Dictionary<string, Tensor> t5Weights = VideoRecipeUtils.StripPrefix(t5Loader.GetAllTensors(), "text_encoders.t5xxl.transformer.");
            if (t5Weights.Count == 0)
            {
                throw new InvalidOperationException($"T5 model file '{t5Path}' has no usable T5 tensors.");
            }
            T5TextEncoder t5 = new T5TextEncoder(T5TextEncoderConfig.Xxl);
            t5.LoadWeights(t5Weights);
            T5Tokenizer tokenizer = new T5Tokenizer(maxLength: TokenLength);

            LtxVideoPipeline pipeline = new LtxVideoPipeline(context.Backend, transformer, vae, config);
            Logs.Info("[LtxVideoRecipe] LTX-Video ready (text-to-video).");
            return new LtxVideoRecipePipeline(context.Backend, pipeline, config, tokenizer, t5, transformer, loaders);
        }
        catch (Exception ex)
        {
            Logs.Error("[LtxVideoRecipe] Construction failed.", ex);
            foreach (SafeTensorsLoader loader in loaders)
            {
                loader.Dispose();
            }
            throw;
        }
    }

    /// <summary>Highest <c>transformer_blocks.{i}</c> index across the converted DiT keys (−1 if none) — tells the
    /// 48-layer 13B (0.9.7/0.9.8) apart from the 28-layer 2B (0.9/0.9.5).</summary>
    private static int MaxTransformerBlockIndex(IEnumerable<string> keys)
    {
        const string Token = "transformer_blocks.";
        int max = -1;
        foreach (string key in keys)
        {
            int at = key.IndexOf(Token, StringComparison.Ordinal);
            if (at < 0)
            {
                continue;
            }
            int start = at + Token.Length, end = start;
            while (end < key.Length && char.IsDigit(key[end]))
            {
                end++;
            }
            if (end > start)
            {
                max = Math.Max(max, int.Parse(key.AsSpan(start, end - start), System.Globalization.CultureInfo.InvariantCulture));
            }
        }
        return max;
    }
}
