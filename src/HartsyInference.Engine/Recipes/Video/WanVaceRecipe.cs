using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
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

/// <summary>Wan2.1 VACE recipe (Video All-in-one Creation and Editing) — the plain Wan backbone plus a parallel control branch (<c>vace_patch_embedding</c> + <c>vace_blocks.*</c>) that conditions every denoise step on a VAE-encoded pose/depth/edge/sketch control clip. Lifted from the SwarmUI backend's <c>WanVaceLoader</c>: umT5-XXL (<see cref="SideModels.Umt5Xxl"/>) and the z=16 Wan2.1 VAE (<see cref="SideModels.Wan21Vae"/>); VACE is text+control only, so there is no CLIP image encoder.</summary>
public sealed class WanVaceRecipe : IVideoRecipe
{
    private readonly string _familyId;

    /// <summary>Binds the recipe to a family id; the 1.3B compat class selects the small VACE preset.</summary>
    public WanVaceRecipe(string familyId = "wan-vace") => _familyId = familyId;

    /// <inheritdoc/>
    public string Name => _familyId;


    /// <inheritdoc/>
    /// <remarks>VACE requires a control image/video — it throws without one, so the init image is mandatory rather than optional.</remarks>
    /// <remarks><see cref="VideoFeatures.Lora"/> added 2026-08-20, matching the plain Wan family that already had it — same original-Wan module naming, so the existing <c>KohyaWan</c> / <c>DiffusersWan</c> mappers cover it unchanged.</remarks>
    public VideoFeatures Supports => VideoFeatures.InitImage | VideoFeatures.Lora;
    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, _familyId, StringComparison.OrdinalIgnoreCase);

    /// <summary>Wan VACE's official sampling settings: 50 steps at guidance 5.0 (<c>WanVideoConfig.NumInferenceSteps</c>/<c>GuidanceScale</c>).</summary>
    public VideoDefaults Defaults { get; } = new VideoDefaults { Steps = 50, CfgScale = 5.0f };

    /// <inheritdoc/>
    public IVideoRecipePipeline Construct(RecipeContext context)
    {
        // TODO(E-IMG-4/5): LoRA and VideoRequest.Components overrides for the umT5 / VAE picks are deferred.
        WanVideoConfig config = string.Equals(_familyId, WanVideoRecipe.Wan21_1_3BCompatClassId, StringComparison.OrdinalIgnoreCase)
            ? WanVideoConfig.Vace_1_3B : WanVideoConfig.Vace_14B;

        string umt5Path = ModelDownloader.EnsureSideModelAsync(SideModels.Umt5Xxl, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
        string vaePath = ModelDownloader.EnsureSideModelAsync(SideModels.Wan21Vae, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();

        (WanVideoCheckpointConverter.ConvertedWeights conv, SafeTensorsLoader ditLoader) = WanVideoCheckpointConverter.LoadAndConvert(context.CheckpointPath);
        List<SafeTensorsLoader> loaders = new List<SafeTensorsLoader> { ditLoader };
        MergedLoraStack? loraStack = null;
        try
        {
            if (conv.Transformer.Count == 0)
            {
                throw new InvalidOperationException($"Wan VACE checkpoint '{context.CheckpointPath}' has no recognized transformer weights after conversion.");
            }
            if (!conv.Transformer.ContainsKey("vace_patch_embedding.weight"))
            {
                throw new InvalidOperationException(
                    $"'{context.CheckpointPath}' is routed as Wan VACE but has no 'vace_patch_embedding' weights after conversion — "
                    + "it may be a plain Wan2.1 checkpoint, or a layout the converter doesn't map yet.");
            }
            Logs.Info($"[WanVaceRecipe] Converted {conv.Transformer.Count} transformer keys (VACE, {config.VaceLayers.Length} control layers, inner {config.InnerDim}).");

            WanVaceTransformer transformer = new WanVaceTransformer(config);
            // Merge any requested LoRAs BEFORE LoadWeights — device caches are identity-keyed, so merging
            // after would leave layers serving the pre-merge tensors (the Sd3Recipe ordering rule).
            loraStack = LoraApplier.BuildAndApply(
                LoraResolver.Resolve(context.Loras), context.Backend, transformerWeights: conv.Transformer);
            transformer.LoadWeights(conv.Transformer);

            (IWanVaeDecoder vaeDecoder, IWanVaeEncoder vaeEncoder) = VideoRecipeUtils.LoadWanVae(vaePath, isWan21: true, loaders);

            (T5TextEncoder umt5, T5Tokenizer tokenizer) = VideoRecipeUtils.LoadUmt5(umt5Path, loaders);

            WanVacePipeline pipeline = new WanVacePipeline(context.Backend, transformer, vaeDecoder, vaeEncoder, config);
            Logs.Info("[WanVaceRecipe] Wan VACE ready (control-video).");
            return new WanVaceRecipePipeline(context.Backend, pipeline, config, tokenizer, umt5, transformer, vaeEncoder, loaders, loraStack);
        }
        catch (Exception ex)
        {
            Logs.Error("[WanVaceRecipe] Construction failed.", ex);
            foreach (SafeTensorsLoader loader in loaders)
            {
                loader.Dispose();
            }
            loraStack?.Dispose();
            throw;
        }
    }
}
