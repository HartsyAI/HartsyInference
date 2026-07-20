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

/// <summary>Wan2.1 VACE recipe (Video All-in-one Creation and Editing) — the plain Wan backbone plus a parallel control
/// branch (<c>vace_patch_embedding</c> + <c>vace_blocks.*</c>) that conditions every denoise step on a VAE-encoded
/// pose/depth/edge/sketch control clip. Lifted from the SwarmUI backend's <c>WanVaceLoader</c>: umT5-XXL
/// (<see cref="SideModels.Umt5Xxl"/>) and the z=16 Wan2.1 VAE (<see cref="SideModels.Wan21Vae"/>); VACE is
/// text+control only, so there is no CLIP image encoder.</summary>
public sealed class WanVaceRecipe : IVideoRecipe
{
    private readonly string _familyId;

    /// <summary>Binds the recipe to a family id; the 1.3B compat class selects the small VACE preset.</summary>
    public WanVaceRecipe(string familyId = "wan-vace") => _familyId = familyId;

    /// <inheritdoc/>
    public string Name => _familyId;

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, _familyId, StringComparison.OrdinalIgnoreCase);

    /// <inheritdoc/>
    public IVideoRecipePipeline Construct(RecipeContext context)
    {
        // TODO(E-IMG-4/5): LoRA and VideoRequest.Components overrides for the umT5 / VAE picks are deferred.
        WanVideoConfig config = string.Equals(_familyId, WanVideoRecipe.Wan21_1_3BCompatClassId, StringComparison.OrdinalIgnoreCase)
            ? WanVideoConfig.Vace_1_3B
            : WanVideoConfig.Vace_14B;

        string umt5Path = ModelDownloader.EnsureSideModelAsync(SideModels.Umt5Xxl, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
        string vaePath = ModelDownloader.EnsureSideModelAsync(SideModels.Wan21Vae, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();

        (WanVideoCheckpointConverter.ConvertedWeights conv, SafeTensorsLoader ditLoader) = WanVideoCheckpointConverter.LoadAndConvert(context.CheckpointPath);
        List<SafeTensorsLoader> loaders = new List<SafeTensorsLoader> { ditLoader };
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
            transformer.LoadWeights(conv.Transformer);

            (Dictionary<string, Tensor> vaeWeightsRaw, IReadOnlyList<SafeTensorsLoader> vaeLoaders) = LanceCheckpointConverter.LoadVae(vaePath);
            loaders.AddRange(vaeLoaders);
            Dictionary<string, Tensor> vaeWeights = VideoRecipeUtils.CastWeights(vaeWeightsRaw, DType.F32);
            Wan21VaeDecoder vaeDecoder = new Wan21VaeDecoder();
            vaeDecoder.LoadWeights(vaeWeights);
            Wan21VaeEncoder vaeEncoder = new Wan21VaeEncoder();
            vaeEncoder.LoadWeights(vaeWeights);

            SafeTensorsLoader umt5Loader = new SafeTensorsLoader();
            umt5Loader.Load(umt5Path);
            loaders.Add(umt5Loader);
            Dictionary<string, Tensor> umt5Weights = CheckpointConvertUtils.ApplyFp8ScaledDequant(umt5Loader.GetAllTensors());
            T5TextEncoder umt5 = new T5TextEncoder(T5TextEncoderConfig.Umt5Xxl);
            umt5.LoadWeights(umt5Weights);
            T5Tokenizer tokenizer = T5Tokenizer.CreateUmt5(maxLength: WanVideoRecipe.TokenLength);

            WanVacePipeline pipeline = new WanVacePipeline(context.Backend, transformer, vaeDecoder, vaeEncoder, config);
            Logs.Info("[WanVaceRecipe] Wan VACE ready (control-video).");
            return new WanVaceRecipePipeline(context.Backend, pipeline, config, tokenizer, umt5, transformer, vaeEncoder, loaders);
        }
        catch (Exception ex)
        {
            Logs.Error("[WanVaceRecipe] Construction failed.", ex);
            foreach (SafeTensorsLoader loader in loaders)
            {
                loader.Dispose();
            }
            throw;
        }
    }
}
