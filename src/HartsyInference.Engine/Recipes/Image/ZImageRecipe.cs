using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Engine.Features;
namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Z-Image recipe (Tongyi Lab NextDiT): the checkpoint carries the transformer only; the Qwen3-4B text encoder and the Flux VAE are resolved as side models. Lifted from the SwarmUI backend's <c>ZImageLoader</c>; constructs the components and drives generation through <see cref="ZImageRecipePipeline"/>.</summary>
public sealed class ZImageRecipe : IArchitectureRecipe
{
    /// <inheritdoc/>
    public string Name => "zimage";

    /// <inheritdoc/>
    /// <remarks>Z-Image shares the Flux VAE; the encoder is constructed alongside the decoder and ZImagePipeline implements the masked path.</remarks>
    public ImageFeatures Supports => ImageFeatures.Img2Img | ImageFeatures.Inpaint | ImageFeatures.Regional;

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "zimage", StringComparison.OrdinalIgnoreCase);

    /// <summary>Z-Image Turbo's official sampling settings: 8 steps, guidance-free (CFG 1.0), 1024x1024 (<c>GenerationDefaults.ZImageTurbo</c>).</summary>
    public static ImageDefaults FamilyDefaults { get; } = new ImageDefaults { Steps = 8, CfgScale = 1.0f, Width = 1024, Height = 1024 };

    /// <summary>Z-Image Base's official undistilled sampling settings: 50 steps, CFG 5.0, 1024x1024, shift 6.</summary>
    public static ImageDefaults BaseDefaults { get; } = new ImageDefaults
    {
        Steps = 50,
        CfgScale = 5.0f,
        Width = 1024,
        Height = 1024,
        SigmaShift = 6.0,
    };

    /// <inheritdoc/>
    public ImageDefaults Defaults => FamilyDefaults;

    /// <inheritdoc/>
    public IRecipePipeline Construct(RecipeContext context)
    {
        // Resolve the two side models synchronously (Construct is a sync seam): Qwen3-4B text encoder + Flux VAE.
        // TODO(E-IMG-4): honor a user-picked Qwen/VAE override from ImageRequest.Components/Extra instead of always
        // taking the canonical SideModels entry (the SwarmUI loader read input.Get(T2IParamTypes.QwenModel/VAE)).
        string qwenPath = ModelDownloader.EnsureSideModelAsync(SideModels.Qwen3_4B, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
        string vaePath = ModelDownloader.EnsureSideModelAsync(SideModels.FluxAe, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();

        SafeTensorsLoader? zLoader = null;
        SafeTensorsLoader? qwenLoader = null;
        SafeTensorsLoader? vaeLoader = null;
        ZImageTransformer? transformer = null;
        LlamaStyleEncoder? qwen = null;
        Qwen3Tokenizer? tokenizer = null;
        Tensor[] transformerWeightTensors = [];
        Tensor[] qwenWeightTensors = [];
        Tensor[] ownedVaeWeights = [];
        HashSet<Tensor> constructionOwnedVaeCasts = new(ReferenceEqualityComparer.Instance);
        try
        {
            // 1. Load + convert the Z-Image transformer (checkpoint carries only these weights).
            (ZImageCheckpointConverter.ConvertedWeights zConv, zLoader) =
                ZImageCheckpointConverter.LoadAndConvert(context.CheckpointPath);
            if (zConv.Transformer.Count == 0)
                throw new InvalidOperationException("Z-Image checkpoint has no transformer weights.");

            bool isBase = zConv.Variant == ZImageCheckpointConverter.CheckpointVariant.Base;
            if (zConv.Variant == ZImageCheckpointConverter.CheckpointVariant.Unknown)
            {
                // Tensor shapes cannot distinguish distilled Turbo from Base. Preserve the long-standing Turbo
                // default for an ambiguously renamed file, but say so rather than silently choosing a schedule.
                Logs.Warning($"[ZImageRecipe] Checkpoint filename '{Path.GetFileName(context.CheckpointPath)}' " +
                    "contains neither an unambiguous 'base' nor 'turbo' token; defaulting to Turbo shift=3. " +
                    "Rename the file with its variant token to select the correct schedule.");
            }
            ZImageConfig zConfig = ZImageConfig.FromWeights(zConv.Transformer, isBase);
            Logs.Info($"[ZImageRecipe] Building {zConv.Variant} transformer " +
                $"(SchedulerShift={zConfig.SchedulerShift}).");
            transformer = new ZImageTransformer(zConfig);
            transformer.LoadWeights(zConv.Transformer);
            transformerWeightTensors = SnapshotWeights(transformer.EnumerateWeights());

            // 2. Load the Qwen3-4B encoder + its embedded tokenizer.
            qwenLoader = new SafeTensorsLoader();
            qwenLoader.Load(qwenPath);
            IReadOnlyDictionary<string, Tensor> qwenWeights = qwenLoader.GetAllTensors();
            if (qwenWeights.Count == 0)
                throw new InvalidOperationException($"Qwen3 model file '{qwenPath}' has no tensors.");

            qwen = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Qwen3_4B);
            qwen.LoadWeights(qwenWeights);
            qwenWeightTensors = SnapshotWeights(qwen.EnumerateWeights());
            // Diffusers' Z-Image contract uses a 512-token Qwen window. The recipe trims right-padding before
            // execution, so ordinary short prompts do not pay the quadratic cost of this reference-compatible cap.
            tokenizer = new Qwen3Tokenizer(maxLength: 512);

            // 3. Load the Flux VAE (Z-Image reuses it verbatim). LoadFluxVaeF32 may create F32 casts that the
            // mmap loader does not own; PreferredVaeDtype may then replace those with BF16 casts. Dispose every
            // superseded staging tensor now and hand the final distinct tensors to the recipe wrapper as explicit
            // owners — VaeDecoder/VaeEncoder deliberately are not IDisposable.
            (Dictionary<string, Tensor> stagedVaeWeights, vaeLoader) = LoaderVaeUtils.LoadFluxVaeF32(vaePath);
            if (stagedVaeWeights.Count == 0)
                throw new InvalidOperationException($"VAE file '{vaePath}' has no usable VAE tensors.");

            HashSet<Tensor> stagedVaeSet = new(stagedVaeWeights.Values, ReferenceEqualityComparer.Instance);
            foreach (Tensor staged in stagedVaeSet)
            {
                if (staged.OwnsMemory)
                    constructionOwnedVaeCasts.Add(staged);
            }

            Dictionary<string, Tensor> vaeWeights = VaePrecisionHelper.CastVaeWeights(stagedVaeWeights,
                VaePrecisionHelper.PreferredVaeDtype(context.VaeBackendOrDefault));
            HashSet<Tensor> finalVaeSet = new(vaeWeights.Values, ReferenceEqualityComparer.Instance);
            foreach (Tensor final in finalVaeSet)
            {
                if (final.OwnsMemory && !stagedVaeSet.Contains(final))
                    constructionOwnedVaeCasts.Add(final);
            }
            HashSet<Tensor> seenStaged = new(ReferenceEqualityComparer.Instance);
            foreach (KeyValuePair<string, Tensor> pair in stagedVaeWeights)
            {
                Tensor staged = pair.Value;
                if (seenStaged.Add(staged) && !finalVaeSet.Contains(staged))
                {
                    constructionOwnedVaeCasts.Remove(staged);
                    staged.Dispose();
                }
            }
            ownedVaeWeights = finalVaeSet.ToArray();

            VaeDecoder vae = new VaeDecoder(VaeConfig.ZImage);
            vae.LoadWeights(vaeWeights);
            VaeEncoder vaeEncoder = LoaderVaeUtils.BuildEncoder(VaeConfig.ZImage, vaeWeights, "ZImageRecipe");

            ZImagePipeline pipeline = new ZImagePipeline(context.Backend, transformer, vae, vaeEncoder, zConfig)
            {
                TextEncoderBackend = context.TextEncoderBackendOrDefault,
                VaeBackend = context.VaeBackendOrDefault,
            };
            Logs.Info("[ZImageRecipe] Z-Image ready.");
            return new ZImageRecipePipeline(pipeline, qwen, tokenizer, transformer, vae, vaeEncoder,
                transformerWeightTensors, qwenWeightTensors, ownedVaeWeights,
                context.Backend, context.TextEncoderBackendOrDefault,
                isBase ? BaseDefaults : FamilyDefaults,
                zLoader, qwenLoader, vaeLoader);
        }
        catch
        {
            if (qwen is not null && qwenWeightTensors.Length == 0)
                TryCleanup(() => qwenWeightTensors = SnapshotWeights(qwen.EnumerateWeights()),
                    "snapshot Qwen weights during construction rollback");
            if (transformer is not null && transformerWeightTensors.Length == 0)
                TryCleanup(() => transformerWeightTensors = SnapshotWeights(transformer.EnumerateWeights()),
                    "snapshot transformer weights during construction rollback");
            DisposeWeights(qwenWeightTensors, "Qwen weight");
            DisposeWeights(transformerWeightTensors, "transformer weight");
            foreach (Tensor weight in constructionOwnedVaeCasts)
                TryCleanup(weight.Dispose, "owned VAE cast");
            if (tokenizer is not null) TryCleanup(tokenizer.Dispose, "tokenizer");
            if (qwen is not null) TryCleanup(qwen.Dispose, "Qwen encoder");
            if (transformer is not null) TryCleanup(transformer.Dispose, "transformer");
            if (vaeLoader is not null) TryCleanup(vaeLoader.Dispose, "VAE loader");
            if (qwenLoader is not null) TryCleanup(qwenLoader.Dispose, "Qwen loader");
            if (zLoader is not null) TryCleanup(zLoader.Dispose, "transformer loader");
            throw;
        }
    }

    private static Tensor[] SnapshotWeights(IEnumerable<Tensor> weights)
    {
        HashSet<Tensor> distinct = new(ReferenceEqualityComparer.Instance);
        foreach (Tensor weight in weights)
            distinct.Add(weight);
        return [.. distinct];
    }

    private static void DisposeWeights(IEnumerable<Tensor> weights, string description)
    {
        foreach (Tensor weight in weights)
            TryCleanup(weight.Dispose, description);
    }

    private static void TryCleanup(Action cleanup, string description)
    {
        try
        {
            cleanup();
        }
        catch (Exception cleanupError)
        {
            // Construction rollback must never replace the exception that explains why construction failed,
            // and one broken owner must not prevent the remaining mmaps/casts from being released.
            Logs.Warning($"[ZImageRecipe] Failed to release {description}: {cleanupError.Message}");
        }
    }
}
