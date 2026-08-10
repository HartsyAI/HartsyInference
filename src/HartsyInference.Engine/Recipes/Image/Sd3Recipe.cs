using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Placement;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Stable Diffusion 3 / 3.5 recipe (MMDiT, triple text encoder). Lifted from the SwarmUI backend's <c>Sd3Loader</c>, extended to ComfyUI's mix-and-match component layout: every component the checkpoint bundles is used as-is, and each one it omits resolves independently as a canonical side model — CLIP-L (<see cref="SideModels.ClipL"/>), CLIP-G (<see cref="SideModels.ClipG"/>), T5-XXL (<see cref="SideModels.T5XxlEnconly"/>), VAE (<see cref="SideModels.Sd35Vae"/>). A stock <c>sd3.5_medium.safetensors</c> bundles the MMDiT and the VAE but no text encoders, so it takes the modular path for CLIP-L/CLIP-G/T5. Constructs and drives through <see cref="Sd3RecipePipeline"/>.</summary>
public sealed class Sd3Recipe : IArchitectureRecipe
{
    /// <inheritdoc/>
    public string Name => "sd3";

    /// <inheritdoc/>
    /// <remarks>SD3's VAE encoder is constructed alongside the decoder, and Sd3Pipeline implements the blend-on-vanilla masked path.</remarks>
    public ImageFeatures Supports => ImageFeatures.Img2Img | ImageFeatures.Inpaint | ImageFeatures.Lora;

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "sd3", StringComparison.OrdinalIgnoreCase);

    /// <summary>SD 3.5's official sampling settings: 28 steps at CFG 7.0, 1024x1024 (diffusers <c>StableDiffusion3Pipeline.__call__</c>, mirrored by <c>GenerationDefaults.Sd35</c>).</summary>
    public static ImageDefaults FamilyDefaults { get; } = new ImageDefaults { Steps = 28, CfgScale = 7.0f, Width = 1024, Height = 1024 };

    /// <inheritdoc/>
    public ImageDefaults Defaults => FamilyDefaults;

    /// <inheritdoc/>
    public IRecipePipeline Construct(RecipeContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        // TODO(E-IMG-4): honor split-file CLIP-L / CLIP-G / T5 / VAE overrides from ImageRequest.Components
        // (the SwarmUI loader read input.Get(T2IParamTypes.ClipLModel/ClipGModel/T5XXLModel/VAE)); a component the
        // checkpoint omits currently always resolves to the canonical side model rather than a user-picked file.
        (Sd3CheckpointConverter.ConvertedWeights converted, SafeTensorsLoader mainLoader) = Sd3CheckpointConverter.LoadAndConvert(context.CheckpointPath);
        if (converted.Transformer.Count == 0)
        {
            mainLoader.Dispose();
            throw new InvalidOperationException($"SD3: '{context.CheckpointPath}' contains no MMDiT transformer weights — not an SD3/SD3.5 diffusion model.");
        }

        List<SafeTensorsLoader> loaders = new List<SafeTensorsLoader> { mainLoader };
        MergedLoraStack? loraStack = null;
        try
        {
            Dictionary<string, Tensor> clipLWeights = converted.ClipL;
            Dictionary<string, Tensor> clipGWeights = converted.ClipG;
            Dictionary<string, Tensor> t5Weights = converted.T5;
            Dictionary<string, Tensor> vaeWeights = converted.Vae;

            if (clipLWeights.Count == 0)
            {
                string path = ResolveComponent(SideModels.ClipL, "CLIP-L");
                clipLWeights = RequireWeights(LoaderClipUtils.StripClipPrefix(OpenSide(loaders, path).GetAllTensors(), "clip_l", 0), "CLIP-L", path);
                Logs.Info($"[Sd3Recipe] CLIP-L resolved as a separate file: {path}.");
            }
            if (clipGWeights.Count == 0)
            {
                string path = ResolveComponent(SideModels.ClipG, "CLIP-G");
                clipGWeights = RequireWeights(LoaderClipUtils.StripClipPrefix(OpenSide(loaders, path).GetAllTensors(), "clip_g", 1), "CLIP-G", path);
                Logs.Info($"[Sd3Recipe] CLIP-G resolved as a separate file: {path}.");
            }
            if (t5Weights.Count == 0)
            {
                string path = ResolveComponent(SideModels.T5XxlEnconly, "T5-XXL");
                t5Weights = RequireWeights(StripT5Prefix(OpenSide(loaders, path).GetAllTensors()), "T5-XXL", path);
                Logs.Info($"[Sd3Recipe] T5-XXL resolved as a separate file: {path}.");
            }
            if (vaeWeights.Count == 0)
            {
                string path = ResolveComponent(SideModels.Sd35Vae, "VAE");
                (Dictionary<string, Tensor> staged, SafeTensorsLoader vaeLoader) = LoaderVaeUtils.LoadFluxVaeF32(path);
                loaders.Add(vaeLoader);
                vaeWeights = RequireWeights(staged, "VAE", path);
                Logs.Info($"[Sd3Recipe] VAE resolved as a separate file: {path}.");
            }

            // Merge BEFORE LoadWeights, not after: the merge swaps dictionary entries, and device caches are
            // identity-keyed, so a tensor already captured by a layer would keep serving its stale copy
            // (same ordering constraint as MiniMaxH3Recipe/WanVideoRecipe's LoRA merges).
            loraStack = LoraApplier.BuildAndApply(
                LoraResolver.Resolve(context.Loras),
                context.Backend,
                transformerWeights: converted.Transformer,
                clipLWeights: clipLWeights,
                clipGWeights: clipGWeights);

            int patchEmbedOutChannels = DetectPatchEmbedOutChannels(converted.Transformer);
            Sd3Config sd3Config = Sd3Config.FromWeightShape(patchEmbedOutChannels);
            Logs.Info($"[Sd3Recipe] Building MMDiT (patch_embed out_channels={patchEmbedOutChannels}).");
            Sd3Transformer transformer = new Sd3Transformer(sd3Config);
            transformer.LoadWeights(converted.Transformer);

            ClipTextEncoder clipL = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipL);
            clipL.LoadWeights(clipLWeights, prefix: "text_model");

            ClipTextEncoder clipG = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipG);
            clipG.LoadWeights(clipGWeights, prefix: "text_model");

            T5TextEncoder t5 = new T5TextEncoder(T5TextEncoderConfig.Xxl);
            t5.LoadWeights(t5Weights);
            T5Tokenizer t5Tokenizer = new T5Tokenizer(maxLength: 256);

            // BF16 on Ampere+ (F32-equivalent range, halves the full-res decode workspace), F32 otherwise —
            // the SDXL-VAE precision policy; LoadFluxVaeF32 force-upcasts to F32, this recovers BF16 where safe.
            vaeWeights = VaePrecisionHelper.CastVaeWeights(vaeWeights, VaePrecisionHelper.PreferredVaeDtype(context.Backend));
            VaeDecoder vaeDecoder = new VaeDecoder(VaeConfig.Sd3);
            vaeDecoder.LoadWeights(vaeWeights);
            VaeEncoder vaeEncoder = LoaderVaeUtils.BuildEncoder(VaeConfig.Sd3, vaeWeights, "Sd3Recipe");

            // Phase 8 DiT sharding (VRAM pooling, not latency — see Sd3Pipeline's use of DitShardBackend): the
            // split point depends on the transformer's actual block count and free VRAM on the two backends,
            // both only known here (RecipeContext is built before weights load), so it's computed at
            // construction time rather than passed in from InferenceEngine. SD3's JointBlocks are homogeneous
            // (dual-attention layers only add extra params, not extra width), so the count-proportional overload
            // applies — same as Krea2/MiniMax-H3.
            int ditShardSplitBlock = 0;
            if (context.DitShardBackend is not null)
            {
                long sharedWeightBytes = 0;
                foreach (Tensor t in transformer.EnumerateSharedWeights())
                {
                    sharedWeightBytes += t.DType.ComputeByteCount(t.ElementCount);
                }
                (long freeA, _) = context.Backend.GetVramInfo();
                (long freeB, _) = context.DitShardBackend.GetVramInfo();
                ditShardSplitBlock = PlacementPlanner.DitSplitPlan([freeA, freeB], transformer.BlockCount, sharedWeightBytes)[0];
                Logs.Info($"[Sd3Recipe] DiT sharding enabled: blocks [0,{ditShardSplitBlock}) on the primary "
                    + $"backend, [{ditShardSplitBlock},{transformer.BlockCount}) on the shard backend.");
            }

            Sd3Pipeline pipeline = new Sd3Pipeline(context.Backend, clipL, clipG, t5, transformer, vaeDecoder, vaeEncoder)
            {
                DitShardBackend = context.DitShardBackend,
                DitShardSplitBlock = ditShardSplitBlock,
            };
            Logs.Info("[Sd3Recipe] SD3 ready.");
            return new Sd3RecipePipeline(pipeline, new ClipTokenizer(), t5Tokenizer, loaders, loraStack);
        }
        catch (Exception ex)
        {
            Logs.Error("[Sd3Recipe] Construction failed.", ex);
            loraStack?.Dispose();
            foreach (SafeTensorsLoader loader in loaders)
            {
                loader.Dispose();
            }
            throw;
        }
    }

    /// <summary>Resolves a component the checkpoint didn't bundle to a file on disk, downloading it once when absent. Failure names the component, the exact path that was searched, and the HuggingFace source that was tried — a generic "missing components" message costs hours to diagnose.</summary>
    private static string ResolveComponent(ModelAsset asset, string component)
    {
        string target = ModelDownloader.TargetPath(asset);
        try
        {
            return ModelDownloader.EnsureSideModelAsync(asset, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
        }
        catch (Exception ex)
        {
            Logs.Error($"[Sd3Recipe] {component} could not be resolved as a separate file.", ex);
            throw new InvalidOperationException(
                $"SD3: this checkpoint does not bundle {component}, and it could not be resolved as a separate file. "
                + $"Looked for '{target}'; the download from HuggingFace '{asset.Repo}/{asset.RepoPath}' failed: {ex.Message}", ex);
        }
    }

    /// <summary>Opens a resolved component file and hands its loader to <paramref name="loaders"/> for disposal, registering before the load so a failed parse still gets cleaned up.</summary>
    private static SafeTensorsLoader OpenSide(List<SafeTensorsLoader> loaders, string path)
    {
        SafeTensorsLoader loader = new SafeTensorsLoader();
        loaders.Add(loader);
        loader.Load(path);
        return loader;
    }

    /// <summary>Guards against a file that opens but carries none of the expected tensors (wrong component picked, or a naming convention the strip helper doesn't cover).</summary>
    private static Dictionary<string, Tensor> RequireWeights(Dictionary<string, Tensor> weights, string component, string path)
    {
        if (weights.Count == 0)
        {
            throw new InvalidOperationException($"SD3: the {component} file '{path}' contains no usable {component} tensors.");
        }
        return weights;
    }

    /// <summary>SD3's MMDiT patch_embed.weight is [embedDim, 16, 2, 2]; the leading dim selects the arch (Medium 1536, 3.5 Large 2432). Reads the ComfyUI (<c>pos_embed.proj.weight</c>) or diffusers (<c>patch_embed.proj.weight</c>) key, falling back to Medium.</summary>
    private static int DetectPatchEmbedOutChannels(Dictionary<string, Tensor> transformer)
    {
        if (transformer.TryGetValue("pos_embed.proj.weight", out Tensor? t) && t.Shape.Rank >= 1)
        {
            return (int)t.Shape[0];
        }
        if (transformer.TryGetValue("patch_embed.proj.weight", out Tensor? t2) && t2.Shape.Rank >= 1)
        {
            return (int)t2.Shape[0];
        }
        return 1536;
    }

    /// <summary>Strips Comfy's <c>text_encoders.t5xxl.transformer.</c> prefix from a standalone T5-XXL safetensors file and drops the <c>position_ids</c> buffer the encoder doesn't consume.</summary>
    private static Dictionary<string, Tensor> StripT5Prefix(IReadOnlyDictionary<string, Tensor> raw)
    {
        Dictionary<string, Tensor> result = new Dictionary<string, Tensor>(raw.Count);
        const string ComfyPrefix = "text_encoders.t5xxl.transformer.";
        foreach (KeyValuePair<string, Tensor> kv in raw)
        {
            string key = kv.Key.StartsWith(ComfyPrefix, StringComparison.Ordinal) ? kv.Key[ComfyPrefix.Length..] : kv.Key;
            if (!key.EndsWith("position_ids", StringComparison.Ordinal))
            {
                result[key] = kv.Value;
            }
        }
        return result;
    }
}
