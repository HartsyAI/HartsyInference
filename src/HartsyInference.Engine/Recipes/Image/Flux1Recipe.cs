using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Placement;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Flux.1 recipe (BFL 12B single-stream DiT, flow matching, Dev with distilled guidance or Schnell distilled 1-4 step). Lifted from the SwarmUI backend's <c>FluxLoader</c>: the checkpoint is loaded all-in-one via <see cref="FluxCheckpointConverter"/>; any component the file omits (a Comfy transformer-only <c>diffusion_models</c> layout) is resolved from the canonical side models — CLIP-L (<see cref="SideModels.ClipL"/>), T5-XXL (<see cref="SideModels.T5XxlEnconly"/>), Flux VAE (<see cref="SideModels.FluxAe"/>). Constructs and drives the text-to-image core through <see cref="Flux1RecipePipeline"/>.</summary>
public sealed class Flux1Recipe : IArchitectureRecipe
{
    /// <inheritdoc/>
    public string Name => "flux1";

    /// <inheritdoc/>
    /// <remarks>Wiring the VAE encoder also makes FLUX.1 Fill reachable: <see cref="FluxPipeline"/> requires a mask for
    /// checkpoints whose <c>x_embedder</c> is ≥384 channels, which could never arrive while img2img was unmapped.
    /// IpAdapter here means the image-PROMPT slot (FLUX.1 Redux via <c>redux.stylemodel</c>) — a real IP-Adapter
    /// checkpoint (<c>ipadapter.model</c>) stays SD15/SDXL-only and is refused in the recipe pipeline.</remarks>
    public ImageFeatures Supports =>
        ImageFeatures.Lora | ImageFeatures.ControlNet | ImageFeatures.VariationSeed
        | ImageFeatures.Img2Img | ImageFeatures.Inpaint | ImageFeatures.IpAdapter;

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "flux1", StringComparison.OrdinalIgnoreCase);

    /// <summary>Flux.1 Dev's official sampling settings: 28 steps at distilled guidance 3.5, 1024x1024 (<c>GenerationDefaults.FluxDev</c>); a Schnell checkpoint narrows this to 4 steps via <see cref="Flux1RecipePipeline.VariantDefaults"/>.</summary>
    public static ImageDefaults FamilyDefaults { get; } = new ImageDefaults { Steps = 28, CfgScale = 3.5f, Width = 1024, Height = 1024 };

    /// <inheritdoc/>
    public ImageDefaults Defaults => FamilyDefaults;

    /// <summary>Flux.1 Schnell's official sampling settings: 4 steps, guidance-free by distillation (<c>GenerationDefaults.FluxSchnell</c>).</summary>
    public static ImageDefaults SchnellDefaults { get; } = new ImageDefaults { Steps = 4, CfgScale = 3.5f, Width = 1024, Height = 1024 };

    /// <inheritdoc/>
    public IRecipePipeline Construct(RecipeContext context)
    {
        // TODO(E-IMG-4/5): split-file + GGUF transformer inputs, and honoring user CLIP-L/T5/VAE overrides from
        // ImageRequest.Components (the SwarmUI loader read T2IParamTypes.ClipLModel/T5XXLModel/VAE), are not yet
        // wired — this takes the all-in-one path with canonical side-model fallback for any missing component.
        // TODO(E-IMG-4/5): FLUX.1 Tools (Canny/Depth/Fill), Kontext, Redux, ControlNet, img2img/inpaint deferred —
        // this ports the vanilla Dev/Schnell text-to-image core only.
        (FluxCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader mainLoader) = FluxCheckpointConverter.LoadAndConvert(context.CheckpointPath);
        Dictionary<string, Tensor> transformerWeights = converted.Transformer;
        if (transformerWeights.Count == 0)
        {
            mainLoader.Dispose();
            throw new InvalidOperationException("Flux.1: this checkpoint contains no transformer weights — not a Flux diffusion model.");
        }

        List<SafeTensorsLoader> loaders = new List<SafeTensorsLoader> { mainLoader };
        MergedLoraStack? loraStack = null;
        try
        {
            Dictionary<string, Tensor> clipLWeights = converted.ClipL;
            Dictionary<string, Tensor> t5Weights = converted.T5;
            Dictionary<string, Tensor> vaeWeights = converted.Vae;

            if (clipLWeights.Count == 0)
            {
                string clipLPath = ModelDownloader.EnsureSideModelAsync(SideModels.ClipL, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
                SafeTensorsLoader clipLLoader = new SafeTensorsLoader();
                clipLLoader.Load(clipLPath);
                loaders.Add(clipLLoader);
                clipLWeights = ConvertClipLFromStandalone(clipLLoader.GetAllTensors());
                Logs.Info("[Flux1Recipe] CLIP-L resolved as side model.");
            }
            if (t5Weights.Count == 0)
            {
                string t5Path = ModelDownloader.EnsureSideModelAsync(SideModels.T5XxlEnconly, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
                SafeTensorsLoader t5Loader = new SafeTensorsLoader();
                t5Loader.Load(t5Path);
                loaders.Add(t5Loader);
                t5Weights = StripT5Prefix(t5Loader.GetAllTensors());
                Logs.Info("[Flux1Recipe] T5-XXL resolved as side model.");
            }
            if (vaeWeights.Count == 0)
            {
                string vaePath = ModelDownloader.EnsureSideModelAsync(SideModels.FluxAe, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
                (vaeWeights, SafeTensorsLoader vaeLoader) = LoaderVaeUtils.LoadFluxVaeF32(vaePath);
                loaders.Add(vaeLoader);
                Logs.Info("[Flux1Recipe] Flux VAE resolved as side model.");
            }
            if (clipLWeights.Count == 0 || t5Weights.Count == 0 || vaeWeights.Count == 0)
            {
                throw new InvalidOperationException("Flux.1: could not assemble CLIP-L / T5-XXL / VAE components for this checkpoint.");
            }

            (int doubleBlocks, int singleBlocks, bool hasGuidance) = FluxCheckpointConverter.DetectArchitecture(transformerWeights);
            FluxConfig config = hasGuidance ? FluxConfig.Dev : FluxConfig.Schnell;
            Logs.Info($"[Flux1Recipe] Architecture: {doubleBlocks} double, {singleBlocks} single, guidance={hasGuidance} ({(hasGuidance ? "Dev" : "Schnell")}).");

            // fp8 checkpoints on hardware without native FP8 GEMM (Ampere and below) cast each fp8 weight to F16/BF16
            // on first use; caching those casts roughly doubles VRAM per fp8 tensor and OOMs a 12 GB card partway
            // through Flux text encoding. Keep the weights fp8-resident with a transient per-GEMM dequant instead.
            // Applied to EVERY placement backend: with the T5 on its own device, that device's backend needs the
            // same flag or it re-inflates the fp8 encoder there.
            if (HasFp8Weights(transformerWeights) || HasFp8Weights(clipLWeights) || HasFp8Weights(t5Weights))
            {
                RecipeBackendFlags.DisableCacheWeightCasts(context, "Flux1Recipe", onlyWithoutNativeFp8Gemm: true);
            }

            loraStack = LoraApplier.BuildAndApply(
                LoraResolver.Resolve(context.Loras),
                context.Backend,
                transformerWeights: transformerWeights,
                clipLWeights: clipLWeights);

            FluxTransformer transformer = new FluxTransformer(config);
            transformer.LoadWeights(transformerWeights);

            // DiT sharding split point — byte-weighted: Flux's 19 double blocks are ~2× its 38 single blocks,
            // so a count-proportional split would misallocate by GBs. Computed post-load (needs live free VRAM).
            int ditShardSplitBlock = 0;
            if (context.DitShardBackend is not null)
            {
                long[] perBlockBytes = new long[transformer.BlockCount];
                for (int i = 0; i < perBlockBytes.Length; i++)
                {
                    foreach (Tensor t in transformer.EnumerateBlockRangeWeights(i, i + 1))
                    {
                        perBlockBytes[i] += t.DType.ComputeByteCount(t.ElementCount);
                    }
                }
                long sharedWeightBytes = 0;
                foreach (Tensor t in transformer.EnumerateSharedWeights())
                {
                    sharedWeightBytes += t.DType.ComputeByteCount(t.ElementCount);
                }
                (long freeA, _) = context.Backend.GetVramInfo();
                (long freeB, _) = context.DitShardBackend.GetVramInfo();
                ditShardSplitBlock = PlacementPlanner.DitSplitPlan([freeA, freeB], perBlockBytes, sharedWeightBytes)[0];
                Logs.Info($"[Flux1Recipe] DiT sharding enabled: blocks [0,{ditShardSplitBlock}) on the primary "
                    + $"backend, [{ditShardSplitBlock},{transformer.BlockCount}) on the shard backend (v1 plain path; "
                    + "ControlNet/Kontext/inpaint/regional generations fall back to unsharded).");
            }

            ClipTextEncoder clipL = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipL);
            clipL.LoadWeights(clipLWeights, prefix: "text_model");

            T5TextEncoder t5 = new T5TextEncoder(T5TextEncoderConfig.Xxl);
            t5.LoadWeights(t5Weights);

            VaeDecoder vae = new VaeDecoder(VaeConfig.Flux);
            vae.LoadWeights(vaeWeights);

                        VaeEncoder? vaeEncoder = LoaderVaeUtils.TryBuildEncoder(VaeConfig.Flux, vaeWeights, "Flux1Recipe");
            FluxPipeline pipeline = new FluxPipeline(context.Backend, clipL, t5, transformer, vae, vaeEncoder, config)
            {
                TextEncoderBackend = context.TextEncoderBackendOrDefault,
                VaeBackend = context.VaeBackendOrDefault,
                // Was missing until 2026-08-04 — the pipeline's CFG-parallel path existed but the recipe never
                // forwarded the backend, so a configured CfgParallelDevice silently did nothing for Flux
                // (caught by FluxCfgParallelFallbackTests asserting a [CfgParallel] decision is always logged).
                CfgParallelBackend = context.CfgParallelBackend,
                DitShardBackend = context.DitShardBackend,
                DitShardSplitBlock = ditShardSplitBlock,
            };
            ClipTokenizer clipTok = new ClipTokenizer();
            T5Tokenizer t5Tok = new T5Tokenizer(maxLength: hasGuidance ? 512 : 256);
            Logs.Info($"[Flux1Recipe] Flux.1 ready (Dev={hasGuidance}).");
            return new Flux1RecipePipeline(pipeline, clipTok, t5Tok, hasGuidance, loaders, loraStack, context.Backend);
        }
        catch (Exception ex)
        {
            Logs.Error("[Flux1Recipe] Construction failed.", ex);
            loraStack?.Dispose();
            foreach (SafeTensorsLoader loader in loaders)
            {
                loader.Dispose();
            }
            throw;
        }
    }

    /// <summary>Strips a standalone CLIP-L safetensors file down to the <c>text_model.*</c> keys the encoder expects (Comfy <c>text_encoders.clip_l.transformer.</c> or LDM <c>conditioner.embedders.0.transformer.</c> wrapping), dropping the <c>position_ids</c> buffer.</summary>
    private static Dictionary<string, Tensor> ConvertClipLFromStandalone(IReadOnlyDictionary<string, Tensor> raw)
    {
        Dictionary<string, Tensor> result = new Dictionary<string, Tensor>(raw.Count);
        foreach (KeyValuePair<string, Tensor> kv in raw)
        {
            if (kv.Key.StartsWith("text_encoders.clip_l.transformer.", StringComparison.Ordinal))
            {
                string rest = kv.Key["text_encoders.clip_l.transformer.".Length..];
                if (!rest.EndsWith("position_ids", StringComparison.Ordinal))
                {
                    result[rest] = kv.Value;
                }
            }
            else if (kv.Key.StartsWith("conditioner.embedders.0.transformer.", StringComparison.Ordinal))
            {
                string rest = kv.Key["conditioner.embedders.0.transformer.".Length..];
                if (!rest.EndsWith("position_ids", StringComparison.Ordinal))
                {
                    result[rest] = kv.Value;
                }
            }
            else
            {
                result[kv.Key] = kv.Value;
            }
        }
        return result;
    }

    /// <summary>Strips Comfy's <c>text_encoders.t5xxl.transformer.</c> prefix from a standalone T5-XXL safetensors file.</summary>
    private static Dictionary<string, Tensor> StripT5Prefix(IReadOnlyDictionary<string, Tensor> raw)
    {
        Dictionary<string, Tensor> result = new Dictionary<string, Tensor>(raw.Count);
        const string ComfyPrefix = "text_encoders.t5xxl.transformer.";
        foreach (KeyValuePair<string, Tensor> kv in raw)
        {
            string key = kv.Key.StartsWith(ComfyPrefix, StringComparison.Ordinal) ? kv.Key[ComfyPrefix.Length..] : kv.Key;
            result[key] = kv.Value;
        }
        return result;
    }

    /// <summary>True if any tensor is still raw FP8 (E4M3/E5M2) — i.e. not pre-dequantized by the converter's scaled-fp8 handling.</summary>
    private static bool HasFp8Weights(Dictionary<string, Tensor> weights)
    {
        foreach (Tensor tensor in weights.Values)
        {
            if (tensor.DType.IsFp8)
            {
                return true;
            }
        }
        return false;
    }
}
