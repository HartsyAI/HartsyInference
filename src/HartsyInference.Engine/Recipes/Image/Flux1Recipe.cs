using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.CheckpointConverters.Utils;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Flux.1 recipe (BFL 12B single-stream DiT, flow matching, Dev with distilled guidance or Schnell distilled 1-4 step). Lifted from the SwarmUI backend's <c>FluxLoader</c>: the checkpoint is loaded all-in-one via <see cref="FluxCheckpointConverter"/>; any component the file omits (a Comfy transformer-only <c>diffusion_models</c> layout) is resolved from the canonical side models — CLIP-L (<see cref="SideModels.ClipL"/>), T5-XXL (<see cref="SideModels.T5XxlEnconly"/>), Flux VAE (<see cref="SideModels.FluxAe"/>). Constructs and drives the text-to-image core through <see cref="Flux1RecipePipeline"/>.</summary>
public sealed class Flux1Recipe : IArchitectureRecipe
{
    /// <inheritdoc/>
    public string Name => "flux1";

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "flux1", StringComparison.OrdinalIgnoreCase);

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
                SafeTensorsLoader vaeLoader = new SafeTensorsLoader();
                vaeLoader.Load(vaePath);
                loaders.Add(vaeLoader);
                vaeWeights = ConvertVaeFromStandalone(vaeLoader.GetAllTensors());
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
            if (context.Backend is HartsyInference.Cuda.CudaBackend cudaBackend && !cudaBackend.EnableNativeFp8Gemm
                && (HasFp8Weights(transformerWeights) || HasFp8Weights(clipLWeights) || HasFp8Weights(t5Weights)))
            {
                cudaBackend.CacheWeightCasts = false;
                Logs.Info("[Flux1Recipe] fp8 checkpoint on non-native-FP8 hardware: CacheWeightCasts disabled (fp8-resident, transient per-GEMM dequant).");
            }

            FluxTransformer transformer = new FluxTransformer(config);
            transformer.LoadWeights(transformerWeights);

            ClipTextEncoder clipL = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipL);
            clipL.LoadWeights(clipLWeights, prefix: "text_model");

            T5TextEncoder t5 = new T5TextEncoder(T5TextEncoderConfig.Xxl);
            t5.LoadWeights(t5Weights);

            VaeDecoder vae = new VaeDecoder(VaeConfig.Flux);
            vae.LoadWeights(vaeWeights);

            FluxPipeline pipeline = new FluxPipeline(context.Backend, clipL, t5, transformer, vae, config);
            ClipTokenizer clipTok = new ClipTokenizer();
            T5Tokenizer t5Tok = new T5Tokenizer(maxLength: hasGuidance ? 512 : 256);
            Logs.Info($"[Flux1Recipe] Flux.1 ready (Dev={hasGuidance}).");
            return new Flux1RecipePipeline(pipeline, clipTok, t5Tok, hasGuidance, loaders);
        }
        catch (Exception ex)
        {
            Logs.Error("[Flux1Recipe] Construction failed.", ex);
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

    /// <summary>Normalizes a standalone Flux VAE file into diffusers key naming: strips a Comfy prefix, then routes every key through <see cref="CheckpointConvertUtils.ConvertVaeKey"/> (maps LDM-native names and passes diffusers names through unchanged).</summary>
    private static Dictionary<string, Tensor> ConvertVaeFromStandalone(IReadOnlyDictionary<string, Tensor> raw)
    {
        Dictionary<string, Tensor> result = new Dictionary<string, Tensor>(raw.Count);
        foreach (KeyValuePair<string, Tensor> kv in raw)
        {
            string ldmKey = kv.Key;
            if (ldmKey.StartsWith("first_stage_model.", StringComparison.Ordinal))
            {
                ldmKey = ldmKey["first_stage_model.".Length..];
            }
            else if (ldmKey.StartsWith("vae.", StringComparison.Ordinal))
            {
                ldmKey = ldmKey["vae.".Length..];
            }
            string? diffusersKey = CheckpointConvertUtils.ConvertVaeKey(ldmKey);
            if (diffusersKey is not null)
            {
                result[diffusersKey] = kv.Value;
            }
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
