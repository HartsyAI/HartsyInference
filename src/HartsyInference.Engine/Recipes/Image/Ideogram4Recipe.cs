using HartsyInference.Core.Logging;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

using HartsyInference.Engine.Features;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Ideogram 4 recipe (ideogram-oss/ideogram4, 9.3B): a Qwen3-VL-8B language tower (13-layer hidden-state tap) feeding TWO 9.3B single-stream DiTs — the picked checkpoint is the conditional one, its unconditional companion (<see cref="SideModels.Ideogram4Unconditional"/>) runs the asymmetric-CFG branch — decoded by the Flux.2 VAE (<see cref="SideModels.Flux2Vae"/>). Lifted from the SwarmUI backend's <c>Ideogram4Loader</c>, including its free-VRAM gate (both DiTs stay resident for the whole denoise loop). Constructs and drives through <see cref="Ideogram4RecipePipeline"/>. Weights are under a NON-COMMERCIAL license.</summary>
public sealed class Ideogram4Recipe : IArchitectureRecipe
{
    /// <summary>Minimum free VRAM to attempt a CUDA load: with nvfp4→fp8 dequant both 9.3B DiTs are ~18.6 GB resident, and per-step activation frees cap the working set at a few GB.</summary>
    private const double MinRequiredVramGb = 20.0;

    /// <inheritdoc/>
    public string Name => "ideogram4";


    /// <inheritdoc/>
    /// <remarks>Ideogram 4 shares the Flux.2 VAE; its packed-latent mask blend uses the channel-inner variant.</remarks>
    public ImageFeatures Supports => ImageFeatures.Img2Img | ImageFeatures.Inpaint | ImageFeatures.Regional | ImageFeatures.SeamlessTiling;
    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "ideogram4", StringComparison.OrdinalIgnoreCase);

    /// <summary>Ideogram 4's official sampling settings: 20 steps, 1024x1024. The step count picks the nearest official sampler preset, which carries its own per-step asymmetric-CFG schedule, so the guidance value here is inert.</summary>
    public static ImageDefaults FamilyDefaults { get; } = new ImageDefaults { Steps = 20, CfgScale = 7.0f, Width = 1024, Height = 1024 };

    /// <inheritdoc/>
    public ImageDefaults Defaults => FamilyDefaults;

    /// <inheritdoc/>
    public IRecipePipeline Construct(RecipeContext context)
    {
        // Gate BEFORE the multi-minute weight load, not after.
        //
        // This used to be an unconditional ">= 20 GB free or refuse" check, because both 9.3B transformers were held
        // resident for the whole denoise loop. Ideogram4Pipeline now block-streams them (two independent sliding
        // windows), so the resident set is the shared weights plus a couple of blocks — the 20 GB figure only
        // describes the FULLY-RESIDENT layout. Refusing on it made a 12 GB card report an out-of-memory failure that
        // never attempted an allocation (measured: 1 s, 82 MiB peak), which read as a capacity limit in benchmark
        // results when it was a policy.
        //
        // The threshold is therefore now only enforced when streaming is unavailable — HARTSY_LOWVRAM=off, or a
        // backend with no streaming cache. Otherwise the planner decides per generation against real free VRAM.
        if (context.Backend is CudaBackend cuda)
        {
            (nuint freeBytes, nuint totalBytes) = cuda.Context.GetMemoryInfo();
            double freeGb = freeBytes / (1024.0 * 1024.0 * 1024.0);
            bool canStream = cuda.StreamingCache is not null && LowVramPolicy.Resolve(cuda) != LowVramMode.ForceOff;
            if (freeGb < MinRequiredVramGb && !canStream)
            {
                throw new InvalidOperationException(
                    $"Ideogram 4 needs >= {MinRequiredVramGb:F0} GB free VRAM to hold both 9.3B transformers resident for asymmetric CFG; " +
                    $"this GPU has {freeGb:F1} GB free of {totalBytes / (1024.0 * 1024.0 * 1024.0):F1} GB total, and weight streaming is " +
                    $"disabled ({LowVramPolicy.EnvironmentVariable}=off). Unset {LowVramPolicy.EnvironmentVariable} to let the engine " +
                    "stream the transformers, or use a higher-VRAM GPU.");
            }
            if (freeGb < MinRequiredVramGb)
            {
                Logs.Info($"[Ideogram4Recipe] {freeGb:F1} GB free — under the {MinRequiredVramGb:F0} GB fully-resident " +
                    "requirement, so the denoise loop will stream both transformers' blocks (slower, but it fits).");
            }
        }
        else
        {
            Logs.Warning("[Ideogram4Recipe] Non-CUDA backend — two 9.3B transformers per step will be extremely slow.");
        }
        Logs.Info("[Ideogram4Recipe] Note: Ideogram 4 weights are under a NON-COMMERCIAL license.");

        // TODO(E-IMG-4): honor user-picked Qwen3-VL / VAE overrides from ImageRequest.Components (the SwarmUI loader
        // read T2IParamTypes.QwenModel/VAE and header-probed the pick, falling back to the canonical component).
        string uncondPath = ModelDownloader.EnsureSideModelAsync(SideModels.Ideogram4Unconditional, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
        string encoderPath = ModelDownloader.EnsureSideModelAsync(SideModels.Qwen3VL_8B, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
        string vaePath = ModelDownloader.EnsureSideModelAsync(SideModels.Flux2Vae, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();

        List<SafeTensorsLoader> loaders = new List<SafeTensorsLoader>();
        try
        {
            // nvfp4ToFp8: the DiTs are ~93% nvfp4. Dequantizing to F16 would need 35.9 GB for the pair; folding the
            // block scale into an fp8 value (global scale on Fp8ScaleFactor) keeps them at 9.3 GB each.
            Logs.Info($"[Ideogram4Recipe] Loading conditional transformer (9.3B, nvfp4->fp8): {Path.GetFileName(context.CheckpointPath)}.");
            Dictionary<string, Tensor> condWeights = LoadComponent(context.CheckpointPath, StripTransformerPrefix, applyFp8Dequant: true, nvfp4ToFp8: true, loaders);

            Logs.Info($"[Ideogram4Recipe] Loading unconditional transformer (9.3B, nvfp4->fp8): {Path.GetFileName(uncondPath)}.");
            Dictionary<string, Tensor> uncondWeights = LoadComponent(uncondPath, StripTransformerPrefix, applyFp8Dequant: true, nvfp4ToFp8: true, loaders);

            Logs.Info($"[Ideogram4Recipe] Loading Qwen3-VL-8B text encoder: {Path.GetFileName(encoderPath)}.");
            Dictionary<string, Tensor> encoderWeights = LoadComponent(encoderPath, RemapQwenKey, applyFp8Dequant: true, nvfp4ToFp8: false, loaders);

            Logs.Info($"[Ideogram4Recipe] Loading Flux.2 VAE: {Path.GetFileName(vaePath)}.");
            Dictionary<string, Tensor> vaeWeights = LoadComponent(vaePath, key => key, applyFp8Dequant: false, nvfp4ToFp8: false, loaders);

            Ideogram4Config config = Ideogram4Config.V4;
            Logs.Info($"[Ideogram4Recipe] Building models (dim={config.LlmFeaturesDim} LLM features, {config.MaxTextTokens} max text tokens).");

            Ideogram4Transformer conditional = new Ideogram4Transformer(config);
            conditional.LoadWeights(condWeights);
            Ideogram4Transformer unconditional = new Ideogram4Transformer(config);
            unconditional.LoadWeights(uncondWeights);

            LlamaStyleEncoder textEncoder = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Qwen3_VL_8B);
            textEncoder.LoadWeights(encoderWeights);

            VaeDecoder vae = new VaeDecoder(VaeConfig.Flux2);
            vae.LoadWeights(vaeWeights);
            VaeEncoder? vaeEncoder = LoaderVaeUtils.TryBuildEncoder(VaeConfig.Flux2, vaeWeights, "Ideogram4Recipe");

            Qwen3Tokenizer tokenizer = new Qwen3Tokenizer(maxLength: config.MaxTextTokens);
            Ideogram4Pipeline pipeline = new Ideogram4Pipeline(context.Backend, textEncoder, conditional, unconditional, vae, vaeEncoder, config);
            Logs.Info("[Ideogram4Recipe] Ideogram 4 ready.");
            return new Ideogram4RecipePipeline(pipeline, tokenizer, textEncoder, conditional, unconditional, loaders);
        }
        catch (Exception ex)
        {
            Logs.Error("[Ideogram4Recipe] Construction failed.", ex);
            foreach (SafeTensorsLoader loader in loaders)
            {
                loader.Dispose();
            }
            throw;
        }
    }

    /// <summary>Loads one component from a single safetensors file: drops fp8 <c>scaled_fp8</c> markers, applies <paramref name="keyTransform"/> (null return drops the key), then optionally folds fp8 <c>*.scale_weight</c> companions. The registered loader owns the tensor memory and must outlive the weights.</summary>
    private static Dictionary<string, Tensor> LoadComponent(string filePath, Func<string, string?> keyTransform, bool applyFp8Dequant, bool nvfp4ToFp8, List<SafeTensorsLoader> loaders)
    {
        SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(filePath);
        loaders.Add(loader);
        Dictionary<string, Tensor> merged = new Dictionary<string, Tensor>();
        foreach (KeyValuePair<string, Tensor> kv in loader.GetAllTensors())
        {
            if (kv.Key.EndsWith(".scaled_fp8", StringComparison.Ordinal) || kv.Key == "scaled_fp8")
            {
                continue;
            }
            string? mapped = keyTransform(kv.Key);
            if (mapped is not null)
            {
                merged[mapped] = kv.Value;
            }
        }
        return applyFp8Dequant ? CheckpointConvertUtils.ApplyFp8ScaledDequant(merged, nvfp4ToFp8) : merged;
    }

    /// <summary>Strips an optional Comfy/diffusers wrapper prefix off a transformer key.</summary>
    private static string StripTransformerPrefix(string key)
    {
        if (key.StartsWith("model.diffusion_model.", StringComparison.Ordinal))
        {
            return key["model.diffusion_model.".Length..];
        }
        if (key.StartsWith("diffusion_model.", StringComparison.Ordinal))
        {
            return key["diffusion_model.".Length..];
        }
        if (key.StartsWith("transformer.", StringComparison.Ordinal))
        {
            return key["transformer.".Length..];
        }
        return key;
    }

    /// <summary>Maps a Qwen3-VL checkpoint key to the <see cref="LlamaStyleEncoder"/> convention (<c>model.embed_tokens</c>, <c>model.layers.{i}.*</c>, <c>model.norm</c>), re-rooting the HF <c>language_model.</c> tower at <c>model.</c>; returns null to drop the vision tower and <c>lm_head</c>.</summary>
    private static string? RemapQwenKey(string key)
    {
        if (key.Contains(".visual.", StringComparison.Ordinal) || key.StartsWith("visual.", StringComparison.Ordinal))
        {
            return null;
        }
        if (key.Contains("lm_head", StringComparison.Ordinal))
        {
            return null;
        }
        int lm = key.LastIndexOf("language_model.", StringComparison.Ordinal);
        string suffix = lm >= 0 ? key[(lm + "language_model.".Length)..] : key;
        if (suffix.StartsWith("model.", StringComparison.Ordinal))
        {
            suffix = suffix["model.".Length..];
        }
        if (suffix.StartsWith("layers.", StringComparison.Ordinal)
            || suffix.StartsWith("embed_tokens.", StringComparison.Ordinal)
            || suffix == "norm.weight")
        {
            return "model." + suffix;
        }
        return null;
    }
}
