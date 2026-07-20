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

/// <summary>Chroma recipe (Lodestone Rock's 8.9B Flux derivative: T5-only, no CLIP/pooled conditioning, joint-attention DiT, same VAE as Flux.1). Lifted from the SwarmUI backend's <c>ChromaLoader</c>; the checkpoint is the DiT, the T5-XXL text encoder and Flux VAE are resolved as side models. Constructs and drives through <see cref="ChromaRecipePipeline"/>.</summary>
public sealed class ChromaRecipe : IArchitectureRecipe
{
    /// <inheritdoc/>
    public string Name => "chroma";

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "chroma", StringComparison.OrdinalIgnoreCase);

    /// <inheritdoc/>
    public IRecipePipeline Construct(RecipeContext context)
    {
        // Resolve the two side models synchronously (Construct is a sync seam): T5-XXL (encoder-only fp8) + Flux VAE.
        // TODO(E-IMG-4): honor a user-picked T5/VAE override from ImageRequest.Components/Extra instead of always
        // taking the canonical SideModels entry (the SwarmUI loader read input.Get(T2IParamTypes.T5XXLModel/VAE)).
        string t5Path = ModelDownloader.EnsureSideModelAsync(SideModels.T5XxlEnconly, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
        string vaePath = ModelDownloader.EnsureSideModelAsync(SideModels.FluxAe, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();

        // 1. Load + convert the Chroma transformer.
        (ChromaCheckpointConverter.ConvertedWeights zConv, SafeTensorsLoader zLoader) = ChromaCheckpointConverter.LoadAndConvert(context.CheckpointPath);
        if (zConv.Transformer.Count == 0)
        {
            zLoader.Dispose();
            throw new InvalidOperationException("Chroma checkpoint has no recognized transformer weights after conversion.");
        }
        ChromaConfig config = ChromaConfig.V1;
        Logs.Info($"[ChromaRecipe] Building transformer ({config.HiddenSize} hidden, {config.Depth} double / {config.DepthSingleBlocks} single).");
        ChromaTransformer transformer = new ChromaTransformer(config);
        transformer.LoadWeights(zConv.Transformer);

        // 2. Load T5-XXL + its embedded tokenizer.
        SafeTensorsLoader t5Loader = new SafeTensorsLoader();
        t5Loader.Load(t5Path);
        Dictionary<string, Tensor> t5Weights = LoadT5FromStandalone(t5Loader.GetAllTensors());
        if (t5Weights.Count == 0)
        {
            t5Loader.Dispose();
            zLoader.Dispose();
            throw new InvalidOperationException($"T5 model file '{t5Path}' has no usable T5 tensors.");
        }
        T5TextEncoder t5 = new T5TextEncoder(T5TextEncoderConfig.Xxl);
        t5.LoadWeights(t5Weights);
        T5Tokenizer tokenizer = new T5Tokenizer(maxLength: 512);

        // 3. Load the Flux VAE (Chroma reuses it verbatim).
        SafeTensorsLoader vaeLoader = new SafeTensorsLoader();
        vaeLoader.Load(vaePath);
        Dictionary<string, Tensor> vaeWeights = LoadVaeFromStandalone(vaeLoader.GetAllTensors());
        if (vaeWeights.Count == 0)
        {
            vaeLoader.Dispose();
            t5Loader.Dispose();
            zLoader.Dispose();
            throw new InvalidOperationException($"VAE file '{vaePath}' has no usable VAE tensors.");
        }
        VaeDecoder vae = new VaeDecoder(VaeConfig.Chroma);
        vae.LoadWeights(vaeWeights);

        ChromaPipeline pipeline = new ChromaPipeline(context.Backend, t5, transformer, vae, config);
        Logs.Info("[ChromaRecipe] Chroma ready.");
        return new ChromaRecipePipeline(pipeline, tokenizer, zLoader, t5Loader, vaeLoader);
    }

    /// <summary>Strips Comfy's <c>text_encoders.t5xxl.transformer.</c> prefix from a standalone T5-XXL safetensors file (keys are otherwise stored as-is), so the encoder finds them.</summary>
    private static Dictionary<string, Tensor> LoadT5FromStandalone(IReadOnlyDictionary<string, Tensor> raw)
    {
        Dictionary<string, Tensor> result = new Dictionary<string, Tensor>(raw.Count);
        const string ComfyPrefix = "text_encoders.t5xxl.transformer.";
        foreach (KeyValuePair<string, Tensor> kv in raw)
        {
            if (kv.Key.StartsWith(ComfyPrefix, StringComparison.Ordinal))
            {
                result[kv.Key[ComfyPrefix.Length..]] = kv.Value;
            }
            else
            {
                result[kv.Key] = kv.Value;
            }
        }
        return result;
    }

    /// <summary>Normalizes a standalone Flux VAE safetensors file into the diffusers key naming the VAE decoder expects (strips a Comfy prefix, then routes every key through <see cref="CheckpointConvertUtils.ConvertVaeKey"/> which maps LDM names and passes already-diffusers names through unchanged).</summary>
    private static Dictionary<string, Tensor> LoadVaeFromStandalone(IReadOnlyDictionary<string, Tensor> raw)
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
}
