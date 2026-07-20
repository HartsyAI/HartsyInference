using HartsyInference.Core.Backends;
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

/// <summary>Z-Image recipe (Tongyi Lab NextDiT): the checkpoint carries the transformer only; the Qwen3-4B text encoder and the Flux VAE are resolved as side models. Lifted from the SwarmUI backend's <c>ZImageLoader</c>; constructs the components and drives generation through <see cref="ZImageRecipePipeline"/>.</summary>
public sealed class ZImageRecipe : IArchitectureRecipe
{
    /// <inheritdoc/>
    public string Name => "zimage";

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "zimage", StringComparison.OrdinalIgnoreCase);

    /// <inheritdoc/>
    public IRecipePipeline Construct(RecipeContext context)
    {
        // Resolve the two side models synchronously (Construct is a sync seam): Qwen3-4B text encoder + Flux VAE.
        // TODO(E-IMG-4): honor a user-picked Qwen/VAE override from ImageRequest.Components/Extra instead of always
        // taking the canonical SideModels entry (the SwarmUI loader read input.Get(T2IParamTypes.QwenModel/VAE)).
        string qwenPath = ModelDownloader.EnsureSideModelAsync(SideModels.Qwen3_4B, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
        string vaePath = ModelDownloader.EnsureSideModelAsync(SideModels.FluxAe, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();

        // 1. Load + convert the Z-Image transformer (checkpoint carries only these weights).
        (ZImageCheckpointConverter.ConvertedWeights zConv, SafeTensorsLoader zLoader) = ZImageCheckpointConverter.LoadAndConvert(context.CheckpointPath);
        if (zConv.Transformer.Count == 0)
        {
            zLoader.Dispose();
            throw new InvalidOperationException("Z-Image checkpoint has no transformer weights.");
        }
        ZImageConfig zConfig = ZImageConfig.FromWeights(zConv.Transformer);
        Logs.Info($"[ZImageRecipe] Building transformer (SchedulerShift={zConfig.SchedulerShift}).");
        ZImageTransformer transformer = new ZImageTransformer(zConfig);
        transformer.LoadWeights(zConv.Transformer);

        // 2. Load the Qwen3-4B encoder + its embedded tokenizer.
        SafeTensorsLoader qwenLoader = new SafeTensorsLoader();
        qwenLoader.Load(qwenPath);
        IReadOnlyDictionary<string, Tensor> qwenWeights = qwenLoader.GetAllTensors();
        if (qwenWeights.Count == 0)
        {
            qwenLoader.Dispose();
            zLoader.Dispose();
            throw new InvalidOperationException($"Qwen3 model file '{qwenPath}' has no tensors.");
        }
        LlamaStyleEncoder qwen = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Qwen3_4B);
        qwen.LoadWeights(qwenWeights);
        Qwen3Tokenizer tokenizer = new Qwen3Tokenizer(maxLength: 256);

        // 3. Load the Flux VAE (Z-Image reuses it verbatim). BF16 on Ampere+ (F32-equivalent range, halves the
        //    full-res decode workspace), F32 otherwise — the SDXL-VAE precision policy.
        SafeTensorsLoader vaeLoader = new SafeTensorsLoader();
        vaeLoader.Load(vaePath);
        Dictionary<string, Tensor> vaeWeights = LoadVaeWeights(vaeLoader.GetAllTensors());
        if (vaeWeights.Count == 0)
        {
            vaeLoader.Dispose();
            qwenLoader.Dispose();
            zLoader.Dispose();
            throw new InvalidOperationException($"VAE file '{vaePath}' has no usable VAE tensors.");
        }
        DType vaeDtype = context.Backend.Capabilities.SupportsBF16 ? DType.BF16 : DType.F32;
        vaeWeights = CastWeights(vaeWeights, vaeDtype);
        VaeDecoder vae = new VaeDecoder(VaeConfig.ZImage);
        vae.LoadWeights(vaeWeights);
        VaeEncoder vaeEncoder = new VaeEncoder(VaeConfig.ZImage);
        vaeEncoder.LoadWeights(vaeWeights);

        ZImagePipeline pipeline = new ZImagePipeline(context.Backend, transformer, vae, vaeEncoder, zConfig);
        Logs.Info("[ZImageRecipe] Z-Image ready.");
        return new ZImageRecipePipeline(pipeline, qwen, tokenizer, context.Backend, zLoader, qwenLoader, vaeLoader);
    }

    /// <summary>Normalizes a standalone Flux/Z-Image VAE safetensors file into the diffusers key naming the VAE decoder expects (strips a Comfy prefix, then routes every key through <see cref="CheckpointConvertUtils.ConvertVaeKey"/> which maps LDM names and passes already-diffusers names through unchanged).</summary>
    private static Dictionary<string, Tensor> LoadVaeWeights(IReadOnlyDictionary<string, Tensor> raw)
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

    /// <summary>Casts a VAE weight dictionary to <paramref name="target"/>, leaving tensors already at that dtype untouched (the SDXL-VAE BF16/F32 precision policy — never F16, which overflows the SDXL/Flux VAE resnets).</summary>
    private static Dictionary<string, Tensor> CastWeights(Dictionary<string, Tensor> weights, DType target)
    {
        Dictionary<string, Tensor> result = new Dictionary<string, Tensor>(weights.Count);
        foreach (KeyValuePair<string, Tensor> kv in weights)
        {
            result[kv.Key] = kv.Value.DType == target ? kv.Value : kv.Value.CastTo(target);
        }
        return result;
    }
}
