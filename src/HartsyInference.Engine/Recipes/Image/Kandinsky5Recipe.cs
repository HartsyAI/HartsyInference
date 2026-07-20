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

/// <summary>Kandinsky 5.0 T2I-Lite recipe (kandinskylab, ~6B DiT): the checkpoint is the transformer — either a single repackaged safetensors or a diffusers <c>transformer/</c> shard directory — and the dual text stack (Qwen2.5-VL-7B sequence embeddings via <see cref="SideModels.Qwen2_5_VL_7B"/> + CLIP-L pooled via <see cref="SideModels.ClipL"/>) plus the 16-channel Flux VAE (<see cref="SideModels.FluxAe"/>) resolve as side models.
/// <para>Unlike the other lifted families this one has NO SwarmUI loader to port — the extension lists Kandinsky 5 as unsupported because <see cref="Kandinsky5Pipeline"/> only accepts PRE-COMPUTED embeddings. Construction here follows the pipeline's own ctor and the <c>Kandinsky5GenerationTests</c> wiring; the live encode in <see cref="Kandinsky5RecipePipeline"/> is ported from the diffusers reference (<c>pipeline_kandinsky_t2i.encode_prompt</c>) and is UNVERIFIED against real weights.</para></summary>
public sealed class Kandinsky5Recipe : IArchitectureRecipe
{
    /// <inheritdoc/>
    public string Name => "kandinsky5";

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "kandinsky5", StringComparison.OrdinalIgnoreCase);

    /// <inheritdoc/>
    public IRecipePipeline Construct(RecipeContext context)
    {
        // TODO(E-IMG-4): honor user-picked Qwen2.5-VL / CLIP-L / VAE overrides from ImageRequest.Components.
        List<SafeTensorsLoader> loaders = new List<SafeTensorsLoader>();
        try
        {
            Kandinsky5CheckpointConverter.ConvertedWeights converted;
            if (Directory.Exists(context.CheckpointPath))
            {
                Logs.Info($"[Kandinsky5Recipe] Loading transformer diffusers dir: {context.CheckpointPath}.");
                (Kandinsky5CheckpointConverter.ConvertedWeights c, List<SafeTensorsLoader> shardLoaders) = Kandinsky5CheckpointConverter.LoadDiffusersFolder(context.CheckpointPath);
                converted = c;
                loaders.AddRange(shardLoaders);
            }
            else
            {
                Logs.Info($"[Kandinsky5Recipe] Loading transformer: {Path.GetFileName(context.CheckpointPath)}.");
                (Kandinsky5CheckpointConverter.ConvertedWeights c, SafeTensorsLoader loader) = Kandinsky5CheckpointConverter.LoadAndConvert(context.CheckpointPath);
                converted = c;
                loaders.Add(loader);
            }
            if (converted.Transformer.Count == 0)
            {
                throw new InvalidOperationException($"Kandinsky 5 checkpoint '{context.CheckpointPath}' has no transformer weights.");
            }

            // BF16 -> F16 (12 GB, native F16 GEMM). The transient BF16 dequant path is validated for fp8/GGUF, not
            // BF16, and produced a blank image in the generation test; F32 would be 24 GB.
            Kandinsky5Config config = Kandinsky5Config.Lite;
            Dictionary<string, Tensor> transformerWeights = new Dictionary<string, Tensor>(converted.Transformer.Count);
            foreach (KeyValuePair<string, Tensor> kv in converted.Transformer)
            {
                transformerWeights[kv.Key] = kv.Value.DType == DType.BF16 ? kv.Value.CastTo(DType.F16) : kv.Value;
            }
            Kandinsky5Transformer transformer = new Kandinsky5Transformer(config);
            transformer.LoadWeights(transformerWeights);

            string qwenPath = ModelDownloader.EnsureSideModelAsync(SideModels.Qwen2_5_VL_7B, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
            SafeTensorsLoader qwenLoader = new SafeTensorsLoader();
            qwenLoader.Load(qwenPath);
            loaders.Add(qwenLoader);
            LlamaStyleEncoder qwen = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Qwen2_5_VL_7B);
            qwen.LoadWeights(qwenLoader.GetAllTensors());

            string clipPath = ModelDownloader.EnsureSideModelAsync(SideModels.ClipL, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
            SafeTensorsLoader clipLoader = new SafeTensorsLoader();
            clipLoader.Load(clipPath);
            loaders.Add(clipLoader);
            ClipTextEncoder clipL = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipL);
            clipL.LoadWeights(ConvertClipLFromStandalone(clipLoader.GetAllTensors()), prefix: "text_model");

            string vaePath = ModelDownloader.EnsureSideModelAsync(SideModels.FluxAe, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
            SafeTensorsLoader vaeLoader = new SafeTensorsLoader();
            vaeLoader.Load(vaePath);
            loaders.Add(vaeLoader);
            VaeDecoder vae = new VaeDecoder(VaeConfig.Flux);
            vae.LoadWeights(CastToF32(ConvertVaeFromStandalone(vaeLoader.GetAllTensors())));

            // Scheduler shift 5.0 and the Flux VAE scale/shift are the Lite defaults baked into the ctor.
            Kandinsky5Pipeline pipeline = new Kandinsky5Pipeline(context.Backend, transformer, vae, config);
            Logs.Info("[Kandinsky5Recipe] Kandinsky 5 T2I-Lite ready (Qwen2.5-VL + CLIP-L live encode).");
            return new Kandinsky5RecipePipeline(pipeline, qwen, clipL, new Qwen2Tokenizer(), new ClipTokenizer(), context.Backend, transformer, loaders);
        }
        catch (Exception ex)
        {
            Logs.Error("[Kandinsky5Recipe] Construction failed.", ex);
            foreach (SafeTensorsLoader loader in loaders)
            {
                loader.Dispose();
            }
            throw;
        }
    }

    /// <summary>Strips a standalone CLIP-L safetensors file down to the <c>text_model.*</c> keys the encoder expects (Comfy or LDM wrapping), dropping the <c>position_ids</c> buffer.</summary>
    private static Dictionary<string, Tensor> ConvertClipLFromStandalone(IReadOnlyDictionary<string, Tensor> raw)
    {
        Dictionary<string, Tensor> result = new Dictionary<string, Tensor>(raw.Count);
        foreach (KeyValuePair<string, Tensor> kv in raw)
        {
            string key = kv.Key;
            if (key.StartsWith("text_encoders.clip_l.transformer.", StringComparison.Ordinal))
            {
                key = key["text_encoders.clip_l.transformer.".Length..];
            }
            else if (key.StartsWith("conditioner.embedders.0.transformer.", StringComparison.Ordinal))
            {
                key = key["conditioner.embedders.0.transformer.".Length..];
            }
            if (!key.EndsWith("position_ids", StringComparison.Ordinal))
            {
                result[key] = kv.Value;
            }
        }
        return result;
    }

    /// <summary>Normalizes a standalone Flux VAE file into diffusers key naming: strips a Comfy/LDM wrapper prefix, then routes every key through <see cref="CheckpointConvertUtils.ConvertVaeKey"/> (null drops non-VAE keys).</summary>
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

    /// <summary>Casts F16/BF16 VAE weights to F32 (the precision the Flux VAE decoder is validated at), passing already-F32 tensors through.</summary>
    private static Dictionary<string, Tensor> CastToF32(Dictionary<string, Tensor> weights)
    {
        Dictionary<string, Tensor> result = new Dictionary<string, Tensor>(weights.Count);
        foreach (KeyValuePair<string, Tensor> kv in weights)
        {
            result[kv.Key] = (kv.Value.DType == DType.F16 || kv.Value.DType == DType.BF16) ? kv.Value.CastTo(DType.F32) : kv.Value;
        }
        return result;
    }
}
