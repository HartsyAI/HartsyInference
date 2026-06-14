using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.CheckpointConverters.Utils;
using HartsyInference.ModelHandler.SafeTensors;

namespace HartsyInference.ModelHandler.CheckpointConverters;

/// <summary>Loads + buckets an Anima single-file safetensors checkpoint into transformer, llm_adapter, VAE, and
/// text-encoder dictionaries. The Anima ComfyUI checkpoint stores every weight under a flat <c>net.</c> prefix:
/// <list type="bullet">
///   <item><c>net.x_embedder.*</c>, <c>net.t_embedder.*</c>, <c>net.t_embedding_norm.*</c>, <c>net.blocks.*</c>,
///        <c>net.final_layer.*</c> — main DiT trunk → <see cref="ConvertedWeights.Transformer"/>.</item>
///   <item><c>net.llm_adapter.*</c> — Anima-specific 6-block adapter → <see cref="ConvertedWeights.LlmAdapter"/>.</item>
/// </list>
/// The converter strips the <c>net.</c> prefix from every key and (for the llm_adapter bucket) the
/// <c>llm_adapter.</c> sub-prefix too, so downstream loaders see clean names like <c>blocks.0.self_attn.q_proj.weight</c>
/// and <c>blocks.0.norm_self_attn.weight</c>.
///
/// VAE + text encoder buckets exist for compatibility with full pipeline checkpoints; Anima ships diffusion-only,
/// so those buckets are empty for the canonical <c>anima-preview3-base.safetensors</c>. The converter still recognizes
/// <c>vae.</c> / <c>text_encoder.</c> prefixes in case a future Anima distribution bundles them.
///
/// **Backwards compatibility:** also tolerates diffusers-style key prefixes (<c>model.diffusion_model.</c>,
/// <c>transformer.</c>) in case an earlier Anima dump used them.</summary>
public sealed class AnimaCheckpointConverter
{
    /// <summary>Result of partitioning an Anima safetensors file.</summary>
    public sealed class ConvertedWeights
    {
        /// <summary>Main DiT trunk weights (prefix-stripped). Keys look like
        /// <c>blocks.0.self_attn.q_proj.weight</c>, <c>x_embedder.proj.1.weight</c>, <c>final_layer.linear.weight</c>.</summary>
        public required Dictionary<string, Tensor> Transformer { get; init; }

        /// <summary>Anima <c>llm_adapter</c> weights (both <c>net.</c> and <c>llm_adapter.</c> prefixes stripped).
        /// Keys look like <c>blocks.0.self_attn.q_proj.weight</c>, <c>norm.weight</c>, <c>out_proj.weight</c>,
        /// <c>embed.weight</c>.</summary>
        public required Dictionary<string, Tensor> LlmAdapter { get; init; }

        /// <summary>VAE weights if bundled (empty for stock Anima).</summary>
        public required Dictionary<string, Tensor> Vae { get; init; }

        /// <summary>Text encoder weights if bundled (empty for stock Anima).</summary>
        public required Dictionary<string, Tensor> TextEncoder { get; init; }

        /// <summary>True iff any transformer tensor is fp8-scaled. Stock Anima ships BF16 so this is false.</summary>
        public required bool IsFp8Mix { get; init; }
    }

    /// <summary>Loads and partitions an Anima single-file checkpoint.</summary>
    public static (ConvertedWeights weights, SafeTensorsLoader loader) LoadAndConvert(string checkpointPath)
    {
        SafeTensorsLoader loader = new();
        loader.Load(checkpointPath);
        ConvertedWeights converted = Convert(loader.GetAllTensors());
        return (converted, loader);
    }

    /// <summary>Partitions a flat dict by key prefix and strips prefixes from the surviving keys.</summary>
    public static ConvertedWeights Convert(Dictionary<string, Tensor> allWeights)
    {
        Dictionary<string, Tensor> dequanted = CheckpointConvertUtils.ApplyFp8ScaledDequant(allWeights);

        Dictionary<string, Tensor> transformer = new();
        Dictionary<string, Tensor> llmAdapter = new();
        Dictionary<string, Tensor> vae = new();
        Dictionary<string, Tensor> textEncoder = new();

        foreach (KeyValuePair<string, Tensor> kvp in dequanted)
        {
            string key = kvp.Key;
            Tensor tensor = kvp.Value;

            if (key.StartsWith("vae.", StringComparison.Ordinal))
            {
                vae[key["vae.".Length..]] = tensor;
                continue;
            }
            if (key.StartsWith("text_encoder.", StringComparison.Ordinal))
            {
                textEncoder[key["text_encoder.".Length..]] = tensor;
                continue;
            }
            if (key.StartsWith("text_encoders.", StringComparison.Ordinal))
            {
                textEncoder[key["text_encoders.".Length..]] = tensor;
                continue;
            }

            // Strip optional diffusers wrappers, then the Anima `net.` root prefix.
            string stripped = key;
            if (stripped.StartsWith("model.diffusion_model.", StringComparison.Ordinal))
                stripped = stripped["model.diffusion_model.".Length..];
            else if (stripped.StartsWith("transformer.", StringComparison.Ordinal))
                stripped = stripped["transformer.".Length..];

            if (stripped.StartsWith("net.", StringComparison.Ordinal))
                stripped = stripped["net.".Length..];

            // Anima's llm_adapter is a self-contained sub-module — give it its own bucket.
            if (stripped.StartsWith("llm_adapter.", StringComparison.Ordinal))
            {
                llmAdapter[stripped["llm_adapter.".Length..]] = tensor;
                continue;
            }

            transformer[stripped] = tensor;
        }

        bool isFp8Mix = false;
        foreach (Tensor t in transformer.Values)
        {
            if (t.DType == DType.F8E4M3 || t.DType == DType.F8E5M2) { isFp8Mix = true; break; }
        }

        return new ConvertedWeights
        {
            Transformer = transformer,
            LlmAdapter = llmAdapter,
            Vae = vae,
            TextEncoder = textEncoder,
            IsFp8Mix = isFp8Mix,
        };
    }
}
