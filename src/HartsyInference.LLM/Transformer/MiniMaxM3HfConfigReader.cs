using System.Text.Json;
using HartsyInference.Core.Exceptions;

namespace HartsyInference.LLM.Transformer;

/// <summary>Reads the <c>text_config</c> of a MiniMax-M3 <c>config.json</c> (<c>model_type</c> <c>minimax_m3_vl</c>) into a <see cref="TransformerConfig"/>; the vision tower and projector are not read.</summary>
public static class MiniMaxM3HfConfigReader
{
    public const string ModelType = "minimax_m3_vl";

    /// <exception cref="HartsyInferenceException">The config is not MiniMax-M3, is missing a field, or uses a layout the engine does not run (a subset of sparse layers, an output gate, a non-max indexer score).</exception>
    public static TransformerConfig FromHuggingFace(JsonElement config)
    {
        HfConfigFields root = new(config, "config", "MiniMax-M3");
        if (!string.Equals(root.OptionalString("model_type"), ModelType, StringComparison.Ordinal))
            throw new HartsyInferenceException($"MiniMax-M3 config: model_type is '{root.OptionalString("model_type")}', not '{ModelType}'.");
        HfConfigFields text = root.Object("text_config");
        HfConfigFields sparse = text.Object("sparse_attention_config");
        int layers = text.Int("num_hidden_layers");
        int heads = text.Int("num_attention_heads");
        int headDim = text.Int("head_dim");
        if (text.Bool("attention_output_gate", false) || text.OptionalString("qk_norm_type") != "per_head" || !text.Bool("use_qk_norm", false))
            throw new HartsyInferenceException("MiniMax-M3 config: only per-head QK norm without an attention output gate is supported.");
        if (!sparse.Bool("use_sparse_attention", false) || sparse.OptionalString("sparse_score_type") != "max" || sparse.Int("sparse_init_block", 0) != 0
            || sparse.Ints("sparse_attention_freq").Length != layers || sparse.Ints("sparse_attention_freq").Any(static f => f != 1)
            || sparse.Ints("sparse_disable_index_value").Any(static f => f != 1))
            throw new HartsyInferenceException("MiniMax-M3 config: only max-scored block-sparse attention on every layer without an index value path is supported.");
        int[] moeLayers = text.Ints("moe_layer_freq");
        int firstDense = Array.IndexOf(moeLayers, 1);
        if (moeLayers.Length != layers || firstDense < 0 || moeLayers.Skip(firstDense).Any(static f => f != 1))
            throw new HartsyInferenceException("MiniMax-M3 config: moe_layer_freq must be dense layers followed by sparse layers only.");
        if (text.OptionalString("scoring_func") != "sigmoid" || !text.Bool("use_routing_bias", false) || text.OptionalString("hidden_act") != "swigluoai")
            throw new HartsyInferenceException("MiniMax-M3 config: only sigmoid routing with a bias and the clamped alpha-SwiGLU is supported.");
        float alpha = (float)text.Double("swiglu_alpha");
        float limit = (float)text.Double("swiglu_limit");
        float rmsEps = (float)text.Double("rms_norm_eps", 1e-6d);
        int rotary = (int)(headDim * text.Double("partial_rotary_factor", 1d));
        return new TransformerConfig
        {
            HiddenSize = text.Int("hidden_size"),
            NumLayers = layers,
            NumHeads = heads,
            NumKvHeads = text.Int("num_key_value_heads"),
            HeadDim = headDim,
            RotaryDim = rotary,
            IntermediateSize = text.Int("dense_intermediate_size"),
            VocabSize = text.Int("vocab_size"),
            MaxPositionEmbeddings = text.Int("max_position_embeddings"),
            RopeTheta = (float)text.Double("rope_theta"),
            RmsNormEps = rmsEps,
            TieWordEmbeddings = text.Bool("tie_word_embeddings", false),
            QkNorm = true,
            // The checkpoint stores the centered Gemma norm weight; the engine adds the 1.
            RmsNormAddOne = text.Bool("use_gemma_norm", false),
            Activation = ActivationKind.SwigluOai,
            SwigluAlpha = alpha,
            SwigluLimit = limit,
            Architecture = "minimax-m3",
            Indexer = new IndexerConfig
            {
                Heads = sparse.Int("sparse_num_index_heads"),
                KeyLength = sparse.Int("sparse_index_dim"),
                BlockSize = sparse.Int("sparse_block_size"),
                TopKBlocks = sparse.Int("sparse_topk_blocks"),
                LocalBlocks = sparse.Int("sparse_local_block"),
            },
            Moe = new MoeConfig
            {
                NumExperts = text.Int("num_local_experts"),
                NumExpertsPerTok = text.Int("num_experts_per_tok"),
                MoeIntermediateSize = text.Int("intermediate_size"),
                SharedExpertIntermediateSize = text.Int("shared_intermediate_size"),
                Scoring = MoeScoring.Sigmoid,
                NormTopKProb = true,
                FirstDenseLayers = firstDense,
                RoutedScalingFactor = (float)text.Double("routed_scaling_factor"),
                Activation = ActivationKind.SwigluOai,
                SwigluAlpha = alpha,
                SwigluLimit = limit,
            },
        };
    }
}
