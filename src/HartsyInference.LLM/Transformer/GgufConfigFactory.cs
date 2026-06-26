using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.Gguf;

namespace HartsyInference.LLM.Transformer;

/// <summary>Builds a <see cref="TransformerConfig"/> from a loaded GGUF model's metadata + remapped weight dict.
/// Reads the llama.cpp architecture kv (<c>{arch}.block_count</c>, <c>{arch}.embedding_length</c>, etc.) and
/// infers the three Qwen variation axes structurally from the weights (QKV bias / per-head q-norm presence,
/// tied vs separate lm_head). Covers the Qwen2 / Qwen3 / Llama family decoder LLMs that the
/// <see cref="GenericTransformer"/> runs; throws on missing required metadata.</summary>
public static class GgufConfigFactory
{
    /// <summary>Derives a <see cref="TransformerConfig"/> from <paramref name="metadata"/> and the HF-remapped
    /// <paramref name="weights"/> (the dict returned by <c>GgufModelLoader.Load(...).Weights</c>).</summary>
    public static TransformerConfig FromGguf(GgufMetadata metadata, IReadOnlyDictionary<string, Tensor> weights, bool lowVramQuant = false)
    {
        string arch = metadata.GetString("general.architecture") ?? "";
        if (arch.Length == 0) throw new ArgumentException("GGUF metadata has no general.architecture.", nameof(metadata));

        int layers = RequireUInt(metadata, $"{arch}.block_count");
        int hidden = RequireUInt(metadata, $"{arch}.embedding_length");
        int heads = RequireUInt(metadata, $"{arch}.attention.head_count");
        int kvHeads = (int)metadata.GetUInt32($"{arch}.attention.head_count_kv", (uint)heads);
        int intermediate = RequireUInt(metadata, $"{arch}.feed_forward_length");

        // head_dim: GGUF key_length when present, else hidden / heads (coupled).
        int headDim = (int)metadata.GetUInt32($"{arch}.attention.key_length", 0u);
        if (headDim == 0) headDim = hidden / heads;

        int maxPos = (int)metadata.GetUInt32($"{arch}.context_length", 32_768u);
        float ropeTheta = metadata.GetFloat32($"{arch}.rope.freq_base", 1_000_000f);
        float rmsEps = metadata.GetFloat32($"{arch}.attention.layer_norm_rms_epsilon", 1e-6f);

        // Vocab from the embedding table. GGUF stores token_embd with dims [hidden, vocab] (reverse of the
        // safetensors [vocab, hidden] order), so derive it from the total element count to be order-agnostic.
        if (!weights.TryGetValue("model.embed_tokens.weight", out Tensor? embed))
            throw new ArgumentException("GGUF weights missing model.embed_tokens.weight (cannot infer vocab).", nameof(weights));
        int vocab = (int)(embed.ElementCount / hidden);

        // Structural feature detection from the remapped layer-0 keys.
        bool attentionBias = weights.ContainsKey("model.layers.0.self_attn.q_proj.bias");
        bool qkNorm = weights.ContainsKey("model.layers.0.self_attn.q_norm.weight");
        bool tied = !weights.ContainsKey("lm_head.weight");

        // Gemma family knobs. Most are architectural constants (GeGLU, (1+w) norm, embedding scale, sandwich
        // norm) plus a couple of values llama.cpp hardcodes rather than storing (Gemma-3 local RoPE base 10000,
        // sliding-window pattern 6). Gemma-2 adds attn/final logit soft-cap; Gemma-3 adds dual-RoPE + Q/K norm.
        bool isGemma = arch.StartsWith("gemma", StringComparison.Ordinal);
        bool isGemma2 = arch == "gemma2";
        bool isGemma3 = arch == "gemma3";
        bool sandwich = isGemma && weights.ContainsKey("model.layers.0.post_feedforward_layernorm.weight");
        float embScale = isGemma ? (float)Math.Sqrt(hidden) : 1f;
        float queryPreAttn = metadata.GetFloat32($"{arch}.attention.query_pre_attn_scalar", 0f);
        float localTheta = isGemma3 ? metadata.GetFloat32($"{arch}.rope.local_freq_base", 10_000f) : 0f;
        int swWindow = (int)metadata.GetUInt32($"{arch}.attention.sliding_window", 0u);
        int swPattern = (int)metadata.GetUInt32($"{arch}.attention.sliding_window_pattern",
            isGemma3 ? 6u : isGemma2 ? 2u : 0u);
        float attnCap = isGemma2 ? metadata.GetFloat32($"{arch}.attn_logit_softcapping", 0f) : 0f;
        float finalCap = isGemma2 ? metadata.GetFloat32($"{arch}.final_logit_softcapping", 0f) : 0f;

        return new TransformerConfig
        {
            HiddenSize = hidden,
            NumLayers = layers,
            NumHeads = heads,
            NumKvHeads = kvHeads,
            HeadDim = headDim,
            IntermediateSize = intermediate,
            VocabSize = vocab,
            MaxPositionEmbeddings = maxPos,
            RopeTheta = ropeTheta,
            RmsNormEps = rmsEps,
            AttentionBias = attentionBias,
            QkNorm = qkNorm,
            TieWordEmbeddings = tied,
            // RoPE pairing depends on how llama.cpp's converter laid out the Q/K weights for this arch:
            //   - llama family (incl. Mistral): convert_hf_to_gguf.py PERMUTES wq/wk so that ggml NORM rope
            //     (interleaved adjacent pairs 2i,2i+1) reproduces HF rotate_half → we must apply Interleaved.
            //   - qwen2/qwen3: no permute, ggml uses NEOX rope (split-half pairs i,i+half) → SplitHalf.
            // Using the wrong pairing leaves attention rotating mismatched dimensions → coherent-looking garbage.
            Rope = arch == "llama" ? RopeStyle.Interleaved : RopeStyle.SplitHalf,
            RopeScaling = BuildRopeScaling(metadata, arch, weights, headDim),
            LowVramQuant = lowVramQuant,
            // Gemma family (all default to no-op for Qwen/Llama).
            Activation = isGemma ? ActivationKind.GeluTanh : ActivationKind.Silu,
            SandwichNorm = sandwich,
            // llama.cpp's GGUF converter already bakes Gemma's (1+w) offset into the stored norm weights, so the
            // GGUF path uses them directly. RmsNormAddOne is only for loading raw (centered) HF safetensors.
            RmsNormAddOne = false,
            EmbeddingScale = embScale,
            QueryPreAttnScalar = queryPreAttn,
            RopeLocalTheta = localTheta,
            SlidingWindow = swWindow,
            SlidingWindowPattern = swPattern,
            AttnLogitSoftcap = attnCap,
            FinalLogitSoftcap = finalCap,
        };
    }

    /// <summary>RoPE scaling from GGUF: the precomputed <c>rope_freqs.weight</c> per-frequency multiplier when
    /// present (llama.cpp bakes Llama-3 scaling there), else the <c>{arch}.rope.scaling.*</c> metadata
    /// (yarn/linear). Returns <see cref="RopeScaling.None"/> for standard RoPE (Qwen/Mistral).</summary>
    private static unsafe RopeScaling BuildRopeScaling(GgufMetadata metadata, string arch,
        IReadOnlyDictionary<string, Tensor> weights, int headDim)
    {
        if (weights.TryGetValue("model.rope_freqs.weight", out Tensor? rf))
        {
            Tensor f32 = rf.DType == DType.F32 ? rf : rf.CastTo(DType.F32);
            int n = Math.Min(headDim / 2, (int)f32.ElementCount);
            float[] factors = new float[n];
            float* p = (float*)f32.DataPointer;
            for (int i = 0; i < n; i++) factors[i] = p[i];
            return new RopeScaling { Type = RopeScalingType.Llama3, InvFreqFactors = factors };
        }

        // Phi-3 LongRope: per-dimension short/long factor tables (the rope_factors_short/long tensors) plus the
        // pre-extension context length and an explicit attention factor (mscale). Short table is used at
        // seqLen ≤ original context, long table above it.
        if (weights.TryGetValue("model.rope_factors_long.weight", out Tensor? rfl)
            && weights.TryGetValue("model.rope_factors_short.weight", out Tensor? rfs))
        {
            double[] longF = ReadF32Factors(rfl, headDim / 2);
            double[] shortF = ReadF32Factors(rfs, headDim / 2);
            double origCtxLr = metadata.GetUInt32($"{arch}.rope.scaling.original_context_length", 4096u);
            double attnLr = metadata.ContainsKey($"{arch}.rope.scaling.attn_factor")
                ? metadata.GetFloat32($"{arch}.rope.scaling.attn_factor") : double.NaN;
            return new RopeScaling
            {
                Type = RopeScalingType.LongRope, ShortFactor = shortF, LongFactor = longF,
                OriginalContextLength = origCtxLr, AttentionFactor = attnLr,
            };
        }

        string type = (metadata.GetString($"{arch}.rope.scaling.type") ?? "").ToLowerInvariant();
        if (type.Length == 0 || type == "none") return RopeScaling.None;

        double factor = metadata.GetFloat32($"{arch}.rope.scaling.factor", 1f);
        double origCtx = metadata.GetUInt32($"{arch}.rope.scaling.original_context_length", 0u);
        double attn = metadata.ContainsKey($"{arch}.rope.scaling.attn_factor")
            ? metadata.GetFloat32($"{arch}.rope.scaling.attn_factor")
            : double.NaN;
        return type switch
        {
            "linear" => new RopeScaling { Type = RopeScalingType.Linear, Factor = factor },
            "yarn" => new RopeScaling { Type = RopeScalingType.Yarn, Factor = factor, OriginalContextLength = origCtx, AttentionFactor = attn },
            _ => RopeScaling.None,
        };
    }

    /// <summary>Reads up to <paramref name="count"/> F32 values from a (possibly half/quantized) rope-factor tensor.</summary>
    private static unsafe double[] ReadF32Factors(Tensor t, int count)
    {
        Tensor f32 = t.DType == DType.F32 ? t : t.CastTo(DType.F32);
        int n = Math.Min(count, (int)f32.ElementCount);
        double[] factors = new double[n];
        float* p = (float*)f32.DataPointer;
        for (int i = 0; i < n; i++) factors[i] = p[i];
        return factors;
    }

    private static int RequireUInt(GgufMetadata metadata, string key)
    {
        if (!metadata.ContainsKey(key))
            throw new ArgumentException($"GGUF metadata missing required key '{key}'.", nameof(metadata));
        return (int)metadata.GetUInt32(key);
    }
}
