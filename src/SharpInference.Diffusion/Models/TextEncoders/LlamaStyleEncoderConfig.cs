namespace SharpInference.Diffusion.Models.TextEncoders;

/// <summary>
/// Configuration for Llama-family decoder transformers used as text encoders for diffusion conditioning
/// (Qwen3 in Flux.2 Klein, Mistral-Small-3 in Flux.2 Dev). The transformer is run as an encoder: a single
/// forward pass over the prompt tokens, with the final hidden states harvested as conditioning input.
/// No autoregressive generation, no KV cache, no sampling.
/// </summary>
public record LlamaStyleEncoderConfig
{
    /// <summary>Hidden dimension of the transformer (e.g. 2560 for Qwen3-4B).</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Number of transformer layers (e.g. 36 for Qwen3-4B).</summary>
    public required int NumLayers { get; init; }

    /// <summary>Number of query attention heads.</summary>
    public required int NumQueryHeads { get; init; }

    /// <summary>Number of key/value heads (less than NumQueryHeads for grouped-query attention; equals NumQueryHeads for vanilla MHA).</summary>
    public required int NumKvHeads { get; init; }

    /// <summary>Per-head dimension. NumQueryHeads × HeadDim is the Q projection size; NumKvHeads × HeadDim is the K/V projection size.</summary>
    public required int HeadDim { get; init; }

    /// <summary>Inner SwiGLU MLP dimension. Typical ~3.5× HiddenSize for Llama-family.</summary>
    public required int IntermediateSize { get; init; }

    /// <summary>Vocabulary size (input embedding rows).</summary>
    public required int VocabSize { get; init; }

    /// <summary>RMSNorm epsilon.</summary>
    public float RmsNormEps { get; init; } = 1e-6f;

    /// <summary>RoPE base frequency (theta). Qwen3 uses 1,000,000; original Llama uses 10,000.</summary>
    public float RopeTheta { get; init; } = 1_000_000f;

    /// <summary>Maximum sequence length the RoPE table is precomputed for. Acts as a cap on prompt length.</summary>
    public int MaxPositionEmbeddings { get; init; } = 8192;

    /// <summary>Whether each attention layer applies a per-head RMSNorm to Q and K before RoPE/attention. Qwen3 does this; standard Llama / Mistral do not.</summary>
    public bool QkHeadNorm { get; init; } = true;

    /// <summary>Whether attention Q/K/V/O projections have bias terms. False for Qwen3 / Mistral / Llama.</summary>
    public bool AttentionBias { get; init; } = false;

    /// <summary>End-of-sequence token id, used for finding the last real token in a batch when masking out padding.</summary>
    public int EosTokenId { get; init; }

    /// <summary>Beginning-of-sequence token id (informational; the encoder doesn't insert it itself).</summary>
    public int BosTokenId { get; init; }

    /// <summary>Number of attention heads in each KV group (= NumQueryHeads / NumKvHeads). For GQA this is the repeat factor for K/V before attention.</summary>
    public int KvGroupSize => NumQueryHeads / NumKvHeads;

    /// <summary>Total Q projection output dim (NumQueryHeads × HeadDim).</summary>
    public int QDim => NumQueryHeads * HeadDim;

    /// <summary>Total K/V projection output dim (NumKvHeads × HeadDim).</summary>
    public int KvDim => NumKvHeads * HeadDim;

    /// <summary>Whether the checkpoint has a final RMSNorm at the model root (<c>model.norm.weight</c>). True for Qwen3 / Llama / Mistral-Instruct full checkpoints. False for the BFL Mistral-Small-3 distill packaged with Flux.2 Dev — it ships without final norm and lm_head, used only as a feature extractor.</summary>
    public bool HasFinalNorm { get; init; } = true;

    /// <summary>Qwen3-4B preset (36 layers, hidden=2560, GQA 32:8, head_dim=128, vocab=151936). Matches the safetensors at Comfy-Org/Flux2-klein/text_encoders/qwen_3_4b.safetensors.</summary>
    public static LlamaStyleEncoderConfig Qwen3_4B => new()
    {
        HiddenSize = 2560,
        NumLayers = 36,
        NumQueryHeads = 32,
        NumKvHeads = 8,
        HeadDim = 128,
        IntermediateSize = 9728,
        VocabSize = 151936,
        RmsNormEps = 1e-6f,
        RopeTheta = 1_000_000f,
        MaxPositionEmbeddings = 40960,
        QkHeadNorm = true,
        AttentionBias = false,
        HasFinalNorm = true,
        EosTokenId = 151645,
        BosTokenId = 151643,
    };

    /// <summary>Qwen3-8B preset (36 layers, hidden=4096, GQA 32:8, head_dim=128, intermediate=12288,
    /// vocab=151936). Same Qwen3 family as 4B — only hidden + intermediate change.
    /// Used by Flux.2 Klein 9B (`Comfy-Org/flux2-klein-9B/.../qwen_3_8b_*.safetensors`).
    /// Note: Comfy distributes this as fp4-mixed; SharpInference doesn't yet support FP4 GEMM,
    /// so this preset is currently usable only with fp8 / fp16 Qwen3-8B variants.</summary>
    public static LlamaStyleEncoderConfig Qwen3_8B => new()
    {
        HiddenSize = 4096,
        NumLayers = 36,
        NumQueryHeads = 32,
        NumKvHeads = 8,
        HeadDim = 128,
        IntermediateSize = 12288,
        VocabSize = 151936,
        RmsNormEps = 1e-6f,
        RopeTheta = 1_000_000f,
        MaxPositionEmbeddings = 40960,
        QkHeadNorm = true,
        AttentionBias = false,
        HasFinalNorm = true,
        EosTokenId = 151645,
        BosTokenId = 151643,
    };

    /// <summary>
    /// Mistral-Small-3 (BFL Flux.2 Dev distill) preset: 30 layers, hidden=5120, GQA 32:8,
    /// head_dim=128, IntermediateSize=32768 (~6.4× ratio — wider FFN than standard Mistral),
    /// vocab=131072 (Tekken tokenizer). No per-head Q/K norm, no final norm in the checkpoint —
    /// it ships as a feature extractor for diffusion conditioning. Verified against
    /// <c>Comfy-Org/Flux2/text_encoders/mistral_3_small_flux2_fp8.safetensors</c>.
    /// </summary>
    public static LlamaStyleEncoderConfig MistralSmall3 => new()
    {
        HiddenSize = 5120,
        NumLayers = 30,
        NumQueryHeads = 32,
        NumKvHeads = 8,
        HeadDim = 128,
        IntermediateSize = 32768,
        VocabSize = 131072,
        RmsNormEps = 1e-5f,
        RopeTheta = 1_000_000f,
        MaxPositionEmbeddings = 32768,
        QkHeadNorm = false,
        AttentionBias = false,
        HasFinalNorm = false,
        // Mistral special tokens (Tekken). EOS=2 (</s>), BOS=1 (<s>) — placeholders matching the
        // standard Mistral tokenizer; pipeline-level use of these is up to the caller.
        EosTokenId = 2,
        BosTokenId = 1,
    };

    /// <summary>
    /// Ministral-3B preset as packaged for ERNIE-Image (Baidu): 26 layers, hidden=3072, GQA 32:8,
    /// head_dim=128, intermediate=9216, vocab=131072 (Tekken). The 3072 hidden matches ERNIE-Image's
    /// <c>text_in_dim=3072</c>, so the encoder hidden is fed straight into <c>text_in: Linear(3072→hidden=4096)</c>.
    /// Verified against <c>baidu/ERNIE-Image/text_encoder/config.json</c> (model_type "ministral3", wrapped
    /// in a <c>Mistral3Model</c> envelope alongside a Pixtral vision_config we ignore for pure t2i).
    ///
    /// RoPE uses YaRN scaling with theta=1M; for short prompts (≤4096 tokens) the unscaled RoPE table is fine.
    /// `tie_word_embeddings=true` — the LM head shares weights with the input embedding (we don't run the
    /// LM head for text-conditioning anyway). No per-head Q/K norm. `HasFinalNorm=true` for the full
    /// Comfy-Org safetensors (`text_encoders/ministral-3-3b.safetensors`); ships the full encoder including
    /// `model.norm.weight`.
    /// </summary>
    public static LlamaStyleEncoderConfig Ministral3B => new()
    {
        HiddenSize = 3072,
        NumLayers = 26,
        NumQueryHeads = 32,
        NumKvHeads = 8,
        HeadDim = 128,
        IntermediateSize = 9216,
        VocabSize = 131072,
        RmsNormEps = 1e-5f,
        RopeTheta = 1_000_000f,
        MaxPositionEmbeddings = 262144,
        QkHeadNorm = false,
        AttentionBias = false,
        HasFinalNorm = true,
        EosTokenId = 2,
        BosTokenId = 1,
    };

    /// <summary>Qwen2.5-VL-7B preset (text-only path): 28 layers, hidden=3584, GQA 28:4, head_dim=128, intermediate=18944, vocab=152064. Matches diffusers' `Qwen2_5_VLForConditionalGeneration` config (text encoder portion); the vision adapter is ignored for pure text conditioning of Qwen-Image. RoPE uses Qwen2's M-RoPE — for text-only forward we collapse to standard RoPE which is correct for non-multimodal usage. RMSNorm eps 1e-6, theta 1M (same as Qwen3).</summary>
    public static LlamaStyleEncoderConfig Qwen2_5_VL_7B => new()
    {
        HiddenSize = 3584,
        NumLayers = 28,
        NumQueryHeads = 28,
        NumKvHeads = 4,
        HeadDim = 128,
        IntermediateSize = 18944,
        VocabSize = 152064,
        RmsNormEps = 1e-6f,
        RopeTheta = 1_000_000f,
        MaxPositionEmbeddings = 32768,
        QkHeadNorm = false,
        AttentionBias = true,
        HasFinalNorm = true,
        EosTokenId = 151645,
        BosTokenId = 151643,
    };

    /// <summary>Qwen2.5-VL-3B preset (text-only path) as packaged for OmniGen2: 36 layers, hidden=2048, GQA 16:2, head_dim=128, intermediate=11008, vocab=151936. Verified against <c>OmniGen2/OmniGen2/mllm/config.json</c>: <c>num_hidden_layers=36, num_attention_heads=16, num_key_value_heads=2, hidden_size=2048, intermediate_size=11008, rms_norm_eps=1e-6, rope_theta=1e6</c>. The vision adapter is ignored for pure t2i (the OmniGen2 transformer's <c>text_feat_dim=2048</c> matches the text-encoder hidden directly). M-RoPE collapsed to standard RoPE for the text-only forward path.</summary>
    public static LlamaStyleEncoderConfig Qwen2_5_VL_3B => new()
    {
        HiddenSize = 2048,
        NumLayers = 36,
        NumQueryHeads = 16,
        NumKvHeads = 2,
        HeadDim = 128,
        IntermediateSize = 11008,
        VocabSize = 151936,
        RmsNormEps = 1e-6f,
        RopeTheta = 1_000_000f,
        MaxPositionEmbeddings = 128000,
        QkHeadNorm = false,
        AttentionBias = true,
        HasFinalNorm = true,
        EosTokenId = 151645,
        BosTokenId = 151643,
    };

    /// <summary>Gemma 2 2B preset (Lumina-Image-2.0 text encoder). 26 layers, hidden=2304, GQA 8:4,
    /// head_dim=256 (note: hidden=2304 != heads*head_dim=2048 — Gemma 2 uses an oversized head_dim
    /// distinct from hidden_size, projected back via the o_proj weight), intermediate=9216,
    /// vocab=256000. RMSNorm eps 1e-6, RoPE theta 10000, no attention bias.
    /// <para>Caveats vs vanilla Llama family: Gemma 2 uses GeGLU (GELU(gate)*up) rather than SwiGLU,
    /// applies a fixed query pre-attention scalar of sqrt(256), softcaps attention logits at 50.0,
    /// alternates sliding-window attention (4096), and uses pre+post norms on both attention and FFN
    /// (4 RMSNorms per block). The current <see cref="LlamaStyleEncoder"/> assumes SwiGLU + 2 norms +
    /// no softcap — running Gemma 2 weights through it as-is will not produce reference-correct text
    /// embeddings. The preset is provided for downstream pipelines (Lumina 2.0) that pre-compute
    /// Gemma 2 embeddings via a Gemma-2-aware encoder; the <see cref="LlamaStyleEncoder"/> path is
    /// approximate. Verified against <c>Alpha-VLLM/Lumina-Image-2.0/text_encoder/config.json</c>.</summary>
    public static LlamaStyleEncoderConfig Gemma2_2B => new()
    {
        HiddenSize = 2304,
        NumLayers = 26,
        NumQueryHeads = 8,
        NumKvHeads = 4,
        HeadDim = 256,
        IntermediateSize = 9216,
        VocabSize = 256000,
        RmsNormEps = 1e-6f,
        RopeTheta = 10_000f,
        MaxPositionEmbeddings = 8192,
        QkHeadNorm = false,
        AttentionBias = false,
        HasFinalNorm = true,
        EosTokenId = 1,
        BosTokenId = 2,
    };
}
