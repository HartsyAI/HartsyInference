namespace SharpInference.Diffusion.Models.TextEncoders;

/// <summary>Configuration for Llama-family decoder transformers used as text encoders for diffusion conditioning (Qwen3 in Flux.2 Klein, Mistral-Small-3 in Flux.2 Dev). Run as an encoder: single forward pass with the final hidden states harvested as conditioning. No autoregressive generation, no KV cache, no sampling.</summary>
public sealed record LlamaStyleEncoderConfig
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

    /// <summary>MLP activation. <see cref="MlpActivation.Silu"/> for Llama / Qwen / Mistral (SwiGLU);
    /// <see cref="MlpActivation.GeluTanh"/> for Gemma 2 (GeGLU with the tanh-approx GELU). The
    /// gated-MLP wiring is identical (<c>down(act(gate) * up)</c>); only the activation function differs.</summary>
    public MlpActivation Activation { get; init; } = MlpActivation.Silu;

    /// <summary>True for models that store RMSNorm scales as offsets from 1.0 (Gemma 2 convention:
    /// effective scale = 1 + weight). False for Llama / Qwen / Mistral. The encoder pre-adds 1 to the
    /// scale tensor at load time, so the runtime path stays unchanged.</summary>
    public bool RmsNormScalePlusOne { get; init; } = false;

    /// <summary>True for Gemma 2 family — adds <c>post_attention_layernorm</c> after attention and
    /// renames the pre-MLP norm to <c>pre_feedforward_layernorm</c>, plus a
    /// <c>post_feedforward_layernorm</c> after the FFN. Result: 4 norms per block instead of 2.
    /// Llama / Qwen use 2 norms.</summary>
    public bool HasFfnSandwichNorms { get; init; } = false;

    // ────────────────────────────────────────────────────────────────────
    // GPT-OSS additions (MoE FFN, attention sinks, sliding window per-layer,
    // YaRN-extended RoPE, clamped non-standard SwiGLU). All default to off so
    // existing Llama / Qwen / Mistral / Gemma presets continue to work.
    // ────────────────────────────────────────────────────────────────────

    /// <summary>Per-layer attention pattern. <c>null</c> (default) → uniform causal attention with no
    /// windowing. Non-null → must contain one entry per layer; each entry must be either
    /// <c>"full_attention"</c> or <c>"sliding_attention"</c>. GPT-OSS alternates sliding/full starting
    /// with sliding at layer 0. When a layer is sliding, only positions within
    /// <see cref="SlidingWindow"/> tokens to the left are attended (plus the current token).</summary>
    public string[]? LayerAttentionTypes { get; init; }

    /// <summary>Sliding-window size (number of attended positions including current) for any layer
    /// marked <c>"sliding_attention"</c> in <see cref="LayerAttentionTypes"/>. GPT-OSS uses 128.
    /// Ignored if <see cref="LayerAttentionTypes"/> is null.</summary>
    public int SlidingWindow { get; init; } = 128;

    /// <summary>Number of routed experts per Mixture-of-Experts FFN. <c>0</c> (default) → dense FFN
    /// (Llama path). GPT-OSS uses 32. Each layer has its own router and expert bank.</summary>
    public int NumLocalExperts { get; init; } = 0;

    /// <summary>Number of experts each token is routed to (top-k). GPT-OSS uses 4 of 32.
    /// Must satisfy <c>0 &lt; NumExpertsPerToken &lt;= NumLocalExperts</c> when MoE is enabled.</summary>
    public int NumExpertsPerToken { get; init; } = 0;

    /// <summary>True when each attention layer has a learnable per-head <c>sinks</c> parameter mixed
    /// into the attention logits (GPT-OSS-only). The sink acts as a per-head "bias bucket" that absorbs
    /// probability mass under softmax, mitigating the attention-sink pathology in long-context models.
    /// Implementation detail: for each head <c>h</c>, the softmax denominator gets an extra
    /// <c>exp(sinks[h])</c> term, equivalent to prepending one virtual key-value pair with logit
    /// <c>sinks[h]</c> and zero value. We implement this by post-scaling attention probabilities by
    /// <c>1 / (1 + exp(sinks[h] - max_logit) / sum_exp)</c> equivalently to running softmax over
    /// <c>[logits, sinks[h]]</c>.</summary>
    public bool HasAttentionSinks { get; init; } = false;

    /// <summary>True when the gated MLP uses GPT-OSS's clamped SwiGLU variant rather than standard
    /// SwiGLU. Math: <c>gate.clamp(max=L)</c>, <c>up.clamp(-L, L)</c>,
    /// <c>glu = gate * sigmoid(α·gate)</c>, output = <c>(up + 1) · glu</c>. Different from standard
    /// SwiGLU in three ways: (1) clamping, (2) SiLU is taken at <c>α·gate</c> with <c>α=1.702</c>
    /// (GELU-approximation), (3) up is shifted by +1.</summary>
    public bool ClampedSwiGlu { get; init; } = false;

    /// <summary>Clamp limit <c>L</c> for the GPT-OSS-style gated MLP. Default 7.0 matches
    /// <c>config.swiglu_limit</c> in <c>openai/gpt-oss-20b</c>. Only used when
    /// <see cref="ClampedSwiGlu"/> is true.</summary>
    public float ClampedSwiGluLimit { get; init; } = 7.0f;

    /// <summary>α coefficient for the SiLU approximation inside GPT-OSS-style gated MLP. Default 1.702
    /// — the "GELU via SiLU" approximation constant. Only used when <see cref="ClampedSwiGlu"/> is true.</summary>
    public float ClampedSwiGluAlpha { get; init; } = 1.702f;

    /// <summary>RoPE scaling type. <see cref="RopeScalingType.None"/> = standard RoPE (default for
    /// Llama / Qwen / Mistral / Gemma). <see cref="RopeScalingType.Yarn"/> = YaRN-extended RoPE used
    /// by GPT-OSS.</summary>
    public RopeScalingType RopeScaling { get; init; } = RopeScalingType.None;

    /// <summary>YaRN scaling factor (original_max / extended_max). GPT-OSS: 32.0 (extends 4096 →
    /// 131072). Only used when <see cref="RopeScaling"/> is <see cref="RopeScalingType.Yarn"/>.</summary>
    public float YarnFactor { get; init; } = 1.0f;

    /// <summary>YaRN high-frequency boundary — wavelengths shorter than
    /// <c>original_max_pos / beta_fast</c> use the original (unscaled) inv-freq.</summary>
    public float YarnBetaFast { get; init; } = 32.0f;

    /// <summary>YaRN low-frequency boundary — wavelengths longer than
    /// <c>original_max_pos / beta_slow</c> use fully-scaled inv-freq (divided by factor).</summary>
    public float YarnBetaSlow { get; init; } = 1.0f;

    /// <summary>The pre-extension max position. YaRN uses this to define which frequencies need
    /// interpolation vs extrapolation. GPT-OSS: 4096.</summary>
    public int YarnOriginalMaxPosition { get; init; } = 4096;

    /// <summary>Qwen3-0.6B-Base preset (28 layers, hidden=1024, GQA 16:8, head_dim=128, intermediate=3072,
    /// vocab=151936). Q dim = 16×128 = 2048, K/V dim = 8×128 = 1024 — attention projections are wider than
    /// the hidden size (Qwen3 0.6B's signature property). Used by Anima as its text encoder
    /// (<c>circlestone-labs/Anima/text_encoders/qwen_3_06b_base.safetensors</c>). Per
    /// <c>Qwen/Qwen3-0.6B-Base/config.json</c>: <c>tie_word_embeddings=true</c>, theta=1M, RMSNorm eps=1e-6,
    /// max_position=32768.</summary>
    public static LlamaStyleEncoderConfig Qwen3_0_6B => new()
    {
        HiddenSize = 1024,
        NumLayers = 28,
        NumQueryHeads = 16,
        NumKvHeads = 8,
        HeadDim = 128,
        IntermediateSize = 3072,
        VocabSize = 151936,
        RmsNormEps = 1e-6f,
        RopeTheta = 1_000_000f,
        MaxPositionEmbeddings = 32768,
        QkHeadNorm = true,
        AttentionBias = false,
        HasFinalNorm = true,
        EosTokenId = 151645,
        BosTokenId = 151643,
    };

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

    /// <summary>Qwen3-Embedding-0.6B preset — ACE-Step 1.5's text + lyric feature extractor (run as a
    /// per-token hidden-state source; the 1024-d states feed the main model's text_projector / lyric
    /// encoder, no sentence pooling). Architecturally identical to <see cref="Qwen3_0_6B"/> except the
    /// embedding model's trimmed vocab (151669) and plain <c>&lt;|endoftext|&gt;</c> EOS — verified against
    /// <c>ACE-Step/Ace-Step1.5/Qwen3-Embedding-0.6B/config.json</c>.</summary>
    public static LlamaStyleEncoderConfig Qwen3_Embedding_0_6B => Qwen3_0_6B with
    {
        VocabSize = 151669,
        EosTokenId = 151643,
    };

    /// <summary>Qwen3-VL-8B-Instruct (language tower) preset, used by Ideogram 4 as its sole conditioning encoder.
    /// Dimensionally identical to <see cref="Qwen3_8B"/> (36 layers, hidden=4096, GQA 32:8, head_dim=128,
    /// intermediate=12288, vocab=151936) EXCEPT <c>rope_theta=5e6</c> (verified against
    /// <c>Qwen/Qwen3-VL-8B-Instruct/config.json</c> <c>text_config.rope_theta</c>; plain Qwen3-8B uses 1e6).
    /// The vision tower is unused — only the text decoder layers run, and Ideogram taps 13 of them via
    /// <see cref="LlamaStyleEncoder.EncodeMultiLayer"/> (M-RoPE collapses to standard RoPE for text-only input).</summary>
    public static LlamaStyleEncoderConfig Qwen3_VL_8B => Qwen3_8B with { RopeTheta = 5_000_000f };

    /// <summary>Llama-3.1-8B-Instruct preset, used as HiDream-I1's fourth text encoder (text_encoder_4).
    /// 32 layers, hidden=4096, GQA 32:8, head_dim=128, intermediate=14336, vocab=128256. Standard Llama:
    /// NO per-head Q/K norm (distinguishes it from Qwen3-8B, which is otherwise dimensionally similar),
    /// no attention bias, theta=500,000 (Llama-3 family — NOT the 1M Qwen uses), RMSNorm eps=1e-5.
    /// HiDream harvests hidden states from all 32 layers (<see cref="Denoisers.HiDreamConfig.LlamaLayers"/>
    /// indexes 0..31), so the layer count must be exactly 32.
    /// <para>RoPE scaling: Llama-3.1 technically uses the "llama3" inv-freq scaling (factor 8, original
    /// context 8192), but we run it as standard RoPE (<see cref="RopeScalingType.None"/>) — same decision
    /// as <see cref="Ministral3B"/>. The scaling only diverges from identity for long contexts; HiDream
    /// feeds the encoder ≤256-token prompts where the unscaled table is effectively exact. Revisit if
    /// end-to-end output shows positional drift.</para>
    /// <para>Matches Comfy-Org/HiDream-I1_ComfyUI <c>llama_3.1_8b_instruct_fp8_scaled.safetensors</c>
    /// (HasFinalNorm=true — the repackaged encoder ships <c>model.norm.weight</c>).</para></summary>
    public static LlamaStyleEncoderConfig Llama31_8B => new()
    {
        HiddenSize = 4096,
        NumLayers = 32,
        NumQueryHeads = 32,
        NumKvHeads = 8,
        HeadDim = 128,
        IntermediateSize = 14336,
        VocabSize = 128256,
        RmsNormEps = 1e-5f,
        RopeTheta = 500_000f,
        MaxPositionEmbeddings = 131072,
        QkHeadNorm = false,
        AttentionBias = false,
        HasFinalNorm = true,
        EosTokenId = 128_001,
        BosTokenId = 128_000,
    };

    /// <summary>Mistral-Small-3 (BFL Flux.2 Dev distill): 30 layers, hidden=5120, GQA 32:8, head_dim=128, IntermediateSize=32768 (~6.4× ratio — wider FFN than standard Mistral), vocab=131072 (Tekken). No per-head Q/K norm, no final norm — ships as a feature extractor. Verified against <c>Comfy-Org/Flux2/text_encoders/mistral_3_small_flux2_fp8.safetensors</c>.</summary>
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

    /// <summary>Ministral-3B as packaged for ERNIE-Image (Baidu): 26 layers, hidden=3072 (matches ERNIE-Image's text_in_dim), GQA 32:8, head_dim=128, intermediate=9216, vocab=131072 (Tekken). YaRN-scaled RoPE with theta=1M (unscaled table is fine for ≤4096 tokens). No per-head Q/K norm; HasFinalNorm=true for the Comfy-Org safetensors which ship the full encoder.</summary>
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
    /// (4 RMSNorms per block). This preset turns on <see cref="Activation"/>=GeluTanh,
    /// <see cref="HasFfnSandwichNorms"/>=true, and <see cref="RmsNormScalePlusOne"/>=true so the
    /// encoder produces Gemma-2-correct hidden states up to the soft-cap and sliding-window
    /// approximations (the encoder uses full-attention without logit clipping; small numerical drift
    /// is expected for long-context prompts but the structural shape and feature-extractor behavior
    /// match). Verified against <c>Alpha-VLLM/Lumina-Image-2.0/text_encoder/config.json</c>.</summary>
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
        Activation = MlpActivation.GeluTanh,
        RmsNormScalePlusOne = true,
        HasFfnSandwichNorms = true,
        EosTokenId = 1,
        BosTokenId = 2,
    };

    /// <summary>GPT-OSS preset (<c>openai/gpt-oss-20b</c>, re-published in <c>microsoft/Lens/text_encoder/</c>).
    /// 24 layers, hidden=2880, GQA 64:8, head_dim=64 (Q proj output = 4096 ≠ hidden=2880), intermediate=2880
    /// per expert × 32 experts × top-4 routing, vocab=201088, RMSNorm eps=1e-5, YaRN-extended RoPE
    /// (theta=150000, factor=32, beta_fast=32, beta_slow=1, original_max=4096), attention biases enabled,
    /// per-layer attention sinks, alternating sliding-128/full-causal attention, clamped GPT-OSS-style
    /// gated MLP (clamp=7.0, α=1.702). Used by Microsoft Lens as its conditioning text encoder.</summary>
    public static LlamaStyleEncoderConfig GptOss => new()
    {
        HiddenSize = 2880,
        NumLayers = 24,
        NumQueryHeads = 64,
        NumKvHeads = 8,
        HeadDim = 64,
        IntermediateSize = 2880,
        VocabSize = 201088,
        RmsNormEps = 1e-5f,
        RopeTheta = 150_000f,
        MaxPositionEmbeddings = 131072,
        QkHeadNorm = false,
        AttentionBias = true,
        HasFinalNorm = true,
        Activation = MlpActivation.Silu,
        // ── GPT-OSS-specific ──
        LayerAttentionTypes = BuildAlternatingAttentionTypes(numLayers: 24, startsWithSliding: true),
        SlidingWindow = 128,
        NumLocalExperts = 32,
        NumExpertsPerToken = 4,
        HasAttentionSinks = true,
        ClampedSwiGlu = true,
        ClampedSwiGluLimit = 7.0f,
        ClampedSwiGluAlpha = 1.702f,
        RopeScaling = RopeScalingType.Yarn,
        YarnFactor = 32.0f,
        YarnBetaFast = 32.0f,
        YarnBetaSlow = 1.0f,
        YarnOriginalMaxPosition = 4096,
        EosTokenId = 200002,
        BosTokenId = 199998,
    };

    /// <summary>Builds the canonical GPT-OSS alternating attention layer-type array
    /// (<c>sliding, full, sliding, full, ...</c>) of length <paramref name="numLayers"/>.</summary>
    public static string[] BuildAlternatingAttentionTypes(int numLayers, bool startsWithSliding)
    {
        string[] result = new string[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            bool sliding = (i % 2 == 0) == startsWithSliding;
            result[i] = sliding ? "sliding_attention" : "full_attention";
        }
        return result;
    }
}
