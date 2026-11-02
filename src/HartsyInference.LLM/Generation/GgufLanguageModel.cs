using HartsyInference.Core.Tensors;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Generation;

/// <summary>Loads a GGUF decoder-LLM checkpoint (Qwen2.5 / Qwen3 / Llama family) into a ready-to-run <see cref="GenericTransformer"/>; owns the underlying GGUF handle (quantized projection weights are memory-mapped and consumed lazily), so this object must outlive any inference using <see cref="Transformer"/>.</summary>
/// <remarks>Pipeline: <c>GgufModelLoader.Load</c> (parse + key-remap, still quantized) → <see cref="GgufConfigFactory.FromGguf"/> (architecture config) → <see cref="GenericTransformer.LoadWeights"/> (embed + norms to F32, projections kept quantized).</remarks>
public sealed class GgufLanguageModel : IDisposable
{
    private readonly GgufModelLoader.LoadedGgufModel? _handle;
    private readonly IDisposable? _owner;
    private int _disposed;

    /// <summary>GGUF quant formats the CUDA path supports directly (dequant-to-F16 and the fused GEMV); other quant tensors are dequantized to F32 at load.</summary>
    private static readonly HashSet<string> GpuSupportedQuant = ["Q8_0", "Q4_0", "Q5_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"];

    /// <summary>Whether a quantized tensor of this type stays compressed for a quantized-capable backend; any other quantized type is dequantized to F32 on load.</summary>
    public static bool KeepsQuantizedOnGpu(string dtypeName) => GpuSupportedQuant.Contains(dtypeName);

    /// <summary>The architecture config inferred from the GGUF metadata + weights.</summary>
    public TransformerConfig Config { get; }

    /// <summary>The ready-to-run transformer (weights loaded).</summary>
    public GenericTransformer Transformer { get; }

    /// <summary>The GGUF <c>general.architecture</c> string (e.g. "qwen2", "qwen3", "llama").</summary>
    public string Architecture => _handle?.Architecture ?? Config.Architecture ?? string.Empty;

    /// <summary>The tokenizer built from the GGUF's embedded vocab/merges/special tokens.</summary>
    public ILlmTokenizer Tokenizer { get; }

    /// <summary>The chat template — the model's own Jinja <c>chat_template</c> when present, else ChatML.</summary>
    public IChatTemplate Template { get; }

    private GgufLanguageModel(GgufModelLoader.LoadedGgufModel? handle, IDisposable? owner, TransformerConfig config,
        GenericTransformer transformer, ILlmTokenizer tokenizer, IChatTemplate template)
    {
        _handle = handle;
        _owner = owner;
        Config = config;
        Transformer = transformer;
        Tokenizer = tokenizer;
        Template = template;
    }

    /// <summary>Wraps a transformer built from a Hugging Face directory (no GGUF) in the same model object the Text path serves, so placement, pipelines and disposal are one code path. <paramref name="owner"/> holds the weights' storage and is disposed with the model.</summary>
    public static GgufLanguageModel FromWeights(TransformerConfig config, GenericTransformer transformer, ILlmTokenizer tokenizer, IChatTemplate template, IDisposable owner)
        => new(null, owner, config, transformer, tokenizer, template);

    /// <summary>Architectures that are NOT <see cref="GenericTransformer"/> decoders — dense/MoE-transformer config inference (head_count etc.) is meaningless and throws for these (e.g. mamba2's <c>attention.head_count</c> is 0, so head_dim = hidden/heads divides by zero); route them through <see cref="Ssm.SsmLanguageModel"/> instead.</summary>
    internal static readonly HashSet<string> SsmArchitectures = ["mamba", "mamba2", "rwkv6", "rwkv", "rwkv7", "qwen35", "qwen35moe"];

    internal static ILlmTokenizer BuildTokenizer(GgufMetadata meta)
    {
        string[]? tokens = meta.GetStringArray("tokenizer.ggml.tokens");
        if (tokens is null) throw new NotSupportedException("GGUF has no embedded tokenizer (tokenizer.ggml.tokens missing).");
        string[]? merges = meta.GetStringArray("tokenizer.ggml.merges");
        int[]? tokenType = meta.GetIntArray("tokenizer.ggml.token_type");
        int? bos = meta.ContainsKey("tokenizer.ggml.bos_token_id") ? (int)meta.GetUInt32("tokenizer.ggml.bos_token_id") : null;
        int? eos = meta.ContainsKey("tokenizer.ggml.eos_token_id") ? (int)meta.GetUInt32("tokenizer.ggml.eos_token_id") : null;
        int? eot = meta.ContainsKey("tokenizer.ggml.eot_token_id") ? (int)meta.GetUInt32("tokenizer.ggml.eot_token_id") : null;
        List<int> extraStops = [];
        if (eot is int e) extraStops.Add(e);

        // SentencePiece (tokenizer.ggml.model == "llama" for Gemma-1/2/3 and Llama-1/2; "gemma4" is the same
        // SPM/Unigram algorithm under a new model-field label). Score-driven merges + byte fallback, a different
        // algorithm from byte-level BPE — falling through to the generic BPE tokenizer below silently drops the
        // "▁" → space substitution, producing coherent-but-unspaced output ("Thecapitalof...").
        string tokModel = meta.GetString("tokenizer.ggml.model") ?? "gpt2";
        if (tokModel is "llama" or "gemma4")
        {
            float[]? scores = meta.GetFloatArray("tokenizer.ggml.scores")
                ?? throw new NotSupportedException("SentencePiece GGUF missing tokenizer.ggml.scores.");
            bool addSpacePrefix = meta.GetBool("tokenizer.ggml.add_space_prefix", true);
            // Only "first" changes behavior; "always" is the existing default and "never" has not appeared next to add_space_prefix.
            bool prefixOnlyAtTextStart = string.Equals(meta.GetString("tokenizer.ggml.metaspace.prepend_scheme"), "first", StringComparison.Ordinal);
            return new SpmGgufTokenizer(tokens, scores, tokenType, bos, eos, extraStops, addSpacePrefix, prefixOnlyAtTextStart);
        }

        // RWKV World (tokenizer.ggml.model == "rwkv"): fixed byte-string vocab, greedy longest-prefix trie
        // match — no BPE merges at all (mamba/rwkv GGUFs have none), so this must run before the "no merges"
        // NotSupportedException below.
        if (tokModel == "rwkv")
        {
            return new RwkvWorldTokenizer(tokens, tokenType, bos, eos, extraStops);
        }

        // llama.cpp's list of every end-of-generation id (Flint ends turns on several control tokens). Only the byte-level BPE path reads it, so
        // SentencePiece and RWKV GGUFs that carry the key keep their stops.
        if (meta.GetIntArray("tokenizer.ggml.eog_token_ids") is { } eog) extraStops.AddRange(eog);

        // A wrong split gives wrong token ids, so the pre-tokenizer name fixes both the split and ignore_merges.
        string pre = meta.GetString("tokenizer.ggml.pre") ?? "default";
        return new GgufTokenizer(tokens, merges, tokenType, bos, eos, extraStops, PreTokenRegexFor(pre),
            ignoreMerges: PreTokenizerFamilyFor(pre) is PreTokenizerFamily.Llama3 or PreTokenizerFamily.Tekken,
            preTokenizer: PreTokenizerPipelineFor(pre));
    }

    /// <summary>The Hugging Face <c>pre_tokenizer</c> of Flint (digit groups, CJK runs, then words), which no single split regex expresses.</summary>
    private const string FlintPreTokenizerJson = """{"type": "Sequence", "pretokenizers": [{"type": "Split", "pattern": {"Regex": "\\p{N}{1,3}"}, "behavior": "Isolated", "invert": false}, {"type": "Split", "pattern": {"Regex": "[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+"}, "behavior": "Isolated", "invert": false}, {"type": "Split", "pattern": {"Regex": "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?| ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false}, {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}]}""";

    private static readonly Lazy<PreTokenizerPipeline> FlintPipeline = new(() =>
    {
        using System.Text.Json.JsonDocument doc = System.Text.Json.JsonDocument.Parse(FlintPreTokenizerJson);
        return PreTokenizerPipeline.FromJson(doc.RootElement);
    });

    /// <summary>The multi-stage pre-tokenizer a <c>tokenizer.ggml.pre</c> name stands for, or null when a single regex (or the default) is enough.</summary>
    internal static PreTokenizerPipeline? PreTokenizerPipelineFor(string pre) => pre == "flint" ? FlintPipeline.Value : null;

    /// <summary>The pre-tokenizer families that choose a split regex.</summary>
    internal enum PreTokenizerFamily { Default, Llama3, Gpt4o, Qwen2, Tekken }

    /// <summary>Maps a <c>tokenizer.ggml.pre</c> name to its family. <c>qwen2</c> is llama.cpp's QWEN2 split; <c>tekken</c> and <c>pixtral</c>
    /// (Mistral Nemo, Pixtral, and the NVIDIA Nemotron Dory GGUF, which writes <c>pixtral</c>) are its TEKKEN/PIXTRAL split.</summary>
    internal static PreTokenizerFamily PreTokenizerFamilyFor(string pre) => pre switch
    {
        "llama-bpe" or "llama3" or "smaug-bpe" => PreTokenizerFamily.Llama3,
        "gpt-4o" or "o200k" => PreTokenizerFamily.Gpt4o,
        "qwen2" or "kolibri1" => PreTokenizerFamily.Qwen2,
        "tekken" or "pixtral" => PreTokenizerFamily.Tekken,
        _ => PreTokenizerFamily.Default,
    };

    /// <summary>The split regex for a pre-tokenizer name, or null for the GPT-2 default.</summary>
    internal static string? PreTokenRegexFor(string pre) => PreTokenizerFamilyFor(pre) switch
    {
        PreTokenizerFamily.Llama3 => Llama3PreTokenRegex,
        PreTokenizerFamily.Gpt4o => Gpt4oPreTokenRegex,
        PreTokenizerFamily.Qwen2 => Qwen2PreTokenRegex,
        PreTokenizerFamily.Tekken => TekkenPreTokenRegex,
        _ => null,
    };

    /// <summary>Compiles the model's own Jinja <c>chat_template</c> when present, but stays tolerant: some models ship templates using constructs the engine doesn't support (Python slicing), so falls back to ChatML (when the tokenizer has <c>&lt;|im_start|&gt;</c>/<c>&lt;|im_end|&gt;</c>) or <see cref="RawCompletionTemplate"/> rather than failing the whole load — a bare <see cref="GenerationRequest.Prompt"/> should never crash regardless of what turn format the model natively wants.</summary>
    internal static IChatTemplate BuildTemplate(GgufMetadata meta, ILlmTokenizer tokenizer)
    {
        IChatTemplate ChatMlOrRaw() =>
            tokenizer.SpecialId("<|im_start|>") is not null && tokenizer.SpecialId("<|im_end|>") is not null
                ? new ChatMlTemplate() : new RawCompletionTemplate();

        string? chatTemplate = meta.GetString("tokenizer.chat_template");
        if (string.IsNullOrWhiteSpace(chatTemplate)) return ChatMlOrRaw();

        // Some llama.cpp GGUF converters (RWKV-World among them) store a bare format-name sentinel here
        // (e.g. "rwkv-world") instead of real Jinja source — a signal meant for llama.cpp's own hardcoded
        // prompt-format table, not a template to render. A string with no "{{"/"{%" substitution syntax can
        // never actually incorporate the conversation, so JinjaEngine "compiles" it fine and every render
        // silently returns that literal constant text regardless of the real messages (confirmed live: three
        // unrelated prompts against an RWKV-6 GGUF all produced byte-identical output). Detect that up front
        // rather than letting it "succeed" — fall back the same as a genuine compile failure.
        if (!chatTemplate.Contains("{{", StringComparison.Ordinal) && !chatTemplate.Contains("{%", StringComparison.Ordinal))
        {
            Console.Error.WriteLine($"[WRN] GGUF: chat template \"{chatTemplate}\" has no Jinja substitution syntax (likely a format-name sentinel, not real template source); falling back (raw completion / embeddings still work).");
            return ChatMlOrRaw();
        }

        // llama.cpp strips a BOS the template rendered first, then its tokenizer adds one, so such a prompt starts with exactly one.
        bool prependBos = meta.GetBool("tokenizer.ggml.add_bos_token", false);
        try { return new JinjaChatTemplate(chatTemplate, prependBos); }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[WRN] GGUF: chat template did not compile ({ex.Message}); falling back (raw completion / embeddings still work).");
            return ChatMlOrRaw();
        }
    }

    internal const string Qwen2PreTokenRegex = GgufTokenizer.Qwen2PreTokenRegex;

    // Tekken byte-level pre-token split (Mistral's tokenizer.json, llama.cpp TEKKEN/PIXTRAL).
    private const string TekkenPreTokenRegex =
        @"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    // Llama-3 / GPT-4 byte-level pre-token split (matches llama.cpp LLAMA3 + HF tokenizer.json).
    private const string Llama3PreTokenRegex =
        @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    // o200k_base / gpt-4o byte-level pre-token split (tiktoken o200k pat_str — Phi-4, GPT-OSS, GPT-4o).
    private const string Gpt4oPreTokenRegex =
        @"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    /// <summary>Checkpoint loaded for TENSOR-PARALLEL construction: the raw (relabeled/split) weight dict plus config/tokenizer/template, with NO <see cref="GenericTransformer"/>; <c>TensorParallelTransformer.LoadWeights</c> copies per-rank slices out of <see cref="Weights"/>, and dict tensors keep borrowing this object's GGUF mmap so it must stay alive as long as the TP transformer does.</summary>
    public sealed class TpCheckpoint : IDisposable
    {
        private readonly GgufModelLoader.LoadedGgufModel _handle;
        private int _disposed;

        /// <summary>The GGUF architecture the checkpoint was loaded as (for example <c>llama</c>).</summary>
        public string Architecture => _handle.Architecture;

        internal TpCheckpoint(GgufModelLoader.LoadedGgufModel handle, Dictionary<string, Tensor> weights,
            TransformerConfig config, ILlmTokenizer tokenizer, IChatTemplate template)
        {
            _handle = handle;
            Weights = weights;
            Config = config;
            Tokenizer = tokenizer;
            Template = template;
        }

        public IReadOnlyDictionary<string, Tensor> Weights { get; }
        public TransformerConfig Config { get; }
        public ILlmTokenizer Tokenizer { get; }
        public IChatTemplate Template { get; }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
            // Mirrors GenericTransformer's weight ownership: dict values are mmap-borrowed views or owned
            // dequant/split copies — Tensor.Dispose is safe for both.
            foreach (Tensor t in Weights.Values) t.Dispose();
            _handle.Dispose();
        }
    }

    /// <summary>Loads the checkpoint for tensor-parallel use — same relabel/dequant/fused-split pipeline as <see cref="Load"/>, stopping before <see cref="GenericTransformer"/> construction.</summary>
    public static TpCheckpoint LoadForTensorParallel(string path, bool lowVramQuant = false)
    {
        GgufModelLoader.LoadedGgufModel handle = GgufModelLoader.Load(path);
        try
        {
            (Dictionary<string, Tensor> weights, TransformerConfig config) = PrepareWeights(handle, lowVramQuant, dequantizeToF32: false);
            ILlmTokenizer tokenizer = BuildTokenizer(handle.Metadata);
            IChatTemplate template = BuildTemplate(handle.Metadata, tokenizer);
            return new TpCheckpoint(handle, weights, config, tokenizer, template);
        }
        catch
        {
            handle.Dispose();
            throw;
        }
    }

    /// <summary>Loads the GGUF at <paramref name="path"/>; <paramref name="lowVramQuant"/> keeps weights compressed on-device instead of caching dequantized F16, and <paramref name="dequantizeToF32"/> widens every quantized/half tensor to F32 for the CPU backend (F32-only, can't consume compressed projection weights). <paramref name="keepQuantized"/> exempts those quantized formats from <paramref name="dequantizeToF32"/>: a backend that reads only some block formats directly (Vulkan) keeps them packed and has the rest widened.</summary>
    public static GgufLanguageModel Load(string path, bool lowVramQuant = false, bool dequantizeToF32 = false,
        IReadOnlySet<DType>? keepQuantized = null)
    {
        GgufModelLoader.LoadedGgufModel handle = GgufModelLoader.Load(path);
        try
        {
            (Dictionary<string, Tensor> weights, TransformerConfig config) = PrepareWeights(handle, lowVramQuant, dequantizeToF32, keepQuantized);
            GenericTransformer transformer = new(config);
            transformer.LoadWeights(weights, "model");

            ILlmTokenizer tokenizer = BuildTokenizer(handle.Metadata);
            IChatTemplate template = BuildTemplate(handle.Metadata, tokenizer);

            return new GgufLanguageModel(handle, null, config, transformer, tokenizer, template);
        }
        catch
        {
            handle.Dispose();
            throw;
        }
    }

    /// <summary>The shared load core: relabel GGUF dims → dequant unsupported formats → split fused tensors → infer config; everything up to (but excluding) transformer construction.</summary>
    internal static (Dictionary<string, Tensor> Weights, TransformerConfig Config) PrepareWeights(
        GgufModelLoader.LoadedGgufModel handle, bool lowVramQuant, bool dequantizeToF32, IReadOnlySet<DType>? keepQuantized = null)
    {
        if (SsmArchitectures.Contains(handle.Architecture))
            throw new NotSupportedException($"'{handle.Architecture}' is a state-space (non-transformer) architecture — load it via HartsyInference.LLM.Ssm.SsmLanguageModel, not GgufLanguageModel.");
        if (Looped.LoopedConfigFactory.Architectures.Contains(handle.Architecture))
            throw new NotSupportedException($"'{handle.Architecture}' is a looped (latent-recurrent) architecture — load it via HartsyInference.LLM.Looped.LoopedLanguageModel, not GgufLanguageModel.");

        Dictionary<string, Tensor> weights = RelabelAndWidenWeights(handle, dequantizeToF32, keepQuantized);

        // Phi-3 fuses q/k/v into one attn_qkv tensor and gate/up into one ffn_up tensor; split them into the
        // separate projections the transformer expects (contiguous row-byte copy, dtype-preserving → works
        // on the quantized weights without dequant).
        if (handle.Architecture == "phi3") SplitFusedPhi(weights, handle.Metadata);
        if (handle.Architecture == "glm4") SplitFusedPhi(weights, handle.Metadata, "glm4");   // fused gate+up → gate/up (q/k/v already separate)
        // GPT-2 / BLOOM fuse q/k/v (weight + bias) into one attn_qkv tensor laid out [q|k|v]; split before the
        // config factory reads tensor presence (so the QKV bias is detected). Runs before FromGguf.
        if (handle.Architecture is "gpt2" or "bloom" or "gptneox") SplitFusedGpt2(weights, handle.Metadata, handle.Architecture);

        TransformerConfig config = GgufConfigFactory.FromGguf(handle.Metadata, weights, lowVramQuant);
        // MoE: GGUF stacks all experts into one 3D tensor per projection; split them into the per-expert 2D
        // weights the MoE block loads (reshape to 2D + contiguous row-byte slice, dtype-preserving).
        if (config.Moe is not null) SplitStackedExperts(weights, config);
        return (weights, config);
    }

    /// <summary>The loader stage every GGUF decoder shares: relabels each rank-2 tensor from GGUF's <c>[in, out]</c> to <c>[out, in]</c> (a metadata swap over the mmap, no data moves) and widens the quantized formats the device cannot read — every quantized and half tensor when <paramref name="dequantizeToF32"/> asks for the F32-only CPU path, except the <paramref name="keepQuantized"/> formats a backend reads packed.</summary>
    internal static Dictionary<string, Tensor> RelabelAndWidenWeights(GgufModelLoader.LoadedGgufModel handle, bool dequantizeToF32,
        IReadOnlySet<DType>? keepQuantized = null)
    {
        Dictionary<string, Tensor> weights = new(StringComparer.Ordinal);
        foreach (KeyValuePair<string, Tensor> kv in handle.Weights)
        {
            Tensor t = kv.Value;
            if (t.Shape.Rank == 2)
                t = t.Reshape(new TensorShape((int)t.Shape[1], (int)t.Shape[0]));
            // Quant formats the GPU matmul/dequant path doesn't handle (e.g. Q5_0 tensors inside a Q4_K_M
            // mix) are dequantized to F32 up front; the supported formats stay quantized for the
            // dequant-to-F16 / fused-GEMV path. The CPU backend is F32-only, so dequantizeToF32 widens
            // everything (quantized and half-float) to F32 instead.
            if (t.DType.IsQuantized && keepQuantized?.Contains(t.DType) == true)
            {
                // Kept packed: the backend reads this block format directly.
            }
            else if (t.DType.IsQuantized && (dequantizeToF32 || !GpuSupportedQuant.Contains(t.DType.Name)))
                t = GgufDequantizer.Dequantize(t, DType.F32);
            else if (dequantizeToF32 && t.DType.IsFloatingPoint && t.DType != DType.F32)
                t = t.CastTo(DType.F32);
            weights[kv.Key] = t;
        }
        return weights;
    }

    /// <summary>Splits Phi-3's fused <c>qkv_proj</c> (→ q/k/v) and <c>gate_up_proj</c> (→ gate/up) into separate per-projection tensors; rows are contiguous in the (possibly quantized) layout, so each split is a dtype-preserving byte-range copy.</summary>
    private static unsafe void SplitFusedPhi(Dictionary<string, Tensor> w, GgufMetadata meta, string arch = "phi3")
    {
        int hq = (int)meta.GetUInt32($"{arch}.attention.head_count");
        int hkv = (int)meta.GetUInt32($"{arch}.attention.head_count_kv", (uint)hq);
        int hidden = (int)meta.GetUInt32($"{arch}.embedding_length");
        // Real head_dim (key_length, else hidden/heads) — NOT rope.dimension_count, which is the (possibly
        // smaller) partial-rotary width and differs from head_dim on Phi-4-mini (96 vs 128).
        int headDim = (int)meta.GetUInt32($"{arch}.attention.key_length", 0u);
        if (headDim == 0) headDim = hidden / Math.Max(1, hq);
        int ffn = (int)meta.GetUInt32($"{arch}.feed_forward_length");
        int layers = (int)meta.GetUInt32($"{arch}.block_count");
        int qRows = hq * headDim, kvRows = hkv * headDim;

        for (int i = 0; i < layers; i++)
        {
            string p = $"model.layers.{i}";
            if (w.TryGetValue($"{p}.self_attn.qkv_proj.weight", out Tensor? qkv))
            {
                w[$"{p}.self_attn.q_proj.weight"] = SliceRows(qkv, 0, qRows);
                w[$"{p}.self_attn.k_proj.weight"] = SliceRows(qkv, qRows, kvRows);
                w[$"{p}.self_attn.v_proj.weight"] = SliceRows(qkv, qRows + kvRows, kvRows);
                w.Remove($"{p}.self_attn.qkv_proj.weight");
            }
            if (w.TryGetValue($"{p}.mlp.gate_up_proj.weight", out Tensor? gu))
            {
                w[$"{p}.mlp.gate_proj.weight"] = SliceRows(gu, 0, ffn);
                w[$"{p}.mlp.up_proj.weight"] = SliceRows(gu, ffn, ffn);
                w.Remove($"{p}.mlp.gate_up_proj.weight");
            }
        }
    }

    /// <summary>Splits GPT-2's fused <c>qkv_proj</c> (weight + bias) into separate q/k/v projections; GPT-2 is MHA and lays the fused tensor out as <c>[full_q | full_k | full_v]</c> contiguous rows, so each split is a byte-range view or float slice.</summary>
    private static unsafe void SplitFusedGpt2(Dictionary<string, Tensor> w, GgufMetadata meta, string arch)
    {
        int hidden = (int)meta.GetUInt32($"{arch}.embedding_length");
        int hq = (int)meta.GetUInt32($"{arch}.attention.head_count");
        int hkv = (int)meta.GetUInt32($"{arch}.attention.head_count_kv", (uint)hq);
        int headDim = hidden / Math.Max(1, hq);
        int layers = (int)meta.GetUInt32($"{arch}.block_count");
        int qRows = hq * headDim, kvRows = hkv * headDim;

        for (int i = 0; i < layers; i++)
        {
            string p = $"model.layers.{i}";
            if (w.TryGetValue($"{p}.self_attn.qkv_proj.weight", out Tensor? qkv))
            {
                w[$"{p}.self_attn.q_proj.weight"] = SliceRows(qkv, 0, qRows);
                w[$"{p}.self_attn.k_proj.weight"] = SliceRows(qkv, qRows, kvRows);
                w[$"{p}.self_attn.v_proj.weight"] = SliceRows(qkv, qRows + kvRows, kvRows);
                w.Remove($"{p}.self_attn.qkv_proj.weight");
            }
            if (w.TryGetValue($"{p}.self_attn.qkv_proj.bias", out Tensor? qkvb))
            {
                w[$"{p}.self_attn.q_proj.bias"] = SliceVec(qkvb, 0, qRows);
                w[$"{p}.self_attn.k_proj.bias"] = SliceVec(qkvb, qRows, kvRows);
                w[$"{p}.self_attn.v_proj.bias"] = SliceVec(qkvb, qRows + kvRows, kvRows);
                w.Remove($"{p}.self_attn.qkv_proj.bias");
            }
        }
    }

    /// <summary>Zero-copy view of <paramref name="count"/> elements of a 1-D tensor starting at <paramref name="start"/> (used to split a fused F32 bias vector).</summary>
    private static Tensor SliceVec(Tensor src, int start, int count) => src.SliceRows(start, count);

    /// <summary>Splits each MoE layer's stacked expert tensors (<c>gate_exps</c>/<c>up_exps</c>/<c>down_exps</c>, shape <c>[E, ·, ·]</c>) into the per-expert 2D <c>experts.{i}.{gate,up,down}_proj</c> weights the MoE block loads via a flatten-to-2D + per-expert row-byte copy (no dequant needed).</summary>
    /// <remarks>Internal (not private) so <c>HartsyInference.LLM.Tests</c> can regression-test the fused-vs-separate expert tensor split directly, without constructing a full synthetic GGUF file.</remarks>
    internal static void SplitStackedExperts(Dictionary<string, Tensor> w, TransformerConfig cfg)
    {
        int e = cfg.Moe!.NumExperts;
        int hidden = cfg.Moe.ExpertWidth(cfg.HiddenSize);
        int inter = cfg.Moe.MoeIntermediateSize;
        for (int i = 0; i < cfg.NumLayers; i++)
        {
            if (!cfg.IsMoeLayer(i)) continue;
            string p = $"model.layers.{i}.mlp";
            // Some checkpoints (e.g. Gemma-4-MoE) fuse gate+up into one ffn_gate_up_exps tensor instead of
            // separate ffn_gate_exps/ffn_up_exps (llama.cpp LLM_TENSOR_FFN_GATE_UP_EXPS) — prefer it when present.
            if (w.ContainsKey($"{p}.gate_up_exps.weight"))
                SplitFusedGateUpExperts(w, $"{p}.gate_up_exps.weight", $"{p}.experts.{{0}}.gate_proj.weight", $"{p}.experts.{{0}}.up_proj.weight", e, inter, hidden);
            else
            {
                SplitExperts(w, $"{p}.gate_exps.weight", $"{p}.experts.{{0}}.gate_proj.weight", e, inter, hidden);
                SplitExperts(w, $"{p}.up_exps.weight", $"{p}.experts.{{0}}.up_proj.weight", e, inter, hidden);
            }
            SplitExperts(w, $"{p}.down_exps.weight", $"{p}.experts.{{0}}.down_proj.weight", e, hidden, inter);
        }
    }

    /// <summary>Splits one fused gate+up stacked expert tensor into per-expert <c>gate_proj</c>/<c>up_proj</c> views (<see cref="StackedExperts.SplitGateUp"/>), then removes the stacked source.</summary>
    private static void SplitFusedGateUpExperts(Dictionary<string, Tensor> w, string stackedKey,
        string gateFormat, string upFormat, int e, int inter, int inDim)
    {
        if (!w.Remove(stackedKey, out Tensor? stacked)) return;
        StackedExperts.SplitGateUp(w, stacked, gateFormat, upFormat, e, inter, inDim);
    }

    /// <summary>Splits one stacked expert tensor into per-expert <c>[outDim, inDim]</c> views (<see cref="StackedExperts.Split"/>), then removes the stacked source.</summary>
    private static void SplitExperts(Dictionary<string, Tensor> w, string stackedKey, string targetFormat,
        int e, int outDim, int inDim)
    {
        if (!w.Remove(stackedKey, out Tensor? stacked)) return;
        StackedExperts.Split(w, stacked, targetFormat, e, outDim, inDim);
    }

    /// <summary>Views rows <c>[startRow, startRow+numRows)</c> of a rank-2 tensor as a new tensor of the same dtype, rooting the source; a row is contiguous in both float and block-quantized layouts (quant blocks run along the column dim), so this is a plain byte-range copy that works on quantized weights.</summary>
    private static Tensor SliceRows(Tensor src, int startRow, int numRows)
    {
        // Zero-copy view over the source's memory: splitting a large MoE checkpoint into hundreds of per-expert
        // tensors must not duplicate the weights in host RAM. The view ROOTS its source (Tensor.SliceRows), so a source
        // that owns its buffer, such as a stacked expert tensor dequantized to F32 for a backend with no quantized
        // path, outlives the split. A raw borrowed pointer here left the per-expert views reading freed memory once
        // the stacked tensor was collected, an AccessViolation on the first weight upload of any MoE GGUF on Vulkan.
        return src.SliceRows(startRow, numRows);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        Transformer.Dispose();
        _handle?.Dispose();
        _owner?.Dispose();
    }
}
