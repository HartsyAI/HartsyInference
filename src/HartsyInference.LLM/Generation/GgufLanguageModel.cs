using HartsyInference.Core.Tensors;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelHandler.Gguf;
using HartsyInference.Tokenizers;

namespace HartsyInference.LLM.Generation;

/// <summary>Loads a GGUF decoder-LLM checkpoint (Qwen2.5 / Qwen3 / Llama family) into a ready-to-run
/// <see cref="GenericTransformer"/>. Owns the underlying GGUF handle (the quantized projection weights are
/// memory-mapped from the file and consumed lazily by the matmul backend), so this object must outlive any
/// inference that uses <see cref="Transformer"/>.
///
/// <para>Pipeline: <c>GgufModelLoader.Load</c> (parse + key-remap, still quantized) →
/// <see cref="GgufConfigFactory.FromGguf"/> (architecture config) → <see cref="GenericTransformer.LoadWeights"/>
/// (embed + norms to F32, projections kept quantized).</para></summary>
public sealed class GgufLanguageModel : IDisposable
{
    private readonly GgufModelLoader.LoadedGgufModel _handle;
    private int _disposed;

    /// <summary>GGUF quant formats the CUDA path supports directly (dequant-to-F16 and the fused GEMV); other
    /// quant tensors are dequantized to F32 at load.</summary>
    private static readonly HashSet<string> GpuSupportedQuant = ["Q8_0", "Q4_0", "Q5_0", "Q4_K", "Q5_K", "Q6_K"];

    /// <summary>The architecture config inferred from the GGUF metadata + weights.</summary>
    public TransformerConfig Config { get; }

    /// <summary>The ready-to-run transformer (weights loaded).</summary>
    public GenericTransformer Transformer { get; }

    /// <summary>The GGUF <c>general.architecture</c> string (e.g. "qwen2", "qwen3", "llama").</summary>
    public string Architecture => _handle.Architecture;

    /// <summary>The tokenizer built from the GGUF's embedded vocab/merges/special tokens.</summary>
    public ILlmTokenizer Tokenizer { get; }

    /// <summary>The chat template — the model's own Jinja <c>chat_template</c> when present, else ChatML.</summary>
    public IChatTemplate Template { get; }

    private GgufLanguageModel(GgufModelLoader.LoadedGgufModel handle, TransformerConfig config,
        GenericTransformer transformer, ILlmTokenizer tokenizer, IChatTemplate template)
    {
        _handle = handle;
        Config = config;
        Transformer = transformer;
        Tokenizer = tokenizer;
        Template = template;
    }

    /// <summary>Architectures that are NOT <see cref="GenericTransformer"/> decoders — dense/MoE-transformer
    /// config inference (head_count etc.) is meaningless for these and throws (e.g. mamba2's
    /// <c>attention.head_count</c> is 0, so deriving head_dim = hidden/heads divides by zero). Route them
    /// through <see cref="Ssm.SsmLanguageModel"/> instead.</summary>
    internal static readonly HashSet<string> SsmArchitectures = ["mamba", "mamba2", "rwkv6", "rwkv", "rwkv7"];

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

        // SentencePiece (tokenizer.ggml.model == "llama"): Gemma, Llama-1/2. Score-driven merges + byte fallback,
        // a different algorithm from byte-level BPE, so route to the SPM tokenizer.
        string tokModel = meta.GetString("tokenizer.ggml.model") ?? "gpt2";
        if (tokModel == "llama")
        {
            float[]? scores = meta.GetFloatArray("tokenizer.ggml.scores")
                ?? throw new NotSupportedException("SentencePiece GGUF missing tokenizer.ggml.scores.");
            bool addSpacePrefix = meta.GetBool("tokenizer.ggml.add_space_prefix", true);
            return new SpmGgufTokenizer(tokens, scores, tokenType, bos, eos, extraStops, addSpacePrefix);
        }

        // RWKV World (tokenizer.ggml.model == "rwkv"): fixed byte-string vocab, greedy longest-prefix trie
        // match — no BPE merges at all (mamba/rwkv GGUFs have none), so this must run before the "no merges"
        // NotSupportedException below.
        if (tokModel == "rwkv")
        {
            return new RwkvWorldTokenizer(tokens, tokenType, bos, eos, extraStops);
        }

        // Pre-tokenizer family decides the regex split + ignore_merges. The GPT-2/Qwen default keeps word
        // spaces and newline runs; the Llama-3 family (llama-bpe) uses a different split (case-insensitive
        // contractions, digits in groups of ≤3, newline-aware whitespace) and emits whole in-vocab pre-tokens
        // directly (ignore_merges). Wrong split → wrong token ids → garbage output.
        string pre = meta.GetString("tokenizer.ggml.pre") ?? "default";
        bool llama3Family = pre is "llama-bpe" or "llama3" or "smaug-bpe";
        bool gpt4o = pre is "gpt-4o" or "o200k";   // o200k_base split (Phi-4, GPT-OSS, GPT-4o)
        string? preRegex = llama3Family ? Llama3PreTokenRegex : gpt4o ? Gpt4oPreTokenRegex : null;
        return new GgufTokenizer(tokens, merges, tokenType, bos, eos, extraStops, preRegex, ignoreMerges: llama3Family);
    }

    /// <summary>Compiles the model's own Jinja <c>chat_template</c> when present, but stays tolerant: some models
    /// (e.g. embedding models) ship templates using constructs the engine doesn't support (Python slicing). Those
    /// models are still usable for raw completion / embeddings, so fall back to ChatML rather than failing the
    /// whole load.</summary>
    internal static IChatTemplate BuildTemplate(GgufMetadata meta)
    {
        string? chatTemplate = meta.GetString("tokenizer.chat_template");
        if (string.IsNullOrWhiteSpace(chatTemplate)) return new ChatMlTemplate();
        try { return new JinjaChatTemplate(chatTemplate); }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[WRN] GGUF: chat template did not compile ({ex.Message}); falling back to ChatML (raw completion / embeddings still work).");
            return new ChatMlTemplate();
        }
    }

    // Llama-3 / GPT-4 byte-level pre-token split (matches llama.cpp LLAMA3 + HF tokenizer.json).
    private const string Llama3PreTokenRegex =
        @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    // o200k_base / gpt-4o byte-level pre-token split (tiktoken o200k pat_str — Phi-4, GPT-OSS, GPT-4o).
    private const string Gpt4oPreTokenRegex =
        @"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    /// <summary>Loads the GGUF at <paramref name="path"/>. Set <paramref name="lowVramQuant"/> to keep weights
    /// compressed on-device (low VRAM, slower decode) instead of caching dequantized F16 weights. Set
    /// <paramref name="dequantizeToF32"/> for the CPU backend, which is F32-only and cannot consume the quantized
    /// projection weights the CUDA path keeps compressed — every quantized/half tensor is widened to F32 at load.</summary>
    public static GgufLanguageModel Load(string path, bool lowVramQuant = false, bool dequantizeToF32 = false)
    {
        GgufModelLoader.LoadedGgufModel handle = GgufModelLoader.Load(path);
        try
        {
            if (SsmArchitectures.Contains(handle.Architecture))
                throw new NotSupportedException($"'{handle.Architecture}' is a state-space (non-transformer) architecture — load it via HartsyInference.LLM.Ssm.SsmLanguageModel, not GgufLanguageModel.");

            // GGUF stores matrix dims in [in, out] order (data already row-major [out, in], same as HF
            // safetensors). Relabel every rank-2 tensor's shape to [out, in] so the matmul backend (which reads
            // N=Shape[0], K=Shape[1]) and the embed/lm_head ([vocab, hidden]) see the convention they expect.
            // Pure metadata swap, no data movement; the tensors keep borrowing the GGUF mmap.
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
                if (t.DType.IsQuantized && (dequantizeToF32 || !GpuSupportedQuant.Contains(t.DType.Name)))
                    t = GgufDequantizer.Dequantize(t, DType.F32);
                else if (dequantizeToF32 && t.DType.IsFloatingPoint && t.DType != DType.F32)
                    t = t.CastTo(DType.F32);
                weights[kv.Key] = t;
            }

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
            GenericTransformer transformer = new(config);
            transformer.LoadWeights(weights, "model");

            ILlmTokenizer tokenizer = BuildTokenizer(handle.Metadata);
            IChatTemplate template = BuildTemplate(handle.Metadata);

            return new GgufLanguageModel(handle, config, transformer, tokenizer, template);
        }
        catch
        {
            handle.Dispose();
            throw;
        }
    }

    /// <summary>Splits Phi-3's fused <c>qkv_proj</c> (→ q/k/v) and <c>gate_up_proj</c> (→ gate/up) into the
    /// separate per-projection tensors. Rows are contiguous in the (possibly quantized) layout, so each split is
    /// a byte-range copy that preserves the dtype.</summary>
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

    /// <summary>Splits GPT-2's fused <c>qkv_proj</c> (weight + bias) into separate q/k/v projections. GPT-2 is MHA
    /// and lays the fused tensor out as <c>[full_q | full_k | full_v]</c> contiguous rows, so each split is a
    /// byte-range view (weights, dtype-preserving) or a float slice (the F32 bias).</summary>
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

    /// <summary>Zero-copy view of <paramref name="count"/> elements of a 1-D tensor starting at <paramref name="start"/>
    /// (used to split a fused F32 bias vector).</summary>
    private static unsafe Tensor SliceVec(Tensor src, int start, int count)
    {
        long elemBytes = src.DType.ComputeByteCount(1);
        byte* s = (byte*)src.DataPointer + (long)start * elemBytes;
        return new Tensor((void*)s, new TensorShape(count), src.DType, src.Device);
    }

    /// <summary>Splits each MoE layer's stacked expert tensors (<c>gate_exps</c>/<c>up_exps</c>/<c>down_exps</c>,
    /// shape <c>[E, ·, ·]</c>) into the per-expert 2D <c>experts.{i}.{gate,up,down}_proj</c> weights the MoE block
    /// loads. Each expert occupies a contiguous block, so a flatten-to-2D + per-expert row-byte copy works on the
    /// quantized weights with no dequant.</summary>
    private static void SplitStackedExperts(Dictionary<string, Tensor> w, TransformerConfig cfg)
    {
        int e = cfg.Moe!.NumExperts;
        int hidden = cfg.HiddenSize;
        int inter = cfg.Moe.MoeIntermediateSize;
        for (int i = 0; i < cfg.NumLayers; i++)
        {
            if (!cfg.IsMoeLayer(i)) continue;
            string p = $"model.layers.{i}.mlp";
            SplitExperts(w, $"{p}.gate_exps.weight", $"{p}.experts.{{0}}.gate_proj.weight", e, inter, hidden);
            SplitExperts(w, $"{p}.up_exps.weight", $"{p}.experts.{{0}}.up_proj.weight", e, inter, hidden);
            SplitExperts(w, $"{p}.down_exps.weight", $"{p}.experts.{{0}}.down_proj.weight", e, hidden, inter);
        }
    }

    /// <summary>Splits one stacked expert tensor (flattened to <c>[E·outDim, inDim]</c>) into <paramref name="e"/>
    /// per-expert <c>[outDim, inDim]</c> tensors keyed by <paramref name="targetFormat"/> (a <c>{0}</c> expert-index
    /// placeholder), then removes the stacked source.</summary>
    private static void SplitExperts(Dictionary<string, Tensor> w, string stackedKey, string targetFormat,
        int e, int outDim, int inDim)
    {
        if (!w.TryGetValue(stackedKey, out Tensor? stacked)) return;
        Tensor flat = stacked.Reshape(new TensorShape(e * outDim, inDim));
        for (int x = 0; x < e; x++)
            w[targetFormat.Replace("{0}", x.ToString())] = SliceRows(flat, x * outDim, outDim);
        w.Remove(stackedKey);
    }

    /// <summary>Copies rows <c>[startRow, startRow+numRows)</c> of a rank-2 tensor into a new tensor of the same
    /// dtype. A row is contiguous in both float and block-quantized layouts (quant blocks run along the column
    /// dim), so this is a plain byte-range copy that works on quantized weights.</summary>
    private static unsafe Tensor SliceRows(Tensor src, int startRow, int numRows)
    {
        int inDim = (int)src.Shape[1];
        long rowBytes = src.DType.ComputeByteCount(inDim);
        // Zero-copy view over the source's memory (the GGUF mmap). Splitting a large MoE checkpoint into hundreds
        // of per-expert tensors must not duplicate the weights in host RAM — the views borrow the mmap, which the
        // GgufLanguageModel keeps alive, and each is uploaded to the GPU exactly once like any other weight.
        byte* s = (byte*)src.DataPointer + (long)startRow * rowBytes;
        return new Tensor((void*)s, new TensorShape(numRows, inDim), src.DType, src.Device);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        Transformer.Dispose();
        _handle.Dispose();
    }
}
