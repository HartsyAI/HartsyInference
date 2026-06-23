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
    private static readonly HashSet<string> GpuSupportedQuant = ["Q8_0", "Q4_K", "Q5_K", "Q6_K"];

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

    private static ILlmTokenizer BuildTokenizer(GgufMetadata meta)
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

        // Pre-tokenizer family decides the regex split + ignore_merges. The GPT-2/Qwen default keeps word
        // spaces and newline runs; the Llama-3 family (llama-bpe) uses a different split (case-insensitive
        // contractions, digits in groups of ≤3, newline-aware whitespace) and emits whole in-vocab pre-tokens
        // directly (ignore_merges). Wrong split → wrong token ids → garbage output.
        string pre = meta.GetString("tokenizer.ggml.pre") ?? "default";
        bool llama3Family = pre is "llama-bpe" or "llama3" or "smaug-bpe";
        string? preRegex = llama3Family ? Llama3PreTokenRegex : null;
        return new GgufTokenizer(tokens, merges, tokenType, bos, eos, extraStops, preRegex, ignoreMerges: llama3Family);
    }

    // Llama-3 / GPT-4 byte-level pre-token split (matches llama.cpp LLAMA3 + HF tokenizer.json).
    private const string Llama3PreTokenRegex =
        @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    /// <summary>Loads the GGUF at <paramref name="path"/>. Set <paramref name="lowVramQuant"/> to keep weights
    /// compressed on-device (low VRAM, slower decode) instead of caching dequantized F16 weights.</summary>
    public static GgufLanguageModel Load(string path, bool lowVramQuant = false)
    {
        GgufModelLoader.LoadedGgufModel handle = GgufModelLoader.Load(path);
        try
        {
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
                // dequant-to-F16 / fused-GEMV path.
                if (t.DType.IsQuantized && !GpuSupportedQuant.Contains(t.DType.Name))
                    t = GgufDequantizer.Dequantize(t, DType.F32);
                weights[kv.Key] = t;
            }

            TransformerConfig config = GgufConfigFactory.FromGguf(handle.Metadata, weights, lowVramQuant);
            GenericTransformer transformer = new(config);
            transformer.LoadWeights(weights, "model");

            ILlmTokenizer tokenizer = BuildTokenizer(handle.Metadata);
            string? chatTemplate = handle.Metadata.GetString("tokenizer.chat_template");
            IChatTemplate template = !string.IsNullOrWhiteSpace(chatTemplate)
                ? new JinjaChatTemplate(chatTemplate)
                : new ChatMlTemplate();

            return new GgufLanguageModel(handle, config, transformer, tokenizer, template);
        }
        catch
        {
            handle.Dispose();
            throw;
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        Transformer.Dispose();
        _handle.Dispose();
    }
}
