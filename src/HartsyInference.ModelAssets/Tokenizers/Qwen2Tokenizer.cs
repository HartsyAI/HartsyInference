using Microsoft.ML.Tokenizers;

namespace HartsyInference.ModelAssets.Tokenizers;

/// <summary>Qwen2/Qwen2.5(-VL) byte-level BPE tokenizer with the ChatML template, used by Lance (chat-templated T2I/T2V prompts) and HunyuanImage 2.1 (Qwen2.5-VL MLLM encoder). Qwen2.5 and Qwen3 share the same base BPE vocab/merges (151,936 entries), so the default constructor reuses the embedded Qwen3 assets; pass explicit paths when a checkpoint ships its own <c>vocab.json</c>/<c>merges.txt</c> (Lance does).</summary>
public sealed class Qwen2Tokenizer : ILlmTokenizer, IDisposable
{
    private static readonly Dictionary<string, int> SpecialTokens = new(StringComparer.Ordinal)
    {
        ["<|endoftext|>"] = EndOfTextId, ["<|im_start|>"] = ImStartId, ["<|im_end|>"] = ImEndId,
        ["<|vision_start|>"] = VisionStartId, ["<|vision_end|>"] = VisionEndId,
        ["<|vision_pad|>"] = VisionPadId, ["<|image_pad|>"] = ImagePadId, ["<|video_pad|>"] = VideoPadId,
    };
    private static readonly string[] SpecialLiteralsByLength = [.. SpecialTokens.Keys.OrderByDescending(s => s.Length)];

    /// <inheritdoc/>
    public IReadOnlyList<string> SpecialLiterals => SpecialLiteralsByLength;

    /// <summary>Vocabulary size (matches Qwen2.5-VL's <c>config.json</c>).</summary>
    public const int VocabSize = 151936;

    /// <summary>Token id for <c>&lt;|endoftext|&gt;</c> — Qwen2's BOS and conventional pad token.</summary>
    public const int EndOfTextId = 151643;

    /// <summary>Token id for <c>&lt;|im_start|&gt;</c>.</summary>
    public const int ImStartId = 151644;

    /// <summary>Token id for <c>&lt;|im_end|&gt;</c> — Qwen2's EOS.</summary>
    public const int ImEndId = 151645;

    /// <summary>Token id for <c>&lt;|vision_start|&gt;</c> (Qwen2.5-VL / Lance sequence sentinel).</summary>
    public const int VisionStartId = 151652;

    /// <summary>Token id for <c>&lt;|vision_end|&gt;</c>.</summary>
    public const int VisionEndId = 151653;

    /// <summary>Token id for <c>&lt;|vision_pad|&gt;</c>.</summary>
    public const int VisionPadId = 151654;

    /// <summary>Token id for <c>&lt;|image_pad|&gt;</c>.</summary>
    public const int ImagePadId = 151655;

    /// <summary>Token id for <c>&lt;|video_pad|&gt;</c>.</summary>
    public const int VideoPadId = 151656;

    /// <summary>Default ChatML system prompt used by Qwen2.5-VL when none is supplied.</summary>
    public const string DefaultSystemPrompt = "You are a helpful assistant.";

    private readonly Tokenizer _tokenizer;
    private readonly Lazy<GgufTokenizer> _exact;
    private int _disposed;

    /// <summary>Creates a Qwen2 tokenizer from the embedded Qwen3 vocab/merges (identical base BPE for Qwen2.5; only the chat conventions differ, which this class owns).</summary>
    public Qwen2Tokenizer()
    {
        using Stream vocabStream = EmbeddedTokenizerResources.OpenQwen3Vocab();
        using Stream mergesStream = EmbeddedTokenizerResources.OpenQwen3Merges();
        _tokenizer = BpeTokenizer.Create(vocabStream, mergesStream);
        _exact = new Lazy<GgufTokenizer>(static () => EmbeddedTokenizerResources.HasQwen3TokenizerJson
            ? Qwen3Tokenizer.EmbeddedExactTokenizer.Value
            : throw new InvalidOperationException("The embedded Qwen tokenizer.json is required for exact encoding."));
    }

    /// <summary>Creates a Qwen2 tokenizer from <c>vocab.json</c> and <c>merges.txt</c> files (e.g. the ones shipped inside a Lance checkpoint folder).</summary>
    public Qwen2Tokenizer(string vocabPath, string mergesPath)
    {
        using Stream vocabStream = File.OpenRead(vocabPath);
        using Stream mergesStream = File.OpenRead(mergesPath);
        _tokenizer = BpeTokenizer.Create(vocabStream, mergesStream);
        // The checkpoint's own vocabulary with Qwen's split rules: its merges can differ from the embedded Qwen3 set.
        _exact = new Lazy<GgufTokenizer>(() => QwenBpe.FromFiles(vocabPath, mergesPath));
    }

    /// <summary>Encodes raw text to BPE token ids — no template, no special tokens, no padding.</summary>
    public IReadOnlyList<int> EncodeRaw(string text)
    {
        ThrowIfDisposed();
        // TODO(byte-level): same bug Qwen3Tokenizer.EncodeRaw was fixed for — BpeTokenizer.Create(vocab,
        // merges) drops leading spaces (" there" -> "there" not "Ġthere"), so space-prefixed words mis-
        // tokenize. It also drops a bare newline ENTIRELY: EncodeRaw("\n") returns zero ids where HF emits 198.
        // That reaches EncodeChat through AppendBpe, so every chat template built here is short by one id per
        // newline — measured on HunyuanImage 2.1, whose 34-token diffusers template tokenizes to 33 for us.
        // Route through ByteLevelCodec.Encode(text) before EncodeToIds, then re-validate the consumers
        // (Lance T2I/T2V, HunyuanImage 2.1) since their conditioning tokens WILL change.
        return _tokenizer.EncodeToIds(text);
    }

    /// <summary>Byte-level-exact raw BPE encode (routes through <see cref="ByteLevelCodec.Encode"/>, matching the HF fast tokenizer — the same fix Qwen3Tokenizer.EncodeRaw got). Use where exact ids are load-bearing (NeuTTS prompt building); <see cref="EncodeRaw"/> keeps the legacy behavior until its consumers are revalidated.</summary>
    public IReadOnlyList<int> EncodeRawByteLevel(string text)
    {
        ThrowIfDisposed();
        return _tokenizer.EncodeToIds(ByteLevelCodec.Encode(text));
    }

    /// <summary>Hugging Face-exact encode of raw text with Qwen's split regex and NFC normalization, so hyphenated words,
    /// space-before-punctuation and CJK text tokenize as <c>transformers.AutoTokenizer</c> does. Uses this instance's
    /// <c>vocab.json</c>/<c>merges.txt</c> when it was built from files, otherwise the embedded Qwen <c>tokenizer.json</c>. Literals in <paramref name="specialTokens"/> are
    /// split out first (longest match wins) and mapped to their ids, mirroring <c>add_special_tokens</c>; no other
    /// added-token literal is recognized. <see cref="EncodeRawByteLevel"/> keeps its existing behavior for its callers.</summary>
    /// <exception cref="InvalidOperationException">Built without files and the embedded Qwen <c>tokenizer.json</c> is not available.</exception>
    public int[] EncodeExact(string text, IReadOnlyDictionary<string, int>? specialTokens = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(text);
        GgufTokenizer exact = _exact.Value;
        string normalized = text.IsNormalized(System.Text.NormalizationForm.FormC) ? text : text.Normalize(System.Text.NormalizationForm.FormC);
        if (specialTokens is null || specialTokens.Count == 0) return exact.Encode(normalized, addSpecial: false);

        string[] literals = [.. specialTokens.Keys.OrderByDescending(static k => k.Length)];
        HashSet<char> starts = [.. literals.Where(static l => l.Length > 0).Select(static l => l[0])];
        List<int> ids = new(normalized.Length / 3 + 4);
        int start = 0;
        int i = 0;
        while (i < normalized.Length)
        {
            string? hit = null;
            if (starts.Contains(normalized[i]))
                foreach (string lit in literals)
                    if (string.CompareOrdinal(normalized, i, lit, 0, lit.Length) == 0) { hit = lit; break; }
            if (hit is null)
            {
                i++;
                continue;
            }
            if (i > start) ids.AddRange(exact.Encode(normalized[start..i], addSpecial: false));
            ids.Add(specialTokens[hit]);
            i += hit.Length;
            start = i;
        }
        if (start < normalized.Length) ids.AddRange(exact.Encode(normalized[start..], addSpecial: false));
        return [.. ids];
    }

    /// <summary>Decodes token ids back to text. Control tokens (id ≥ <see cref="EndOfTextId"/>) are dropped, and the GPT-2 byte-level mapping is reversed via <see cref="ByteLevelCodec"/> (so spaces render correctly instead of leaking <c>Ġ</c>).</summary>
    public string Decode(IReadOnlyList<int> ids)
    {
        ThrowIfDisposed();
        List<int> filtered = new(ids.Count);
        foreach (int id in ids) if (id < EndOfTextId) filtered.Add(id);
        string raw = _tokenizer.Decode(filtered) ?? string.Empty;
        return ByteLevelCodec.Decode(raw);
    }

    private static readonly Dictionary<int, byte[]> SpecialBytesById =
        SpecialTokens.ToDictionary(kv => kv.Value, kv => System.Text.Encoding.UTF8.GetBytes(kv.Key));

    /// <summary>The bytes a token contributes, as <see cref="Decode"/> would produce them; a control token gives its literal only when <paramref name="includeSpecial"/> is set.</summary>
    public byte[]? TokenBytes(int id, bool includeSpecial)
    {
        ThrowIfDisposed();
        if (id >= EndOfTextId)
            return includeSpecial && SpecialBytesById.TryGetValue(id, out byte[]? literal) ? literal : [];
        if (id < 0) return [];
        return ByteLevelCodec.DecodeToBytes(_tokenizer.Decode([id]) ?? string.Empty);
    }

    // ── ILlmTokenizer ────────────────────────────────────────────────────────────────────────────────
    /// <summary>Ordinary BPE encode (no special-token handling).</summary>
    public int[] EncodeOrdinary(string text) { ThrowIfDisposed(); return [.. _tokenizer.EncodeToIds(text)]; }

    /// <summary>Special-aware encode: when <paramref name="addSpecial"/>, ChatML/vision control-token literals in <paramref name="text"/> map to their ids; otherwise plain BPE.</summary>
    public int[] Encode(string text, bool addSpecial)
    {
        ThrowIfDisposed();
        if (!addSpecial) return EncodeOrdinary(text);
        List<int> ids = new(text.Length / 3 + 8);
        int i = 0;
        while (i < text.Length)
        {
            string? hit = null;
            foreach (string lit in SpecialLiteralsByLength)
                if (i + lit.Length <= text.Length && string.CompareOrdinal(text, i, lit, 0, lit.Length) == 0) { hit = lit; break; }
            if (hit is not null) { ids.Add(SpecialTokens[hit]); i += hit.Length; continue; }
            int next = text.Length;
            foreach (string lit in SpecialLiteralsByLength) { int idx = text.IndexOf(lit, i, StringComparison.Ordinal); if (idx >= 0 && idx < next) next = idx; }
            ids.AddRange(EncodeOrdinary(text[i..next]));
            i = next;
        }
        return [.. ids];
    }

    /// <summary>Special-token id by literal, or null.</summary>
    public int? SpecialId(string token) => SpecialTokens.TryGetValue(token, out int id) ? id : null;

    /// <summary>Qwen has no dedicated BOS token.</summary>
    public int? BosId => null;

    /// <summary>Qwen2/2.5 EOS is <c>&lt;|im_end|&gt;</c>.</summary>
    public int? EosId => ImEndId;

    /// <summary>Stop ids: end-of-turn and end-of-text.</summary>
    public IReadOnlyList<int> StopIds => [ImEndId, EndOfTextId];

    /// <summary>No BOS literal.</summary>
    public string? BosToken => null;

    /// <summary>EOS literal.</summary>
    public string? EosToken => "<|im_end|>";

    /// <summary>Encodes a prompt with the Qwen2.5 ChatML template, returning the raw (unpadded, variable-length) token ids. Format: <c>&lt;|im_start|&gt;system\n{system}&lt;|im_end|&gt;\n&lt;|im_start|&gt;user\n{prompt}&lt;|im_end|&gt;</c>, plus a trailing <c>\n&lt;|im_start|&gt;assistant\n</c> when <paramref name="addGenerationPrompt"/> is true (Lance uses true; HunyuanImage 2.1's encode template uses false).</summary>
    /// <param name="systemPrompt">System message; null uses <see cref="DefaultSystemPrompt"/>. Pass an empty string to omit the system turn entirely.</param>
    public int[] EncodeChat(string prompt, string? systemPrompt = null, bool addGenerationPrompt = true)
    {
        ThrowIfDisposed();
        string system = systemPrompt ?? DefaultSystemPrompt;

        List<int> ids = new(prompt.Length / 2 + 32);
        if (system.Length > 0)
        {
            ids.Add(ImStartId);
            AppendBpe(ids, "system\n" + system);
            ids.Add(ImEndId);
            AppendBpe(ids, "\n");
        }
        ids.Add(ImStartId);
        AppendBpe(ids, "user\n" + prompt);
        ids.Add(ImEndId);
        if (addGenerationPrompt)
        {
            AppendBpe(ids, "\n");
            ids.Add(ImStartId);
            AppendBpe(ids, "assistant\n");
        }
        return ids.ToArray();
    }

    /// <summary>Right-pads (or truncates) ids to <paramref name="length"/> using <see cref="EndOfTextId"/> — the HF convention for Qwen2 models with no dedicated pad token.</summary>
    public static int[] PadToLength(int[] ids, int length)
    {
        if (length <= 0)
            throw new ArgumentOutOfRangeException(nameof(length));
        int[] result = new int[length];
        int copy = Math.Min(ids.Length, length);
        for (int i = 0; i < copy; i++)
            result[i] = ids[i];
        for (int i = copy; i < length; i++)
            result[i] = EndOfTextId;
        return result;
    }

    /// <summary>Builds a [seq] attention mask: 1 for real tokens, 0 for the trailing <see cref="EndOfTextId"/> padding run.</summary>
    public static int[] CreateAttentionMask(int[] tokenIds)
    {
        int[] mask = new int[tokenIds.Length];
        int lastReal = -1;
        for (int i = tokenIds.Length - 1; i >= 0; i--)
        {
            if (tokenIds[i] != EndOfTextId)
            {
                lastReal = i;
                break;
            }
        }
        for (int i = 0; i <= lastReal; i++)
            mask[i] = 1;
        return mask;
    }

    private void AppendBpe(List<int> dst, string text)
    {
        IReadOnlyList<int> ids = _tokenizer.EncodeToIds(text);
        for (int i = 0; i < ids.Count; i++)
            dst.Add(ids[i]);
    }

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            // Microsoft.ML.Tokenizers.Tokenizer is not IDisposable in v2.0; nothing to free.
        }
    }
}
