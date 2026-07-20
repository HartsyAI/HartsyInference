using System.Text.RegularExpressions;
using Microsoft.ML.Tokenizers;

namespace HartsyInference.ModelAssets.Tokenizers;

/// <summary>CLIP BPE tokenizer matching OpenAI's implementation. Wraps <see cref="Microsoft.ML.Tokenizers.BpeTokenizer"/> with CLIP-specific configuration: lowercase normalization, a regex pre-tokenizer that splits on word/contraction/punctuation boundaries, the <c>&lt;/w&gt;</c> end-of-word suffix that CLIP's BPE merges depend on, and SOT/EOT padding to 77 tokens.
///
/// Without the regex pre-tokenizer + <c>EndOfWordSuffix</c>, generic BPE produces character-level fragments (e.g. "photograph" → 'ph','ot','og','raph') instead of CLIP's correct merges (→ 'photograph&lt;/w&gt;'). That breaks every CLIP-conditioned model — see PHASE_3_DEVIATIONS #1 / SDXL silent CLIP failures and the SD3.5 patch-grid output we hit before this fix.</summary>
public sealed class ClipTokenizer : IDisposable
{
    /// <summary>Start-of-text token ID.</summary>
    public const int StartOfTextId = 49406;

    /// <summary>End-of-text token ID.</summary>
    public const int EndOfTextId = 49407;

    /// <summary>Maximum sequence length including SOT and EOT tokens.</summary>
    public const int MaxLength = 77;

    /// <summary>Vocabulary size.</summary>
    public const int VocabSize = 49408;

    /// <summary>End-of-word suffix used by CLIP's BPE merges.</summary>
    private const string EndOfWordSuffix = "</w>";

    /// <summary>OpenAI CLIP pre-tokenization regex (mirrors `huggingface/transformers` <c>CLIPTokenizer</c>):
    /// matches the special tokens, English contractions, letter sequences, single digits, and runs of
    /// non-whitespace symbols. Letter sequences use <c>\p{L}</c> so unicode letters are kept together.</summary>
    private static readonly Regex _clipPreTokenRegex = new(
        @"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    private static readonly IReadOnlyDictionary<string, int> _clipSpecialTokens =
        new Dictionary<string, int>
        {
            ["<|startoftext|>"] = StartOfTextId,
            ["<|endoftext|>"] = EndOfTextId,
        };

    private readonly Tokenizer _tokenizer;
    private int _disposed;

    /// <summary>Creates a CLIP tokenizer using the canonical OpenAI CLIP vocab/merges
    /// embedded in this assembly. This is the right constructor for SD 1.5, SDXL,
    /// SD3, Flux, and the SDXL refiner — they all share the same 49,408-token
    /// CLIP BPE.</summary>
    public ClipTokenizer()
    {
        using Stream vocabStream = EmbeddedTokenizerResources.OpenClipVocab();
        using Stream mergesStream = EmbeddedTokenizerResources.OpenClipMerges();
        _tokenizer = CreateTokenizer(vocabStream, mergesStream);
    }

    /// <summary>Creates a CLIP tokenizer from a vocabulary file and merges file.</summary>
    public ClipTokenizer(string vocabPath, string mergesPath)
    {
        using Stream vocabStream = File.OpenRead(vocabPath);
        using Stream mergesStream = File.OpenRead(mergesPath);
        _tokenizer = CreateTokenizer(vocabStream, mergesStream);
    }

    /// <summary>Creates a CLIP tokenizer from streams.</summary>
    public ClipTokenizer(Stream vocabStream, Stream mergesStream)
    {
        _tokenizer = CreateTokenizer(vocabStream, mergesStream);
    }

    private static BpeTokenizer CreateTokenizer(Stream vocabStream, Stream mergesStream)
    {
        // Argument order for this overload: vocab, merges, preTokenizer, normalizer, specialTokens, unknownToken, continuingSubwordPrefix, endOfWordSuffix, fuseUnknownTokens.
        return BpeTokenizer.Create(
            vocabStream,
            mergesStream,
            preTokenizer: new RegexPreTokenizer(_clipPreTokenRegex, _clipSpecialTokens),
            normalizer: LowercaseNormalizer.Instance,
            specialTokens: _clipSpecialTokens,
            unknownToken: null,
            continuingSubwordPrefix: null,
            endOfWordSuffix: EndOfWordSuffix,
            fuseUnknownTokens: false);
    }

    /// <summary>Tokenizes text into token IDs matching OpenAI CLIP output. Prepends SOT, appends EOT, pads with EOT to <see cref="MaxLength"/>. CLIP uses the EOT token as its pad token (matching HuggingFace <c>CLIPTokenizer(padding="max_length")</c>) — never zero-pad, since position 0 is a real token id (the byte-level "!").</summary>
    public int[] Encode(string text)
    {
        ThrowIfDisposed();

        IReadOnlyList<int> tokenIds = _tokenizer.EncodeToIds(text);

        int[] result = new int[MaxLength];
        result[0] = StartOfTextId;

        int maxTokens = MaxLength - 2; // reserve slots for SOT and EOT
        int tokenCount = Math.Min(tokenIds.Count, maxTokens);

        for (int i = 0; i < tokenCount; i++)
            result[i + 1] = tokenIds[i];

        result[tokenCount + 1] = EndOfTextId;
        // CLIP convention: pad with EOT, not zero. Mirrors HuggingFace `CLIPTokenizer.pad_token == eos_token`.
        for (int i = tokenCount + 2; i < MaxLength; i++)
            result[i] = EndOfTextId;

        return result;
    }

    /// <summary>Finds the position of the first EOT token in an encoded sequence.</summary>
    public static int FindEosPosition(ReadOnlySpan<int> tokenIds)
    {
        for (int i = 0; i < tokenIds.Length; i++)
            if (tokenIds[i] == EndOfTextId)
                return i;
        return -1;
    }

    /// <summary>Tokenizes text and returns just the BPE token IDs without padding (no SOT/EOT).</summary>
    public IReadOnlyList<int> EncodeRaw(string text)
    {
        ThrowIfDisposed();
        return _tokenizer.EncodeToIds(text);
    }

    /// <summary>Decodes token IDs back to text, filtering out SOT/EOT and zero-pad.</summary>
    public string Decode(ReadOnlySpan<int> tokenIds)
    {
        ThrowIfDisposed();
        List<int> filtered = new(tokenIds.Length);
        for (int i = 0; i < tokenIds.Length; i++)
        {
            int id = tokenIds[i];
            if (id != StartOfTextId && id != EndOfTextId && id != 0)
                filtered.Add(id);
        }
        return _tokenizer.Decode(filtered) ?? string.Empty;
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0)
            throw new ObjectDisposedException(nameof(ClipTokenizer));
    }

    /// <summary>Disposes the underlying tokenizer resources.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            if (_tokenizer is IDisposable disposable)
                disposable.Dispose();
        }
    }

    /// <summary>Lowercase-only normalizer used as the pre-BPE step. CLIP normalizes input by lowercasing — no NFKC, no whitespace cleanup beyond what the regex pre-tokenizer handles.</summary>
    private sealed class LowercaseNormalizer : Normalizer
    {
        public static readonly LowercaseNormalizer Instance = new();
        public override string Normalize(string original) => original.ToLowerInvariant();
        public override string Normalize(ReadOnlySpan<char> original) => original.ToString().ToLowerInvariant();
    }
}
