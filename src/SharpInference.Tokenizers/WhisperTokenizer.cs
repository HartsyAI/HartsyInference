using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.ML.Tokenizers;

namespace SharpInference.Tokenizers;

/// <summary>Whisper byte-level BPE tokenizer matching OpenAI / HuggingFace exactly.
/// Whisper inherited GPT-2's byte-level BPE: every input byte is first mapped to a
/// printable Unicode codepoint via the GPT-2 byte-encoder table, then run through a
/// regex pre-tokenizer, then BPE-merged. Special tokens (<c>&lt;|startoftranscript|&gt;</c>,
/// language tags, <c>&lt;|0.00|&gt;</c> timestamps, etc.) are recognized in-text and emitted
/// as fixed IDs outside the BPE merge process.
///
/// <para>Construction loads <c>vocab.json</c> + <c>merges.txt</c> + (optionally)
/// <c>added_tokens.json</c> from a per-model HuggingFace checkout. The multilingual
/// vocab is 51865 tokens (Whisper &lt;= v2 and turbo) or 51866 (Whisper v3+, +Cantonese).
/// English-only models (<c>whisper-*.en</c>) ship a different 51864-token vocab.</para>
///
/// <para>This class is the canonical SharpInference Whisper tokenizer; the audio
/// package depends on it through a project reference.</para></summary>
public sealed class WhisperTokenizer : IDisposable
{
    /// <summary>End-of-text / pad token. ID 50257 in all Whisper variants.</summary>
    public const int EndOfTextId = 50_257;

    /// <summary>Start-of-transcript token. ID 50258.</summary>
    public const int StartOfTranscriptId = 50_258;

    /// <summary>Translate-task token. ID 50358.</summary>
    public const int TranslateTokenId = 50_358;

    /// <summary>Transcribe-task token. ID 50359.</summary>
    public const int TranscribeTokenId = 50_359;

    /// <summary>No-speech token (emitted when audio has no speech). ID 50362.</summary>
    public const int NoSpeechTokenId = 50_362;

    /// <summary>No-timestamps token; suppresses the 1501 timestamp tokens. ID 50363.</summary>
    public const int NoTimestampsTokenId = 50_363;

    /// <summary>First timestamp token (0.00 s). Timestamps span 50364..51864 in 0.02 s steps.</summary>
    public const int TimestampStartId = 50_364;

    // GPT-2 pre-tokenization regex (verbatim from HuggingFace whisper tokenizer.json
    // "pre_tokenizer" section). Splits on contractions, letter runs, digit runs,
    // non-whitespace symbol runs, and whitespace.
    private static readonly Regex _gpt2PreTokenRegex = new(
        @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
        RegexOptions.Compiled);

    /// <summary>GPT-2 byte encoder: maps each byte (0..255) to a printable unicode
    /// codepoint that BPE merges can operate on. Bytes 33-126, 161-172, 174-255 map
    /// to themselves; the remaining "non-printable" bytes are remapped above U+0100.
    /// This table is identical across GPT-2, RoBERTa, Whisper, and every model that
    /// uses byte-level BPE.</summary>
    private static readonly char[] _byteToUnicode = BuildByteToUnicode();

    /// <summary>Reverse of <see cref="_byteToUnicode"/>: unicode codepoint → byte.</summary>
    private static readonly Dictionary<char, byte> _unicodeToByte = BuildUnicodeToByte();

    /// <summary>Languages supported by Whisper, in the same order as the language
    /// token IDs (50259 + index). Used by <see cref="LanguageToTokenId"/>.</summary>
    public static readonly IReadOnlyList<string> Languages = WhisperLanguageTable.Codes;

    private readonly BpeTokenizer _bpe;
    private readonly Dictionary<string, int> _specialTokens;
    private readonly Dictionary<int, string> _specialTokensReverse;
    private readonly int _vocabSize;
    private int _disposed;

    /// <summary>Total vocab size including special tokens. Use this to size embedding matrices.</summary>
    public int VocabSize => _vocabSize;

    /// <summary>Creates a Whisper tokenizer from HuggingFace-format files. Pass the
    /// directory holding <c>vocab.json</c>, <c>merges.txt</c>, and (optionally)
    /// <c>added_tokens.json</c>. Multilingual models always have <c>added_tokens.json</c>;
    /// English-only models bake their special tokens into <c>vocab.json</c>.</summary>
    public WhisperTokenizer(string modelDirectory)
    {
        string vocabPath = Path.Combine(modelDirectory, "vocab.json");
        string mergesPath = Path.Combine(modelDirectory, "merges.txt");
        if (!File.Exists(vocabPath) || !File.Exists(mergesPath))
            throw new FileNotFoundException(
                $"Whisper tokenizer requires vocab.json + merges.txt under '{modelDirectory}'. " +
                "Download a Whisper checkpoint via AudioModelCache first.");

        using Stream vocabStream = File.OpenRead(vocabPath);
        using Stream mergesStream = File.OpenRead(mergesPath);

        _specialTokens = LoadSpecialTokens(Path.Combine(modelDirectory, "added_tokens.json"));

        _bpe = BpeTokenizer.Create(
            vocabStream,
            mergesStream,
            preTokenizer: new RegexPreTokenizer(_gpt2PreTokenRegex, _specialTokens),
            normalizer: null,
            specialTokens: _specialTokens,
            unknownToken: null,
            continuingSubwordPrefix: null,
            endOfWordSuffix: null,
            fuseUnknownTokens: false);

        _vocabSize = ComputeVocabSize(vocabPath);
        _specialTokensReverse = new Dictionary<int, string>(_specialTokens.Count);
        foreach ((string tok, int id) in _specialTokens) _specialTokensReverse[id] = tok;
    }

    /// <summary>Tokenizes plain text into raw BPE token IDs (no special prefix / suffix).
    /// Use <see cref="BuildPromptIds"/> to assemble the full Whisper decoder prompt with
    /// SOT + language + task + notimestamps.</summary>
    public int[] EncodeText(string text)
    {
        ThrowIfDisposed();
        IReadOnlyList<int> ids = _bpe.EncodeToIds(text);
        int[] result = new int[ids.Count];
        for (int i = 0; i < ids.Count; i++) result[i] = ids[i];
        return result;
    }

    /// <summary>Builds the canonical Whisper decoder prompt:
    /// <c>[SOT, &lt;|lang|&gt;, transcribe/translate, &lt;|notimestamps|&gt;]</c>. Pass <c>null</c>
    /// for <paramref name="language"/> to skip the language token (English-only models)
    /// and <c>null</c> for <paramref name="task"/> to skip the task token.</summary>
    public int[] BuildPromptIds(string? language = "en", bool translate = false, bool withTimestamps = false)
    {
        ThrowIfDisposed();
        List<int> ids = new(4) { StartOfTranscriptId };
        if (language is not null)
        {
            int langId = LanguageToTokenId(language);
            ids.Add(langId);
        }
        ids.Add(translate ? TranslateTokenId : TranscribeTokenId);
        if (!withTimestamps) ids.Add(NoTimestampsTokenId);
        return ids.ToArray();
    }

    /// <summary>Returns the token ID for a Whisper language code such as <c>"en"</c>,
    /// <c>"zh"</c>, <c>"yue"</c>. Throws if the code is unknown.</summary>
    public static int LanguageToTokenId(string code)
    {
        int idx = WhisperLanguageTable.IndexOf(code);
        if (idx < 0) throw new ArgumentException($"Unknown Whisper language code '{code}'.", nameof(code));
        return 50_259 + idx;
    }

    /// <summary>Returns the language code for a Whisper language token ID (50259..50357
    /// or 50358 for Cantonese on v3). Returns <c>null</c> for non-language tokens.</summary>
    public static string? TokenIdToLanguage(int id)
    {
        int idx = id - 50_259;
        if (idx < 0 || idx >= WhisperLanguageTable.Codes.Count) return null;
        return WhisperLanguageTable.Codes[idx];
    }

    /// <summary>Decodes token IDs back to text. Drops special tokens (everything ≥ 50257
    /// and timestamp tokens) unless <paramref name="includeSpecial"/> is true.</summary>
    public string Decode(ReadOnlySpan<int> tokenIds, bool includeSpecial = false)
    {
        ThrowIfDisposed();
        List<int> bpeIds = new(tokenIds.Length);
        StringBuilder special = new();
        foreach (int id in tokenIds)
        {
            if (id >= 50_257)
            {
                if (includeSpecial && _specialTokensReverse.TryGetValue(id, out string? tok))
                    special.Append(tok);
                // Otherwise: drop. Timestamps + language tags + EOT are filtered.
                continue;
            }
            bpeIds.Add(id);
        }

        // ML.Tokenizers' Decode returns the post-merge string in the byte-level
        // unicode space — every byte of the source UTF-8 input was mapped to a
        // printable unicode codepoint via the GPT-2 byte_encoder table before BPE,
        // so we must reverse that mapping here (each char → byte) and then UTF-8
        // decode. Without this, ASCII space (0x20) shows up as 'Ġ' (U+0120) and
        // every other non-printable byte as its remapped codepoint.
        string raw = _bpe.Decode(bpeIds) ?? string.Empty;
        string text = ByteLevelDecode(raw);
        return includeSpecial && special.Length > 0 ? text + special.ToString() : text;
    }

    /// <summary>Decodes a single token ID. Useful for streaming output.</summary>
    public string DecodeOne(int tokenId)
    {
        ThrowIfDisposed();
        if (tokenId >= 50_257)
            return _specialTokensReverse.TryGetValue(tokenId, out string? tok) ? tok : string.Empty;
        string raw = _bpe.Decode([tokenId]) ?? string.Empty;
        return ByteLevelDecode(raw);
    }

    /// <summary>Reverses the GPT-2 byte_encoder mapping: each char → byte → UTF-8 string.
    /// Multi-byte UTF-8 characters (CJK, emoji, accented Latin) span multiple BPE tokens;
    /// the raw concatenation gives us a complete byte sequence that UTF-8-decodes correctly.</summary>
    private string ByteLevelDecode(string raw)
    {
        if (raw.Length == 0) return string.Empty;
        byte[] bytes = new byte[raw.Length];
        int n = 0;
        foreach (char c in raw)
        {
            if (_unicodeToByte.TryGetValue(c, out byte b))
                bytes[n++] = b;
            // Unknown codepoint: skip silently. Should not happen for valid BPE output.
        }
        return Encoding.UTF8.GetString(bytes, 0, n);
    }

    /// <summary>Returns true if the token ID is a timestamp token.</summary>
    public static bool IsTimestamp(int id) => id >= TimestampStartId && id < 51_865;

    /// <summary>Converts a timestamp token ID to seconds (0..30 s in 0.02 s steps).</summary>
    public static double TimestampToSeconds(int id) => (id - TimestampStartId) * 0.02;

    private Dictionary<string, int> LoadSpecialTokens(string addedTokensPath)
    {
        Dictionary<string, int> special = new(StringComparer.Ordinal);
        if (!File.Exists(addedTokensPath))
        {
            // English-only Whisper models bake special tokens into vocab.json directly.
            // We still register the hardcoded core tokens so they get recognized by the
            // pre-tokenizer regex.
            special["<|endoftext|>"] = EndOfTextId;
            special["<|startoftranscript|>"] = StartOfTranscriptId;
            special["<|translate|>"] = TranslateTokenId;
            special["<|transcribe|>"] = TranscribeTokenId;
            special["<|nospeech|>"] = NoSpeechTokenId;
            special["<|notimestamps|>"] = NoTimestampsTokenId;
            return special;
        }

        using FileStream fs = File.OpenRead(addedTokensPath);
        using JsonDocument doc = JsonDocument.Parse(fs);
        foreach (JsonProperty prop in doc.RootElement.EnumerateObject())
        {
            special[prop.Name] = prop.Value.GetInt32();
        }
        return special;
    }

    private static int ComputeVocabSize(string vocabPath)
    {
        // vocab.json is { "token": id, ... }. Count the entries; the size includes the
        // base BPE vocab plus any tokens added inline. Special tokens loaded from
        // added_tokens.json extend further; the caller's embedding matrix must be
        // sized to max(vocabFromJson, max(addedTokenId)+1).
        using FileStream fs = File.OpenRead(vocabPath);
        using JsonDocument doc = JsonDocument.Parse(fs);
        int max = 0;
        int count = 0;
        foreach (JsonProperty prop in doc.RootElement.EnumerateObject())
        {
            count++;
            int id = prop.Value.GetInt32();
            if (id > max) max = id;
        }
        // Whisper's multilingual checkpoints stop at 50256 in vocab.json and add 1608
        // special tokens via added_tokens.json (= 51864 total for english-only, 51865
        // for multilingual, 51866 for v3). Return max+1 to size the embedding matrix.
        return Math.Max(count, max + 1);
    }

    private static char[] BuildByteToUnicode()
    {
        // Identical to GPT-2's `bytes_to_unicode()` in tokenizer_gpt2.py.
        // Characters in [33..126] ∪ [161..172] ∪ [174..255] map to themselves; the rest
        // get assigned codepoints starting at 256.
        char[] table = new char[256];
        List<int> printables = new();
        for (int b = '!'; b <= '~'; b++) printables.Add(b);
        for (int b = 161; b <= 172; b++) printables.Add(b);
        for (int b = 174; b <= 255; b++) printables.Add(b);
        bool[] used = new bool[256];
        foreach (int b in printables) { table[b] = (char)b; used[b] = true; }
        int next = 256;
        for (int b = 0; b < 256; b++)
        {
            if (!used[b]) { table[b] = (char)next; next++; }
        }
        return table;
    }

    private static Dictionary<char, byte> BuildUnicodeToByte()
    {
        Dictionary<char, byte> map = new(256);
        char[] forward = BuildByteToUnicode();
        for (int b = 0; b < 256; b++) map[forward[b]] = (byte)b;
        return map;
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0) throw new ObjectDisposedException(nameof(WhisperTokenizer));
    }

    /// <summary>Disposes the BPE tokenizer resources.</summary>
    public void Dispose()
    {
        // BpeTokenizer doesn't itself implement IDisposable in the current ML.Tokenizers
        // release — there's nothing to free here besides the disposed flag.
        Interlocked.Exchange(ref _disposed, 1);
    }
}
