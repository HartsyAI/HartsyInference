using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.ML.Tokenizers;

namespace HartsyInference.ModelAssets.Tokenizers;

/// <summary>Whisper byte-level BPE tokenizer matching OpenAI / HuggingFace exactly. Whisper inherited GPT-2's byte-level BPE: every input byte is first mapped to a printable Unicode codepoint via the GPT-2 byte-encoder table, then run through a regex pre-tokenizer, then BPE-merged. Special tokens (<c>&lt;|startoftranscript|&gt;</c>, language tags, <c>&lt;|0.00|&gt;</c> timestamps, etc.) are recognized in-text and emitted as fixed IDs outside the BPE merge process.
///
/// <para>Construction loads <c>vocab.json</c> + <c>merges.txt</c> + (optionally) <c>added_tokens.json</c> from a per-model HuggingFace checkout. Two layouts exist and every special id is read from the checkpoint rather than assumed: the multilingual vocab is 51865 tokens (Whisper &lt;= v2 and turbo; EOT 50257, SOT 50258) or 51866 (Whisper v3+, +Cantonese), while the English-only <c>whisper-*.en</c> and <c>distil-*.en</c> checkpoints ship a 51864-token vocab in which every special sits one lower (EOT 50256, SOT 50257) and the decoder prompt carries no language or task token. Both families ship <c>added_tokens.json</c>.</para>
///
/// <para>This class is the canonical HartsyInference Whisper tokenizer; the audio package depends on it through a project reference.</para></summary>
public sealed class WhisperTokenizer : IDisposable
{
    /// <summary>End-of-text / pad token in the multilingual layout. ID 50257; English-only checkpoints use 50256 — prefer the instance <see cref="EotId"/>.</summary>
    public const int EndOfTextId = 50_257;

    /// <summary>Start-of-transcript token in the multilingual layout. ID 50258; English-only checkpoints use 50257 — prefer <see cref="SotId"/>.</summary>
    public const int StartOfTranscriptId = 50_258;

    /// <summary>Translate-task token in the multilingual &lt;=v2 layout. ID 50358. large-v3 added a 100th language (<c>&lt;|yue|&gt;</c>), which shifts this and every later special up by one, and the English-only layout sits one lower — prefer the instance <see cref="TranslateId"/>, which reads the checkpoint's own <c>added_tokens.json</c>.</summary>
    public const int TranslateTokenId = 50_358;

    /// <summary>Transcribe-task token in the multilingual &lt;=v2 layout. ID 50359. See <see cref="TranscribeId"/>.</summary>
    public const int TranscribeTokenId = 50_359;

    /// <summary>No-speech token in the multilingual &lt;=v2 layout. ID 50362. See <see cref="NoSpeechId"/>.</summary>
    public const int NoSpeechTokenId = 50_362;

    /// <summary>No-timestamps token in the multilingual &lt;=v2 layout. ID 50363. See <see cref="NoTimestampsId"/>.</summary>
    public const int NoTimestampsTokenId = 50_363;

    /// <summary>First timestamp token (0.00 s) in the multilingual &lt;=v2 layout. ID 50364. See <see cref="FirstTimestampId"/>.</summary>
    public const int TimestampStartId = 50_364;

    /// <summary>First language token (<c>&lt;|en|&gt;</c>) in the multilingual layout. ID 50259. See <see cref="FirstLanguageId"/>.</summary>
    private const int LanguageStartId = 50_259;

    /// <summary>OpenAI's multilingual test (<c>whisper/model.py is_multilingual</c>): a vocabulary of at least this many entries carries the multilingual layout; the English-only releases have 51864.</summary>
    public const int MultilingualVocabSize = 51_865;

    /// <summary>Number of timestamp tokens: 0.00 s to 30.00 s inclusive in 0.02 s steps.</summary>
    private const int TimestampCount = 1_501;

    // GPT-2 pre-tokenization regex (verbatim from HuggingFace whisper tokenizer.json
    // "pre_tokenizer" section). Splits on contractions, letter runs, digit runs,
    // non-whitespace symbol runs, and whitespace.
    private static readonly Regex _gpt2PreTokenRegex = new(
        @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
        RegexOptions.Compiled);

    /// <summary>GPT-2 byte encoder: maps each byte (0..255) to a printable unicode codepoint that BPE merges can operate on. Bytes 33-126, 161-172, 174-255 map to themselves; the remaining "non-printable" bytes are remapped above U+0100. This table is identical across GPT-2, RoBERTa, Whisper, and every model that uses byte-level BPE.</summary>
    private static readonly char[] _byteToUnicode = BuildByteToUnicode();

    /// <summary>Reverse of <see cref="_byteToUnicode"/>: unicode codepoint → byte.</summary>
    private static readonly Dictionary<char, byte> _unicodeToByte = BuildUnicodeToByte();

    /// <summary>Languages supported by Whisper, in the same order as the language token IDs (<see cref="FirstLanguageId"/> + index). Used by <see cref="LanguageToTokenId"/>.</summary>
    public static readonly IReadOnlyList<string> Languages = WhisperLanguageTable.Codes;

    private readonly BpeTokenizer _bpe;
    private readonly Dictionary<string, int> _specialTokens;
    private readonly Dictionary<int, string> _specialTokensReverse;
    private readonly int _vocabSize;
    private readonly int _firstSpecialId;
    private int _disposed;

    /// <summary>Total vocab size including special tokens (51865 multilingual, 51866 v3+, 51864 English-only). Use this to size embedding matrices.</summary>
    public int VocabSize => _vocabSize;

    /// <summary>Whether this checkpoint uses the multilingual layout, by OpenAI's rule (a vocabulary of at least 51865 entries). English-only checkpoints prompt with SOT alone and have no language or task slot.</summary>
    public bool IsMultilingual { get; }

    /// <summary>End-of-text token for THIS checkpoint (50257 multilingual, 50256 English-only). The decode loop must stop on this id; stopping on the multilingual constant leaves an English-only decode running to the token limit.</summary>
    public int EotId { get; }

    /// <summary>Start-of-transcript token for THIS checkpoint (50258 multilingual, 50257 English-only). Feeding the multilingual constant to an English-only checkpoint starts the prompt with <c>&lt;|en|&gt;</c> instead.</summary>
    public int SotId { get; }

    /// <summary>First language token (<c>&lt;|en|&gt;</c>) for THIS checkpoint; the language block follows SOT in every layout.</summary>
    public int FirstLanguageId { get; }

    /// <summary>Translate-task token for THIS checkpoint, read from its <c>added_tokens.json</c>.</summary>
    public int TranslateId { get; }

    /// <summary>Transcribe-task token for THIS checkpoint.</summary>
    public int TranscribeId { get; }

    /// <summary>No-speech token for THIS checkpoint. The HuggingFace files name it <c>&lt;|nocaptions|&gt;</c>; OpenAI's name <c>&lt;|nospeech|&gt;</c> is honored first.</summary>
    public int NoSpeechId { get; }

    /// <summary>No-timestamps token for THIS checkpoint. Feeding the &lt;=v2 constant to a v3 checkpoint lands on <c>&lt;|nospeech|&gt;</c> instead, and the decoder answers with an immediate EOT — an empty transcript.</summary>
    public int NoTimestampsId { get; }

    /// <summary>First timestamp token (0.00 s) for THIS checkpoint.</summary>
    public int FirstTimestampId { get; }

    /// <summary>Last timestamp token (30.00 s) for THIS checkpoint, inclusive.</summary>
    public int LastTimestampId => FirstTimestampId + TimestampCount - 1;

    /// <summary>Whether <paramref name="id"/> is one of this checkpoint's timestamp tokens.</summary>
    public bool IsTimestampId(int id) => id >= FirstTimestampId && id <= LastTimestampId;

    /// <summary>Seconds encoded by one of this checkpoint's timestamp tokens (0.02 s per step).</summary>
    public double SecondsForTimestamp(int id) => (id - FirstTimestampId) * 0.02;

    /// <summary>Creates a Whisper tokenizer from HuggingFace-format files. Pass the directory holding <c>vocab.json</c>, <c>merges.txt</c>, and (optionally) <c>added_tokens.json</c>. Every OpenAI and distil-whisper release ships <c>added_tokens.json</c>; a directory without one is read as the multilingual &lt;=v2 layout.</summary>
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

        _specialTokensReverse = new Dictionary<int, string>(_specialTokens.Count);
        int minSpecial = int.MaxValue;
        int maxSpecial = -1;
        foreach ((string tok, int id) in _specialTokens)
        {
            _specialTokensReverse[id] = tok;
            if (id < minSpecial) minSpecial = id;
            if (id > maxSpecial) maxSpecial = id;
        }

        // Every id comes from the checkpoint: v3 pushed the post-language specials up by one, and the English-only
        // layout sits one lower throughout. The constants only fill in for a name the files do not declare.
        (int vocabEntries, int vocabMaxId, int vocabEot) = ScanVocab(vocabPath);
        EotId = vocabEot >= 0 ? vocabEot : SpecialId("<|endoftext|>", EndOfTextId);
        SotId = SpecialId("<|startoftranscript|>", StartOfTranscriptId);
        FirstLanguageId = SpecialId("<|en|>", SotId + 1);
        TranslateId = SpecialId("<|translate|>", TranslateTokenId);
        TranscribeId = SpecialId("<|transcribe|>", TranscribeTokenId);
        NoSpeechId = SpecialId("<|nospeech|>", SpecialId("<|nocaptions|>", NoSpeechTokenId));
        NoTimestampsId = SpecialId("<|notimestamps|>", NoTimestampsTokenId);
        FirstTimestampId = SpecialId("<|0.00|>", NoTimestampsId + 1);
        _vocabSize = Math.Max(Math.Max(vocabEntries, vocabMaxId + 1), maxSpecial + 1);
        _firstSpecialId = Math.Min(EotId, minSpecial);
        IsMultilingual = _vocabSize >= MultilingualVocabSize;
    }

    /// <summary>The checkpoint's id for a special token, or <paramref name="fallback"/> when it declares none.</summary>
    private int SpecialId(string token, int fallback)
        => _specialTokens.TryGetValue(token, out int id) ? id : fallback;

    /// <summary>Tokenizes plain text into raw BPE token IDs (no special prefix / suffix). Use <see cref="BuildPromptIds"/> to assemble the full Whisper decoder prompt with SOT + language + task + notimestamps.</summary>
    public int[] EncodeText(string text)
    {
        ThrowIfDisposed();
        IReadOnlyList<int> ids = _bpe.EncodeToIds(text);
        int[] result = new int[ids.Count];
        for (int i = 0; i < ids.Count; i++) result[i] = ids[i];
        return result;
    }

    /// <summary>Builds the decoder prompt for this checkpoint. Multilingual layout: <c>[SOT, &lt;|lang|&gt;, transcribe/translate, &lt;|notimestamps|&gt;]</c>; pass <c>null</c> for <paramref name="language"/> to skip the language token. English-only layout: <c>[SOT, &lt;|notimestamps|&gt;]</c> — the prompt has no language or task slot (OpenAI's <c>sot_sequence</c> for <c>.en</c> models is SOT alone, HuggingFace's <c>forced_decoder_ids</c> is <c>[[1, notimestamps]]</c>), so <paramref name="language"/> and <paramref name="translate"/> are ignored.</summary>
    public int[] BuildPromptIds(string? language = "en", bool translate = false, bool withTimestamps = false)
    {
        ThrowIfDisposed();
        List<int> ids = new(4) { SotId };
        if (IsMultilingual)
        {
            if (language is not null) ids.Add(LanguageId(language));
            ids.Add(translate ? TranslateId : TranscribeId);
        }
        if (!withTimestamps) ids.Add(NoTimestampsId);
        return ids.ToArray();
    }

    /// <summary>Returns THIS checkpoint's token ID for a Whisper language code such as <c>"en"</c>, <c>"zh"</c>, <c>"yue"</c>. Throws if the code is unknown.</summary>
    public int LanguageId(string code) => FirstLanguageId + LanguageIndex(code);

    /// <summary>Returns the multilingual-layout token ID for a Whisper language code such as <c>"en"</c>, <c>"zh"</c>, <c>"yue"</c>. Throws if the code is unknown. Prefer the instance <see cref="LanguageId"/>, which follows the checkpoint's layout.</summary>
    public static int LanguageToTokenId(string code) => LanguageStartId + LanguageIndex(code);

    /// <summary>Index of a language code in Whisper's table, throwing for an unknown code.</summary>
    private static int LanguageIndex(string code)
    {
        int idx = WhisperLanguageTable.IndexOf(NormalizeLanguageCode(code));
        if (idx < 0) throw new ArgumentException($"Unknown Whisper language code '{code}'.", nameof(code));
        return idx;
    }

    /// <summary>Normalizes a caller-supplied language code to the ISO-639-1 form Whisper's table uses: lowercases and strips any BCP-47 / locale region subtag, so <c>"en-US"</c>, <c>"en_US"</c>, and <c>"EN"</c> all map to <c>"en"</c>. Multi-letter codes without a separator (e.g. <c>"yue"</c>) pass through unchanged. Without this, a UI/API defaulting to a locale like <c>"en-US"</c> throws "Unknown Whisper language code".</summary>
    private static string NormalizeLanguageCode(string code)
    {
        if (string.IsNullOrWhiteSpace(code)) return code;
        string c = code.Trim().ToLowerInvariant();
        int sep = c.IndexOfAny(['-', '_']);
        return sep > 0 ? c[..sep] : c;
    }

    /// <summary>Returns the language code for a multilingual-layout language token ID (50259..50357 or 50358 for Cantonese on v3). Returns <c>null</c> for non-language tokens.</summary>
    public static string? TokenIdToLanguage(int id)
    {
        int idx = id - LanguageStartId;
        if (idx < 0 || idx >= WhisperLanguageTable.Codes.Count) return null;
        return WhisperLanguageTable.Codes[idx];
    }

    /// <summary>Decodes token IDs back to text. Drops special tokens (everything from this checkpoint's EOT upward, which covers the prompt tokens, language tags and timestamps) unless <paramref name="includeSpecial"/> is true.</summary>
    public string Decode(ReadOnlySpan<int> tokenIds, bool includeSpecial = false)
    {
        ThrowIfDisposed();
        List<int> bpeIds = new(tokenIds.Length);
        StringBuilder special = new();
        foreach (int id in tokenIds)
        {
            if (id >= _firstSpecialId)
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
        if (tokenId >= _firstSpecialId)
            return _specialTokensReverse.TryGetValue(tokenId, out string? tok) ? tok : string.Empty;
        string raw = _bpe.Decode([tokenId]) ?? string.Empty;
        return ByteLevelDecode(raw);
    }

    /// <summary>Reverses the GPT-2 byte_encoder mapping: each char → byte → UTF-8 string. Multi-byte UTF-8 characters (CJK, emoji, accented Latin) span multiple BPE tokens; the raw concatenation gives us a complete byte sequence that UTF-8-decodes correctly.</summary>
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

    /// <summary>Returns true if the token ID is a timestamp token in the multilingual &lt;=v2 layout. Prefer the instance <see cref="IsTimestampId"/>.</summary>
    public static bool IsTimestamp(int id) => id >= TimestampStartId && id < TimestampStartId + TimestampCount;

    /// <summary>Converts a multilingual &lt;=v2 timestamp token ID to seconds (0..30 s in 0.02 s steps). Prefer the instance <see cref="SecondsForTimestamp"/>.</summary>
    public static double TimestampToSeconds(int id) => (id - TimestampStartId) * 0.02;

    private static Dictionary<string, int> LoadSpecialTokens(string addedTokensPath)
    {
        Dictionary<string, int> special = new(StringComparer.Ordinal);
        if (!File.Exists(addedTokensPath))
        {
            // No declaration at all: assume the multilingual <=v2 layout in full so the pre-tokenizer recognizes
            // the same specials it would read from a stock added_tokens.json.
            special["<|endoftext|>"] = EndOfTextId;
            special["<|startoftranscript|>"] = StartOfTranscriptId;
            for (int i = 0; i < 99; i++) special[$"<|{Languages[i]}|>"] = LanguageStartId + i;
            special["<|translate|>"] = TranslateTokenId;
            special["<|transcribe|>"] = TranscribeTokenId;
            special["<|startoflm|>"] = TranscribeTokenId + 1;
            special["<|startofprev|>"] = TranscribeTokenId + 2;
            special["<|nospeech|>"] = NoSpeechTokenId;
            special["<|notimestamps|>"] = NoTimestampsTokenId;
            for (int i = 0; i < TimestampCount; i++)
                special["<|" + (i * 0.02).ToString("0.00", CultureInfo.InvariantCulture) + "|>"] = TimestampStartId + i;
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

    /// <summary>Reads <c>vocab.json</c> (<c>{ "token": id, ... }</c>) once for its entry count, highest id and the id of <c>&lt;|endoftext|&gt;</c> (-1 when absent). The specials from <c>added_tokens.json</c> extend the vocabulary further; the caller sizes it to the highest id of either file plus one.</summary>
    private static (int Entries, int MaxId, int EotId) ScanVocab(string vocabPath)
    {
        using FileStream fs = File.OpenRead(vocabPath);
        using JsonDocument doc = JsonDocument.Parse(fs);
        int max = 0;
        int count = 0;
        int eot = -1;
        foreach (JsonProperty prop in doc.RootElement.EnumerateObject())
        {
            count++;
            int id = prop.Value.GetInt32();
            if (id > max) max = id;
            if (prop.NameEquals("<|endoftext|>")) eot = id;
        }
        return (count, max, eot);
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
