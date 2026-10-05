namespace HartsyInference.ModelAssets.Tokenizers;

/// <summary>IndexTTS-2.5's tiktoken-format text tokenizer (<c>multilingual_zh_ja_yue_char_del.tiktoken</c>,
/// 58,836 mergeable ranks) wrapped the same way the real <c>indextts.utils.tokenizer.get_tokenizer(multilingual=
/// True)</c> builds it: a byte-level BPE over <see cref="PatStr"/>-split pre-tokens, plus a fixed list of special
/// tokens appended above the rank file (language tags, ASR/audio-event/emotion markers inherited from the
/// Whisper-derived tokenizer base, and 1,501 Whisper-style timestamp tokens) assigned ids in the exact order the
/// real <c>get_encoding()</c> registers them — this order is baked into the trained <c>text_embedding</c>/
/// <c>text_head</c> rows (58,836 ranks + 1,673 specials = 60,509, matching <c>config.yaml</c>'s
/// <c>gpt.number_text_tokens</c> exactly, confirmed against the real checkpoint).
/// <para>No BOS/EOS: like <see cref="IndexTtsTokenizer"/>, IndexTTS inserts explicit <c>start_text_token</c>
/// (0) / <c>stop_text_token</c> (1) sentinels itself, outside this vocabulary.</para></summary>
public sealed class IndexTts2TiktokenTokenizer
{
    /// <summary>The real upstream pre-tokenizer regex (<c>indextts/utils/tokenizer.py</c>'s <c>get_encoding</c>)
    /// — GPT-4-style: contractions, then runs of letters / digits / other-non-space, then whitespace.</summary>
    public const string PatStr = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

    private readonly GgufTokenizer _tokenizer;

    public IndexTts2TiktokenTokenizer(string tiktokenPath)
    {
        byte[] json = TiktokenConverter.ToHuggingFaceJson(tiktokenPath, PatStr, normalizer: null, addedTokens: BuildSpecials(58_836));
        using MemoryStream stream = new(json);
        _tokenizer = HfTokenizerJson.LoadByteLevelBpe(stream);
    }

    /// <summary>Encodes text to ids, recognizing the literal <c>&lt;|...|&gt;</c> special-token substrings the
    /// real tokenizer's <c>allowed_special="all"</c> does (e.g. the pronunciation-annotation wrapper IndexTTS-2.5
    /// injects around non-standard pronunciations).</summary>
    public int[] Encode(string text) => _tokenizer.Encode(text, addSpecial: true);

    public string Decode(ReadOnlySpan<int> tokenIds) => _tokenizer.Decode(tokenIds.ToArray());

    /// <summary>Ports <c>indextts/utils/tokenizer.py</c>'s <c>get_encoding(num_languages=99)</c> special-token
    /// construction verbatim, in the same order, starting at <paramref name="baseVocabSize"/> (the rank file's own
    /// entry count). Changing this order or set would desync every id from the trained embedding rows.</summary>
    internal static List<(string Token, int Id)> BuildSpecials(int baseVocabSize)
    {
        List<string> specials =
        [
            "<|endoftext|>",
            "<|startoftranscript|>",
        ];
        // list(LANGUAGES.keys())[:99] — the real call never overrides num_languages, so only the first 99 of the
        // 106-entry LANGUAGES dict become special tokens; "yue"/"minnan"/"wuyu"/"dialect"/"zh/en"/"en/zh"/"common"
        // (the last 7) are deliberately excluded (Cantonese support instead comes from the base char vocab).
        specials.AddRange(LanguageCodes.Take(99).Select(code => $"<|{code}|>"));
        specials.AddRange(AudioEvents.Select(e => $"<|{e}|>"));
        specials.AddRange(Emotions.Select(e => $"<|{e}|>"));
        specials.AddRange(["<|translate|>", "<|transcribe|>", "<|startoflm|>", "<|startofprev|>", "<|nospeech|>", "<|notimestamps|>"]);
        for (int i = 1; i <= 30; i++) specials.Add($"<|SPECIAL_TOKEN_{i}|>");
        specials.AddRange(TtsVocalTokens.Select(t => $"<|{t}|>"));
        for (int i = 0; i < 1501; i++) specials.Add($"<|{i * 0.02:F2}|>");

        List<(string, int)> result = new(specials.Count);
        for (int i = 0; i < specials.Count; i++) result.Add((specials[i], baseVocabSize + i));
        return result;
    }

    /// <summary>Real <c>lang_to_token</c>: the GPT's <c>lang_embedding</c> row index for a language code —
    /// literally that code's position in <see cref="LanguageCodes"/> (declaration order in the real
    /// <c>LANGUAGES</c> dict), falling back to <c>"common"</c>'s index (the last entry) for an unrecognized
    /// code, matching the real source's own fallback exactly.</summary>
    public static int LangToToken(string lang)
    {
        int index = Array.IndexOf(LanguageCodes, lang.ToLowerInvariant());
        return index >= 0 ? index : Array.IndexOf(LanguageCodes, "common");
    }

    // indextts/utils/tokenizer.py's LANGUAGES dict, keys in declaration order (values are display names, unused
    // for tokenization — only the codes become special-token literals).
    private static readonly string[] LanguageCodes =
    [
        "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi",
        "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la",
        "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy",
        "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be",
        "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
        "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su", "yue", "minnan", "wuyu", "dialect", "zh/en", "en/zh",
        "common",
    ];

    private static readonly string[] AudioEvents =
        ["ASR", "AED", "SER", "Speech", "/Speech", "BGM", "/BGM", "Laughter", "/Laughter", "Applause", "/Applause"];

    private static readonly string[] Emotions = ["HAPPY", "SAD", "ANGRY", "NEUTRAL"];

    private static readonly string[] TtsVocalTokens =
    [
        "TTS/B", "TTS/O", "TTS/Q", "TTS/A", "TTS/CO", "TTS/CL", "TTS/H",
        "TTS/SP01", "TTS/SP02", "TTS/SP03", "TTS/SP04", "TTS/SP05", "TTS/SP06", "TTS/SP07", "TTS/SP08", "TTS/SP09",
        "TTS/SP10", "TTS/SP11", "TTS/SP12", "TTS/SP13",
    ];
}
