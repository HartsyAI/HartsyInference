namespace HartsyInference.Audio.Models.NeuTts;

/// <summary>Builds the exact NeuTTS prompt prefix, verified against upstream <c>neutts/neutts.py</c>
/// <c>_apply_chat_template</c>: <c>enc("user: Convert the text to speech:") + [TEXT_PROMPT_START] +
/// enc(phonemes) + [TEXT_PROMPT_END] + enc("\nassistant:")</c>; <see cref="NeuTtsPipeline.Synthesize"/> then
/// appends <c>[SPEECH_GENERATION_START] + refCodes</c>. The text MUST be espeak IPA phonemes (upstream
/// phonemizes with espeak-ng <c>with_stress=True, preserve_punctuation=True</c>; for cloning it is
/// <c>phones(refText) + " " + phones(text)</c>). Takes a byte-level-exact plain-BPE encode delegate (e.g.
/// <c>Qwen2Tokenizer.EncodeRawByteLevel</c>) so the Audio package carries no tokenizer dependency; the IPA
/// added-token matching (see <see cref="NeuTtsPhonemeTokens"/>) is handled here.</summary>
public static class NeuTtsPromptBuilder
{
    /// <summary>Chat prefix before the text-prompt specials (upstream literal; no trailing space).</summary>
    public const string ChatPrefix = "user: Convert the text to speech:";
    /// <summary>Chat suffix after <c>TEXT_PROMPT_END</c>, before the speech-gen marker.</summary>
    public const string ChatSuffix = "\nassistant:";

    // Per-table longest-match lookup, cached by (table identity, base id).
    private static readonly Dictionary<(object, int), (Dictionary<string, int> Map, int MaxLen)> LookupCache = [];

    /// <summary>Tokenizes the full prompt prefix for <paramref name="phonemes"/> (already-phonemized text).</summary>
    public static int[] BuildPromptPrefix(NeuTtsConfig cfg, Func<string, IReadOnlyList<int>> encodeText, string phonemes)
    {
        List<int> ids = new(phonemes.Length + 24);
        ids.AddRange(encodeText(ChatPrefix));
        ids.Add(cfg.TextPromptStart);
        EncodePhonemes(cfg, encodeText, phonemes, ids);
        ids.Add(cfg.TextPromptEnd);
        ids.AddRange(encodeText(ChatSuffix));
        return [.. ids];
    }

    /// <summary>Encodes phonemized text the way the HF tokenizer does: greedy added-token matching over the
    /// checkpoint's IPA tokens, with the remaining spans byte-level BPE'd independently (so a segment's
    /// leading space stays attached to its word, as upstream).</summary>
    public static void EncodePhonemes(NeuTtsConfig cfg, Func<string, IReadOnlyList<int>> encodeText, string text, List<int> dst)
    {
        (Dictionary<string, int> map, int maxLen) = GetLookup(cfg);
        int segStart = 0;
        for (int i = 0; i < text.Length;)
        {
            int id = -1, matched = 0;
            for (int len = Math.Min(maxLen, text.Length - i); len >= 1; len--)
            {
                if (map.TryGetValue(text.Substring(i, len), out int hit)) { id = hit; matched = len; break; }
            }
            if (id < 0) { i++; continue; }
            if (i > segStart) dst.AddRange(encodeText(text[segStart..i]));
            dst.Add(id);
            i += matched;
            segStart = i;
        }
        if (segStart < text.Length) dst.AddRange(encodeText(text[segStart..]));
    }

    private static (Dictionary<string, int>, int) GetLookup(NeuTtsConfig cfg)
    {
        IReadOnlyList<string> table = cfg.PhonemeTokens;
        (object, int) key = (table, cfg.PhonemeTokenBase);
        lock (LookupCache)
        {
            if (LookupCache.TryGetValue(key, out (Dictionary<string, int> Map, int MaxLen) cached))
                return cached;
            Dictionary<string, int> map = new(table.Count, StringComparer.Ordinal);
            int maxLen = 1;
            for (int i = 0; i < table.Count; i++)
            {
                map[table[i]] = cfg.PhonemeTokenBase + i;
                if (table[i].Length > maxLen) maxLen = table[i].Length;
            }
            LookupCache[key] = (map, maxLen);
            return (map, maxLen);
        }
    }
}
