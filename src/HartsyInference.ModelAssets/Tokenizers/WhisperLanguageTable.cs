namespace HartsyInference.ModelAssets.Tokenizers;

/// <summary>The 99-language table that backs Whisper's language token IDs (50259..50357 for &lt;= v2; large-v3 adds Cantonese at 50358 for 100 total). Order matches OpenAI's <c>whisper/tokenizer.py LANGUAGES</c> dict.</summary>
internal static class WhisperLanguageTable
{
    /// <summary>ISO-639-1 language codes in token-id order. Index 0 → token 50259 (English), index 1 → 50260 (Chinese), and so on.</summary>
    public static readonly IReadOnlyList<string> Codes =
    [
        "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
        "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
        "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
        "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
        "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
        "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
        "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
        "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
        "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
        "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su",
        "yue",  // large-v3 only (token 50358)
    ];

    /// <summary>Looks up a language code; returns -1 if unknown.</summary>
    public static int IndexOf(string code)
    {
        for (int i = 0; i < Codes.Count; i++)
            if (Codes[i] == code) return i;
        return -1;
    }
}
