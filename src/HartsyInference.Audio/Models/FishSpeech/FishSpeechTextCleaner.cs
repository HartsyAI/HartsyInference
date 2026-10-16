using System.Text.RegularExpressions;

namespace HartsyInference.Audio.Models.FishSpeech;

/// <summary>Fish-Speech 1.4's prompt text cleaner: upstream <c>clean_text</c> (<c>fish_speech/text/clean.py</c>, v1.4.3). It trims
/// the text, maps CJK and typographic punctuation to ASCII, drops emoji and collapses runs of periods or commas. The prompt is
/// built from its output, so the mapping is part of the prompt contract.</summary>
public static class FishSpeechTextCleaner
{
    private static readonly Dictionary<string, string> SymbolMap = new()
    {
        ["\n"] = "", ["…"] = ".", ["“"] = "'", ["”"] = "'", ["‘"] = "'", ["’"] = "'",
        ["【"] = "", ["】"] = "", ["["] = "", ["]"] = "", ["（"] = "", ["）"] = "", ["("] = "", [")"] = "",
        ["・"] = "", ["·"] = "", ["「"] = "'", ["」"] = "'", ["《"] = "'", ["》"] = "'", ["—"] = "", ["～"] = "", ["~"] = "",
        ["："] = ",", ["；"] = ",", [";"] = ",", [":"] = ",",
    };

    private static readonly Regex SymbolRegex = new(string.Join("|", SymbolMap.Keys.Select(Regex.Escape)), RegexOptions.CultureInvariant);

    // Surrogate-pair ranges for the emoji blocks upstream strips (U+1F600–1F64F, 1F300–1F5FF, 1F680–1F6FF, 1F1E0–1F1FF).
    private static readonly Regex EmojiRegex = new(
        @"(?:\uD83D[\uDE00-\uDE4F]|\uD83C[\uDF00-\uDFFF]|\uD83D[\uDC00-\uDDFF]|\uD83D[\uDE80-\uDEFF]|\uD83C[\uDDE0-\uDDFF])+",
        RegexOptions.CultureInvariant);

    private static readonly Regex RepeatedPunctuation = new(@"[.,]{2,}", RegexOptions.CultureInvariant);

    /// <summary>Cleans <paramref name="text"/> the way upstream's <c>clean_text</c> does, for the 1.4 prompt.</summary>
    public static string Clean(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        string result = SymbolRegex.Replace(text.Trim(), m => SymbolMap[m.Value]);
        result = EmojiRegex.Replace(result, "");
        return RepeatedPunctuation.Replace(result, m => m.Value[..1]);
    }
}
