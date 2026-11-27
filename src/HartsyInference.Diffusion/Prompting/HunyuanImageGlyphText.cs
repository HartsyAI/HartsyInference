using System.Text;
using System.Text.RegularExpressions;

namespace HartsyInference.Diffusion.Prompting;

/// <summary>The text Hunyuan-Image 2.1 renders as glyphs: every quoted span in the prompt, collected as ComfyUI's
/// <c>HunyuanImageTokenizer</c> does (ASCII <c>"…"</c> spans first, then <c>‘…’</c>, then <c>“…”</c>, each in order of
/// appearance, non-greedy) and joined as <c>Text "&lt;span&gt;". </c> per span. That string is what the ByT5 encoder
/// reads; a prompt without quotes has no glyph text and the second text stream stays empty.</summary>
public static partial class HunyuanImageGlyphText
{
    /// <summary>The ByT5 input for <paramref name="prompt"/>, or null when it quotes nothing.</summary>
    public static string? Extract(string? prompt)
    {
        if (string.IsNullOrEmpty(prompt))
            return null;
        List<string> spans = [];
        foreach (Regex pattern in (Regex[])[AsciiDouble(), ChineseSingle(), ChineseDouble()])
        {
            foreach (Match match in pattern.Matches(prompt))
                spans.Add(match.Groups[1].Value);
        }
        if (spans.Count == 0)
            return null;
        StringBuilder text = new StringBuilder();
        foreach (string span in spans)
            text.Append("Text \"").Append(span).Append("\". ");
        return text.ToString();
    }

    // Python's re: '.' stops at a newline, '*?' is lazy; RegexOptions.None matches both.
    [GeneratedRegex("\"(.*?)\"")]
    private static partial Regex AsciiDouble();

    [GeneratedRegex("‘(.*?)’")]
    private static partial Regex ChineseSingle();

    [GeneratedRegex("“(.*?)”")]
    private static partial Regex ChineseDouble();
}
