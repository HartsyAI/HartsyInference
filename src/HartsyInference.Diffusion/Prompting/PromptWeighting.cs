using System.Collections.Generic;
using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace HartsyInference.Diffusion.Prompting;

/// <summary>Parses the ComfyUI emphasis grammar into weighted text spans. <c>(x)</c> multiplies weight by 1.1, <c>(x:1.3)</c> sets it to 1.3, nesting compounds, and a backslash before a parenthesis escapes it to a literal. Square brackets are left untouched (they are scheduling/alternation, handled by <see cref="PromptScheduling"/>). This is a faithful port of ComfyUI's <c>parse_parentheses</c>/<c>token_weights</c> so output matches its conditioning.</summary>
public static partial class PromptWeighting
{
    /// <summary>Parses emphasis into weighted spans. Empty-text spans are dropped; whitespace is preserved so adjacent words do not merge.</summary>
    public static IReadOnlyList<WeightedSpan> Parse(string prompt)
    {
        if (prompt is null)
        {
            throw new ArgumentNullException(nameof(prompt));
        }
        List<WeightedSpan> collected = new List<WeightedSpan>();
        TokenWeights(prompt, 1.0f, collected);
        List<WeightedSpan> result = new List<WeightedSpan>(collected.Count);
        foreach (WeightedSpan span in collected)
        {
            string text = Unescape(span.Text);
            if (text.Length == 0)
            {
                continue;
            }
            result.Add(new WeightedSpan(text, span.Weight));
        }
        return result;
    }

    /// <summary>Splits the prompt on <c>&lt;break&gt;</c> (case-insensitive) into independently-encoded chunks, then parses each into weighted spans. Empty chunks are preserved so positive/negative prompts can be padded to equal chunk counts by the caller.</summary>
    public static IReadOnlyList<IReadOnlyList<WeightedSpan>> ParseChunks(string prompt)
    {
        if (prompt is null)
        {
            throw new ArgumentNullException(nameof(prompt));
        }
        string[] rawChunks = BreakRegex().Split(prompt);
        List<IReadOnlyList<WeightedSpan>> chunks = new List<IReadOnlyList<WeightedSpan>>(rawChunks.Length);
        foreach (string raw in rawChunks)
        {
            chunks.Add(Parse(raw));
        }
        return chunks;
    }

    private static void TokenWeights(string text, float currentWeight, List<WeightedSpan> output)
    {
        List<string> parts = ParseParentheses(text);
        foreach (string part in parts)
        {
            if (part.Length >= 2 && part[0] == '(' && part[^1] == ')')
            {
                string inner = part.Substring(1, part.Length - 2);
                float weight = currentWeight * 1.1f;
                int colon = inner.LastIndexOf(':');
                if (colon > 0
                    && float.TryParse(inner.AsSpan(colon + 1), NumberStyles.Float, CultureInfo.InvariantCulture, out float parsed))
                {
                    weight = currentWeight * parsed;
                    inner = inner.Substring(0, colon);
                }
                TokenWeights(inner, weight, output);
            }
            else
            {
                output.Add(new WeightedSpan(part, currentWeight));
            }
        }
    }

    private static List<string> ParseParentheses(string text)
    {
        List<string> result = new List<string>();
        StringBuilder current = new StringBuilder();
        int nesting = 0;
        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            bool escaped = i > 0 && text[i - 1] == '\\';
            if (c == '(' && !escaped)
            {
                if (nesting == 0 && current.Length > 0)
                {
                    result.Add(current.ToString());
                    current.Clear();
                }
                current.Append(c);
                nesting++;
            }
            else if (c == ')' && !escaped && nesting > 0)
            {
                nesting--;
                current.Append(c);
                if (nesting == 0)
                {
                    result.Add(current.ToString());
                    current.Clear();
                }
            }
            else
            {
                current.Append(c);
            }
        }
        if (current.Length > 0)
        {
            result.Add(current.ToString());
        }
        return result;
    }

    private static string Unescape(string text)
    {
        if (text.IndexOf('\\') < 0)
        {
            return text;
        }
        StringBuilder sb = new StringBuilder(text.Length);
        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            if (c == '\\' && i + 1 < text.Length && (text[i + 1] == '(' || text[i + 1] == ')'))
            {
                sb.Append(text[i + 1]);
                i++;
            }
            else
            {
                sb.Append(c);
            }
        }
        return sb.ToString();
    }

    [GeneratedRegex("<break>", RegexOptions.IgnoreCase)]
    private static partial Regex BreakRegex();
}
