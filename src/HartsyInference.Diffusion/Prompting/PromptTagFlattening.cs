using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;

namespace HartsyInference.Diffusion.Prompting;

/// <summary>Recursively resolves <c>&lt;weight[N]:...&gt;</c>, <c>&lt;alt/alternate:...&gt;</c>, <c>&lt;fromto[N]:...&gt;</c>
/// tags that SwarmUI's updated (2026-09-01) prompt parser now emits as literal text in the resolved prompt
/// string, in place of the Comfy-native <c>(word:1.5)</c>/<c>[a|b]</c>/<c>[a:b:N]</c> syntax it used to emit for
/// these same constructs. <c>&lt;weight[N]:inner&gt;</c> converts to the legacy <c>(inner:N)</c> parens grammar
/// <see cref="PromptWeighting"/> already implements — this mirrors what SwarmUI's own reference implementation
/// does internally for CLIP models (<c>join_text</c> in its <c>SwarmText.py</c>). <c>&lt;alt&gt;</c>/<c>&lt;alternate&gt;</c>/
/// <c>&lt;fromto[N]&gt;</c> flatten to their first ("step 0") value, matching what a step-unaware consumer would see
/// for the very first denoise step — see <see cref="PromptTagScheduling"/> for real per-step resolution.
/// Every other tag (<c>&lt;region:&gt;</c>, <c>&lt;break&gt;</c>, <c>&lt;embed:&gt;</c>, <c>&lt;lora:&gt;</c>, <c>&lt;refcrop:&gt;</c>, etc.)
/// passes through byte-for-byte untouched — this is a narrow, targeted fix for exactly the three tag kinds
/// SwarmUI's update changed, not a general <c>&lt;...&gt;</c> strip, so it is safe to run ahead of every other
/// prompt-tag parser in the engine.</summary>
public static class PromptTagFlattening
{
    /// <summary>Flattens the three tag kinds above. <paramref name="flattenScheduling"/> controls only
    /// <c>alt</c>/<c>alternate</c>/<c>fromto</c>: pass <c>false</c> for a caller that wants those tags preserved
    /// to build a real per-step schedule instead (see <see cref="PromptTagScheduling"/>).
    /// <paramref name="weightsAsParens"/> controls <c>weight</c>: <c>true</c> emits the <c>(text:N)</c> grammar
    /// for the architectures whose tokenizer applies per-token weights (CLIP, via
    /// <see cref="PromptWeighting"/>); <c>false</c> emits the inner text alone, which is what SwarmUI's own
    /// reference does for every encoder it cannot weight (<c>join_text(leaves, False)</c> in <c>SwarmText.py</c>
    /// — it never round-trips a weight back into prompt text). Leaving parens in the text for a
    /// Qwen/T5/Gemma-conditioned DiT would feed the literal digits to the encoder as prose.</summary>
    public static string Flatten(string? prompt, bool flattenScheduling = true, bool weightsAsParens = true)
    {
        if (string.IsNullOrEmpty(prompt) || prompt.IndexOf('<') < 0)
        {
            return prompt ?? "";
        }
        return FlattenInner(prompt, flattenScheduling, weightsAsParens);
    }

    private static string FlattenInner(string text, bool flattenScheduling, bool weightsAsParens)
    {
        StringBuilder result = new StringBuilder(text.Length);
        int i = 0;
        while (i < text.Length)
        {
            char c = text[i];
            if (c != '<')
            {
                result.Append(c);
                i++;
                continue;
            }
            int close = FindMatchingTag(text, i);
            if (close == -1)
            {
                result.Append(c);
                i++;
                continue;
            }
            string content = text[(i + 1)..close];
            (string prefix, string? predata, string data) = SplitTag(content);
            if (prefix == "weight" && predata is not null
                && double.TryParse(predata, NumberStyles.Float, CultureInfo.InvariantCulture, out double weight))
            {
                // Escape literal parens in the RAW data before flattening, then flatten — so a nested
                // <weight[...]:...> tag (which only exists after flattening) produces fresh, unescaped
                // (text:innerWeight) parens that PromptWeighting.Parse's own nested-paren compounding
                // handles naturally (e.g. <weight[1.5]:<weight[1.2]:cat>>> -> ((cat:1.2):1.5), which
                // PromptWeighting.Parse already resolves to weight 1.2*1.5 via its LastIndexOf(':') strip),
                // while any parens the user actually typed as prose stay escaped and literal.
                if (!weightsAsParens)
                {
                    // No token-weight machinery downstream, so the markup is dropped rather than handed to the
                    // encoder as prose. Data is NOT paren-escaped here: nothing will re-parse it.
                    result.Append(FlattenInner(data, flattenScheduling, weightsAsParens));
                    i = close + 1;
                    continue;
                }
                string inner = FlattenInner(EscapeParens(data), flattenScheduling, weightsAsParens);
                result.Append('(').Append(inner).Append(':')
                    .Append(weight.ToString("0.######", CultureInfo.InvariantCulture)).Append(')');
                i = close + 1;
                continue;
            }
            if (flattenScheduling && prefix is "alt" or "alternate")
            {
                result.Append(FirstFlattened(data, flattenScheduling, weightsAsParens));
                i = close + 1;
                continue;
            }
            if (flattenScheduling && prefix == "fromto" && TryParseWhen(predata, out _))
            {
                result.Append(FirstFlattened(data, flattenScheduling, weightsAsParens));
                i = close + 1;
                continue;
            }
            // Unrecognized tag (or a scheduling tag held back by flattenScheduling=false). Keep the tag itself
            // — prefix, predata brackets and all — byte-for-byte, but RECURSE INTO ITS DATA, mirroring what
            // SwarmUI's own LegacyPromptParser does for subtags. Load-bearing: a weight tag nested inside a
            // held-back scheduling tag (`<alternate:<weight[1.5]:a>, b>`, a shape SwarmUI explicitly emits —
            // `(layers of [a|b] features:1.5)` becomes `<weight[1.5]:layers of <alternate:a,b> features>`)
            // would otherwise survive as a raw tag and reach the tokenizer as literal garbage once
            // PromptTagScheduling picked that entry. Splicing on the top-level colon (rather than re-emitting
            // from SplitTag's parsed pieces) keeps `<param[cfgscale]:5>`-style predata intact.
            int dataColon = IndexOfNoncontained(content, ':');
            if (dataColon == -1)
            {
                result.Append(text, i, close - i + 1);
            }
            else
            {
                result.Append('<').Append(content, 0, dataColon + 1)
                    .Append(FlattenInner(content[(dataColon + 1)..], flattenScheduling, weightsAsParens)).Append('>');
            }
            i = close + 1;
        }
        return result.ToString();
    }

    /// <summary>Splits <paramref name="data"/> on SwarmUI's smart separator and flattens only the first entry
    /// — correct for step 0 of both <c>alternate</c> (cycles by <c>step % count</c>, so step 0 is entry 0) and
    /// <c>fromto</c> (switches at <c>when</c>, which is virtually always &gt; 0, so step 0 is always the "from"
    /// value, entry 0).</summary>
    private static string FirstFlattened(string data, bool flattenScheduling, bool weightsAsParens)
    {
        string[] parts = SplitSmart(data);
        return parts.Length > 0 ? FlattenInner(parts[0], flattenScheduling, weightsAsParens) : "";
    }

    /// <summary>Parses a <c>&lt;fromto[when]:...&gt;</c> threshold. A non-numeric <paramref name="predata"/> means the
    /// tag is not scheduling at all and stays literal prose — the same rule the weight tag follows, and what
    /// SwarmUI's reference <c>SwarmText.py</c> does (<c>except: return ParsedText(text=...)</c>). Internal (not
    /// private) so <see cref="PromptTagScheduling"/> applies the identical test.</summary>
    internal static bool TryParseWhen(string? predata, out float when)
    {
        when = 0f;
        return predata is not null
            && float.TryParse(predata.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out when);
    }

    /// <summary>Finds the index of the <c>&gt;</c> that closes the <c>&lt;</c> at <paramref name="openIndex"/>,
    /// honoring nested <c>&lt;...&gt;</c> depth. Returns -1 if unterminated. Internal (not private) so
    /// <see cref="PromptTagScheduling"/> shares this exact tag-boundary scan rather than re-implementing it.</summary>
    internal static int FindMatchingTag(string text, int openIndex)
    {
        int depth = 0;
        for (int i = openIndex; i < text.Length; i++)
        {
            if (text[i] == '<')
            {
                depth++;
            }
            else if (text[i] == '>')
            {
                depth--;
                if (depth == 0)
                {
                    return i;
                }
            }
        }
        return -1;
    }

    /// <summary>The index of the first top-level (not inside a nested <c>&lt;...&gt;</c>) occurrence of
    /// <paramref name="target"/>, or -1. Ported from SwarmUI's <c>T2IPromptHandling.IndexOfNoncontained</c>.</summary>
    private static int IndexOfNoncontained(string val, char target)
    {
        int depth = 0;
        for (int i = 0; i < val.Length; i++)
        {
            char c = val[i];
            if (c == '<')
            {
                depth++;
            }
            else if (c == '>')
            {
                depth--;
            }
            else if (c == target && depth == 0)
            {
                return i;
            }
        }
        return -1;
    }

    /// <summary>Splits a tag's inner content (everything between <c>&lt;</c> and <c>&gt;</c>) into its lowercase
    /// prefix, optional bracketed predata, and post-colon data. Ported from SwarmUI's <c>split_tag</c>
    /// (<c>SwarmText.py</c>) / the equivalent C# tag-splitting in <c>T2IPromptHandling.cs</c>. Internal (not
    /// private) so <see cref="PromptTagScheduling"/> shares it.</summary>
    internal static (string Prefix, string? Predata, string Data) SplitTag(string tag)
    {
        string prefix = tag;
        string data = "";
        int colonIndex = IndexOfNoncontained(tag, ':');
        if (colonIndex != -1)
        {
            prefix = tag[..colonIndex];
            data = tag[(colonIndex + 1)..];
        }
        string? predata = null;
        int bracket = IndexOfNoncontained(prefix, '[');
        if (prefix.EndsWith(']') && bracket != -1)
        {
            predata = prefix[(bracket + 1)..^1];
            prefix = prefix[..bracket];
        }
        return (prefix.ToLowerInvariant(), predata, data);
    }

    /// <summary>Splits on whichever of <c>||</c>, <c>|</c>, or <c>,</c> is most unique in <paramref name="input"/>
    /// (in that preference order), never splitting inside a nested <c>&lt;...&gt;</c>. Exact 1:1 port of
    /// SwarmUI's <c>T2IPromptHandling.SplitSmart</c> — the algorithm that produced whatever separator is
    /// actually present in an <c>&lt;alternate:&gt;</c>/<c>&lt;fromto[N]:&gt;</c> tag's data, so splitting must match it
    /// exactly or entries with commas in their own text would be split incorrectly. Internal (not private)
    /// so <see cref="PromptTagScheduling"/> shares it.</summary>
    internal static string[] SplitSmart(string input)
    {
        string separator = ",";
        int depth = 0;
        for (int i = 0; i < input.Length; i++)
        {
            if (input[i] == '<')
            {
                depth++;
            }
            else if (input[i] == '>')
            {
                depth--;
            }
            else if (depth == 0 && input[i] == '|' && i > 0 && input[i - 1] == '|')
            {
                separator = "||";
                break;
            }
            else if (depth == 0 && input[i] == '|')
            {
                separator = "|";
            }
        }
        List<string> output = new List<string>();
        depth = 0;
        int start = 0;
        for (int i = 0; i < input.Length; i++)
        {
            if (input[i] == '<')
            {
                depth++;
            }
            else if (input[i] == '>')
            {
                depth--;
            }
            else if (depth == 0 && i + separator.Length - 1 < input.Length && input[i..(i + separator.Length)] == separator)
            {
                output.Add(input[start..i]);
                start = i + separator.Length;
                i += separator.Length - 1;
            }
        }
        if (start <= input.Length)
        {
            output.Add(input[start..]);
        }
        return output.Select(v => v.Trim()).ToArray();
    }

    /// <summary>Backslash-escapes literal <c>(</c>/<c>)</c> so a weighted span's own text does not terminate the
    /// <c>(text:weight)</c> group early — the inverse of <see cref="PromptWeighting"/>'s unescape step.</summary>
    private static string EscapeParens(string text)
    {
        if (text.IndexOf('(') < 0 && text.IndexOf(')') < 0)
        {
            return text;
        }
        StringBuilder sb = new StringBuilder(text.Length);
        foreach (char c in text)
        {
            if (c is '(' or ')')
            {
                sb.Append('\\');
            }
            sb.Append(c);
        }
        return sb.ToString();
    }
}
