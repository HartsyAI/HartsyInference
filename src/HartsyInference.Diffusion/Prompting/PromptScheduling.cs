using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace HartsyInference.Diffusion.Prompting;

/// <summary>Resolves the universal bracket scheduling grammar into a per-step prompt. <c>[a|b|c]</c> alternates a different element each step; <c>[from:to:when]</c> switches from <c>from</c> to <c>to</c> at <c>when</c> (a fraction in (0,1) or an absolute step index); <c>[to:when]</c> is the same with an empty <c>from</c>. Groups may nest. Brackets with no <c>|</c> or <c>:</c> are dropped to their inner text. Emphasis <c>(…)</c> is preserved for <see cref="PromptWeighting"/> to handle afterwards.</summary>
public static class PromptScheduling
{
    /// <summary>True if the prompt contains at least one bracket group with a top-level <c>|</c> or <c>:</c> (i.e. its conditioning changes across steps).</summary>
    public static bool HasScheduling(string prompt)
    {
        if (prompt is null)
        {
            throw new ArgumentNullException(nameof(prompt));
        }
        for (int i = 0; i < prompt.Length; i++)
        {
            if (prompt[i] != '[')
            {
                continue;
            }
            int end = FindMatchingBracket(prompt, i);
            if (end < 0)
            {
                continue;
            }
            string inner = prompt.Substring(i + 1, end - i - 1);
            if (TopLevelContains(inner, '|') || TopLevelContains(inner, ':'))
            {
                return true;
            }
            i = end;
        }
        return false;
    }

    /// <summary>Resolves the prompt for a single step out of <paramref name="totalSteps"/>.</summary>
    public static string ResolveAt(string prompt, int step, int totalSteps)
    {
        if (prompt is null)
        {
            throw new ArgumentNullException(nameof(prompt));
        }
        StringBuilder sb = new StringBuilder(prompt.Length);
        int i = 0;
        while (i < prompt.Length)
        {
            char c = prompt[i];
            if (c == '[')
            {
                int end = FindMatchingBracket(prompt, i);
                if (end < 0)
                {
                    sb.Append(c);
                    i++;
                    continue;
                }
                string inner = prompt.Substring(i + 1, end - i - 1);
                sb.Append(EvaluateGroup(inner, step, totalSteps));
                i = end + 1;
            }
            else
            {
                sb.Append(c);
                i++;
            }
        }
        return sb.ToString();
    }

    /// <summary>Resolves all steps, deduplicates identical prompts, and returns the variant list plus a per-step index. When the prompt has no scheduling, returns a single variant for every step.</summary>
    public static PromptSchedule Resolve(string prompt, int totalSteps)
    {
        if (totalSteps <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(totalSteps), "totalSteps must be positive.");
        }
        List<string> variants = new List<string>();
        Dictionary<string, int> seen = new Dictionary<string, int>();
        int[] stepToVariant = new int[totalSteps];
        for (int s = 0; s < totalSteps; s++)
        {
            string resolved = ResolveAt(prompt, s, totalSteps);
            if (!seen.TryGetValue(resolved, out int idx))
            {
                idx = variants.Count;
                variants.Add(resolved);
                seen[resolved] = idx;
            }
            stepToVariant[s] = idx;
        }
        return new PromptSchedule(variants, stepToVariant);
    }

    private static string EvaluateGroup(string inner, int step, int totalSteps)
    {
        List<string> alternatives = TopLevelSplit(inner, '|');
        if (alternatives.Count > 1)
        {
            string chosen = alternatives[step % alternatives.Count];
            return ResolveAt(chosen, step, totalSteps);
        }
        List<string> parts = TopLevelSplit(inner, ':');
        if (parts.Count < 2)
        {
            return ResolveAt(inner, step, totalSteps);
        }
        string from;
        string to;
        string when;
        if (parts.Count >= 3)
        {
            from = parts[0];
            to = parts[1];
            when = parts[^1];
        }
        else
        {
            from = string.Empty;
            to = parts[0];
            when = parts[1];
        }
        int threshold = ParseWhen(when, totalSteps);
        string branch = step < threshold ? from : to;
        return ResolveAt(branch, step, totalSteps);
    }

    private static int ParseWhen(string when, int totalSteps)
    {
        string trimmed = when.Trim();
        if (!float.TryParse(trimmed, NumberStyles.Float, CultureInfo.InvariantCulture, out float value))
        {
            return 0;
        }
        int threshold;
        if (trimmed.Contains('.') || (value > 0f && value < 1f))
        {
            threshold = (int)MathF.Round(value * totalSteps);
        }
        else
        {
            threshold = (int)value;
        }
        return Math.Clamp(threshold, 0, totalSteps);
    }

    private static int FindMatchingBracket(string text, int openIndex)
    {
        int depth = 0;
        for (int i = openIndex; i < text.Length; i++)
        {
            if (text[i] == '[')
            {
                depth++;
            }
            else if (text[i] == ']')
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

    private static List<string> TopLevelSplit(string text, char separator)
    {
        List<string> parts = new List<string>();
        StringBuilder current = new StringBuilder();
        int depth = 0;
        foreach (char c in text)
        {
            if (c == '[')
            {
                depth++;
            }
            else if (c == ']' && depth > 0)
            {
                depth--;
            }
            if (c == separator && depth == 0)
            {
                parts.Add(current.ToString());
                current.Clear();
            }
            else
            {
                current.Append(c);
            }
        }
        parts.Add(current.ToString());
        return parts;
    }

    private static bool TopLevelContains(string text, char target)
    {
        int depth = 0;
        foreach (char c in text)
        {
            if (c == '[')
            {
                depth++;
            }
            else if (c == ']' && depth > 0)
            {
                depth--;
            }
            else if (c == target && depth == 0)
            {
                return true;
            }
        }
        return false;
    }
}
