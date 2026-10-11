using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.ChatTemplates;

/// <summary>Keeps user-controlled text from becoming control tokens. A chat template renders control literals (<c>&lt;|im_start|&gt;</c>, <c>&lt;tool_call&gt;</c>) into the prompt, and the prompt is then tokenized with those literals as special ids, so message content carrying the same literal would open a turn or emit a tool marker. <see cref="Escape"/> swaps the first character of each such literal in content for a private-use character before rendering; <see cref="EncodeRendered"/> maps the template's own literals to ids and encodes the content spans as ordinary text with the characters restored.</summary>
/// <remarks>The reasoning markers are the one exception: Qwen3's template splits assistant content on <c>&lt;/think&gt;</c> to separate reasoning from the answer, so they must stay literal in content. A user's reasoning marker can therefore only close or open a reasoning block, never a turn.</remarks>
public static class SpecialLiteralEscaper
{
    /// <summary>The literals a model may write around its reasoning. Not escaped; see the remarks.</summary>
    public static IReadOnlyList<string> ReasoningMarkers { get; } = ["<think>", "</think>"];

    private const char PrivateUseStart = '';
    private const char PrivateUseEnd = '';

    /// <summary>Escapes every occurrence of the non-reasoning <paramref name="literals"/> in <paramref name="text"/>. Private-use characters already in the text become U+FFFD first, so the restore step is unambiguous. Returns the text unchanged when there are no literals to guard.</summary>
    public static string Escape(string text, IReadOnlyList<string> literals)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(literals);
        List<string> guarded = Guarded(literals);
        if (guarded.Count == 0 || text.Length == 0) return text;
        Dictionary<char, char> forward = FirstCharMap(guarded);
        System.Text.StringBuilder output = new(text.Length + 8);
        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            if (c >= PrivateUseStart && c <= PrivateUseEnd)
            {
                output.Append('�');
                continue;
            }
            string? literal = LiteralAt(text, i, guarded);
            if (literal is null)
            {
                output.Append(c);
                continue;
            }
            output.Append(forward[literal[0]]);
            output.Append(literal, 1, literal.Length - 1);
            i += literal.Length - 1;
        }
        return output.ToString();
    }

    /// <summary>Encodes a rendered prompt: the tokenizer's special literals become their ids, and the text between them is restored and encoded ordinarily. Equal to <c>tokenizer.Encode(rendered, addSpecial: true)</c> whenever no content escaped anything.</summary>
    public static int[] EncodeRendered(string rendered, ILlmTokenizer tokenizer)
    {
        ArgumentNullException.ThrowIfNull(rendered);
        ArgumentNullException.ThrowIfNull(tokenizer);
        IReadOnlyList<string> literals = tokenizer.SpecialLiterals;
        if (literals.Count == 0) return tokenizer.Encode(rendered, addSpecial: true);
        List<string> all = [.. literals.OrderByDescending(l => l.Length)];
        Dictionary<char, char> forward = FirstCharMap(Guarded(all));
        Dictionary<char, char> restore = forward.ToDictionary(kv => kv.Value, kv => kv.Key);
        List<int> ids = new(rendered.Length / 3 + 8);
        System.Text.StringBuilder span = new();
        int i = 0;
        while (i < rendered.Length)
        {
            string? literal = LiteralAt(rendered, i, all);
            if (literal is not null)
            {
                FlushSpan(span, restore, tokenizer, ids);
                ids.Add(tokenizer.SpecialId(literal) ?? throw new InvalidOperationException($"Special literal '{literal}' has no id."));
                i += literal.Length;
                continue;
            }
            span.Append(rendered[i]);
            i++;
        }
        FlushSpan(span, restore, tokenizer, ids);
        return [.. ids];
    }

    private static void FlushSpan(System.Text.StringBuilder span, Dictionary<char, char> restore, ILlmTokenizer tokenizer, List<int> ids)
    {
        if (span.Length == 0) return;
        for (int k = 0; k < span.Length; k++)
        {
            if (restore.TryGetValue(span[k], out char original)) span[k] = original;
        }
        ids.AddRange(tokenizer.EncodeOrdinary(span.ToString()));
        span.Clear();
    }

    /// <summary>The literals that get escaped: all of them except the reasoning markers, longest first.</summary>
    private static List<string> Guarded(IReadOnlyList<string> literals)
    {
        List<string> guarded = [];
        foreach (string literal in literals)
        {
            if (literal.Length == 0 || ReasoningMarkers.Contains(literal)) continue;
            guarded.Add(literal);
        }
        guarded.Sort((a, b) => b.Length.CompareTo(a.Length));
        return guarded;
    }

    /// <summary>One private-use character per distinct first character, in first-seen order; both directions derive from the same literal list, so they agree.</summary>
    private static Dictionary<char, char> FirstCharMap(IReadOnlyList<string> literals)
    {
        Dictionary<char, char> map = new();
        foreach (string literal in literals)
        {
            if (map.ContainsKey(literal[0])) continue;
            int next = PrivateUseStart + map.Count;
            if (next > PrivateUseEnd) throw new InvalidOperationException("Too many distinct special-literal first characters to escape.");
            map[literal[0]] = (char)next;
        }
        return map;
    }

    /// <summary>The first literal (longest first) that starts at <paramref name="at"/>, or null.</summary>
    private static string? LiteralAt(string text, int at, IReadOnlyList<string> literals)
    {
        foreach (string literal in literals)
        {
            if (literal.Length <= text.Length - at && string.CompareOrdinal(text, at, literal, 0, literal.Length) == 0) return literal;
        }
        return null;
    }
}
