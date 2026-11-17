using System.Text;

namespace HartsyInference.ModelAssets.Tokenizers;

/// <summary>Rewrites the Oniguruma (Ruby syntax) quantifier forms that Hugging Face <c>tokenizers</c> accepts in a <c>Split</c> pattern but .NET
/// rejects or reads differently:
/// <list type="bullet">
/// <item>An interval followed by <c>+</c> is a nested quantifier, not a possessive one: <c>X{3}+</c> is one or more runs of three, so it becomes
/// <c>(?:X{3})+</c>. ERNIE-5.5 groups digits from the right with <c>\p{Nd}{3}+</c>.</item>
/// <item><c>X++</c>, <c>X*+</c> and <c>X?+</c> are possessive, so they become the atomic <c>(?&gt;X+)</c>, <c>(?&gt;X*)</c> and <c>(?&gt;X?)</c>.</item>
/// <item><c>{,n}</c> is <c>{0,n}</c>; .NET would match it as literal text.</item>
/// </list>
/// Everything else, character classes included, is copied as written.</summary>
internal static class OnigurumaPattern
{
    /// <summary>The .NET form of <paramref name="pattern"/>; a pattern with none of the rewritten forms comes back unchanged.</summary>
    /// <exception cref="NotSupportedException">A quantifier that needs rewriting has nothing before it to apply to. That includes a quantifier
    /// that follows another one, such as <c>a{2}{3}+</c> or <c>\d+{2}+</c>: Oniguruma reads those as nested repeats, which this rewrite refuses
    /// rather than guess at (no known tokenizer uses them).</exception>
    public static string ToDotNet(string pattern)
    {
        ArgumentNullException.ThrowIfNull(pattern);
        StringBuilder sb = new(pattern.Length + 8);
        Stack<(int Start, bool Extended)> groups = new(); // each open group's start in sb and the extended mode outside it
        bool extended = false; // (?x): whitespace is ignored and '#' starts a comment that runs to the end of the line
        int atom = -1; // where the last complete atom starts in sb; -1 when a quantifier would have nothing to apply to
        int i = 0;
        while (i < pattern.Length)
        {
            char c = pattern[i];
            if (c == '\\')
            {
                atom = sb.Length;
                int length = EscapeLength(pattern, i);
                sb.Append(pattern, i, length);
                i += length;
            }
            else if (c == '[')
            {
                atom = sb.Length;
                int end = ClassEnd(pattern, i);
                sb.Append(pattern, i, end - i);
                i = end;
            }
            else if (extended && c == '#')
            {
                // a line comment is copied whole, so nothing in it is read as pattern
                int end = pattern.IndexOf('\n', i);
                if (end < 0) end = pattern.Length;
                sb.Append(pattern, i, end - i);
                i = end;
            }
            else if (extended && c is ' ' or '\t' or '\n' or '\v' or '\f' or '\r')
            {
                // ignored whitespace (only these ASCII characters; U+00A0 and the like stay literal) leaves the atom before it as the one a
                // quantifier repeats
                sb.Append(c);
                i++;
            }
            else if (c == '(' && pattern.AsSpan(i).StartsWith("(?#"))
            {
                // a comment means nothing, so its text is dropped (.NET would end it at an escaped ')' that Oniguruma skips) and an empty
                // (?#) stands in, so the tokens on either side stay apart (a{(?#)2} must not become the interval a{2}). One never closed is
                // copied as written for the regex parser to report.
                int length = CommentLength(pattern, i);
                if (length < 0)
                {
                    sb.Append(pattern, i, pattern.Length - i);
                    break;
                }
                sb.Append("(?#)");
                i += length;
                atom = -1;
            }
            else if (c == '(')
            {
                int start = sb.Length, length = GroupOpenLength(pattern, i);
                bool inner = ExtendedAfter(pattern.AsSpan(i, length), extended);
                sb.Append(pattern, i, length);
                i += length;
                atom = -1;
                // an inline option setting like (?x) closes at once, is not an atom, and holds to the end of the enclosing group
                if (sb[^1] != ')') groups.Push((start, extended));
                extended = inner;
            }
            else if (c == ')')
            {
                if (groups.Count == 0) throw new NotSupportedException($"The pattern '{pattern}' closes a group it never opened.");
                (atom, extended) = groups.Pop();
                sb.Append(c);
                i++;
            }
            else if (c == '{' && TryInterval(pattern, i, out int intervalEnd, out string interval))
            {
                i = AppendQuantifier(sb, pattern, intervalEnd, interval, ref atom, isInterval: true);
            }
            else if (c is '*' or '+' or '?')
            {
                i = AppendQuantifier(sb, pattern, i + 1, c.ToString(), ref atom, isInterval: false);
            }
            else
            {
                atom = sb.Length;
                sb.Append(c);
                i++;
            }
        }
        // an unbalanced pattern is left for the regex parser to report
        return sb.ToString();
    }

    // Appends one quantifier and its suffix (lazy ?, or Oniguruma's + that .NET lacks) and returns where scanning resumes.
    private static int AppendQuantifier(StringBuilder sb, string pattern, int next, string quantifier, ref int atom, bool isInterval)
    {
        char suffix = next < pattern.Length ? pattern[next] : '\0';
        if (suffix == '+')
        {
            if (atom < 0) throw new NotSupportedException($"The pattern '{pattern}' has a '{quantifier}+' with nothing before it to repeat.");
            // an interval then + repeats the interval; any other quantifier then + is possessive
            sb.Insert(atom, isInterval ? "(?:" : "(?>");
            sb.Append(isInterval ? quantifier + ")+" : quantifier + ")");
            next++;
        }
        else
        {
            sb.Append(quantifier);
            if (suffix == '?') { sb.Append('?'); next++; }
        }
        atom = -1;
        return next;
    }

    // {n}, {n,}, {n,m} or {,m} starting at start; Oniguruma reads any other brace as a literal.
    private static bool TryInterval(string pattern, int start, out int end, out string interval)
    {
        end = start + 1;
        int lowDigits = 0, highDigits = 0;
        bool comma = false;
        while (end < pattern.Length && char.IsAsciiDigit(pattern[end])) { end++; lowDigits++; }
        if (end < pattern.Length && pattern[end] == ',')
        {
            comma = true;
            end++;
            while (end < pattern.Length && char.IsAsciiDigit(pattern[end])) { end++; highDigits++; }
        }
        if (end >= pattern.Length || pattern[end] != '}' || lowDigits + highDigits == 0)
        {
            interval = "";
            return false;
        }
        end++;
        string body = pattern[(start + 1)..(end - 1)];
        interval = comma && lowDigits == 0 ? "{0" + body + "}" : "{" + body + "}";
        return true;
    }

    // \p{..}, \P{..}, \x{..} and \k<..> run to their closing bracket; \uHHHH is six characters, \xHH four; any other escape is two.
    private static int EscapeLength(string pattern, int start)
    {
        if (start + 1 >= pattern.Length) return 1;
        char kind = pattern[start + 1];
        if (start + 2 < pattern.Length)
        {
            char open = pattern[start + 2];
            if (kind is 'p' or 'P' or 'x' && open == '{') return Through(pattern, start, '}');
            if (kind == 'k' && open == '<') return Through(pattern, start, '>');
        }
        return kind switch
        {
            'u' => Math.Min(6, pattern.Length - start),
            'x' => Math.Min(4, pattern.Length - start),
            _ => 2,
        };
    }

    private static int Through(string pattern, int start, char close)
    {
        int end = pattern.IndexOf(close, start);
        return end < 0 ? pattern.Length - start : end - start + 1;
    }

    // The index just past the class opened at start, counting nested classes ([[:alpha:]], [a[b]]) and skipping escapes. A ']' right after
    // the opening '[' or '[^' is a literal.
    private static int ClassEnd(string pattern, int start)
    {
        int i = start + 1, depth = 1;
        if (i < pattern.Length && pattern[i] == '^') i++;
        if (i < pattern.Length && pattern[i] == ']') i++;
        while (i < pattern.Length)
        {
            char c = pattern[i];
            if (c == '\\') { i += EscapeLength(pattern, i); continue; }
            if (c == '[') depth++;
            else if (c == ']' && --depth == 0) return i + 1;
            i++;
        }
        return pattern.Length;
    }

    // The length of a (?#...) comment at start through its first unescaped ')' (Oniguruma skips the character after a '\' in a comment),
    // or -1 when none closes it.
    private static int CommentLength(string pattern, int start)
    {
        for (int j = start + 3; j < pattern.Length; j++)
        {
            if (pattern[j] == '\\') j++;
            else if (pattern[j] == ')') return j + 1 - start;
        }
        return -1;
    }

    // Whether (?x) holds after a group opener such as (?x), (?x:, (?i-x: or (?-x); any other opener keeps the current mode.
    private static bool ExtendedAfter(ReadOnlySpan<char> open, bool current)
    {
        if (open.Length < 3 || open[1] != '?' || !(char.IsAsciiLetter(open[2]) || open[2] == '-')) return current;
        bool on = true;
        foreach (char ch in open[2..])
        {
            if (ch == '-') on = false;
            else if (ch == 'x') current = on;
            else if (ch is ':' or ')') break;
        }
        return current;
    }

    // "(", or a "(?" construct through the character that starts its body: (?: (?= (?! (?> (?<= (?<! (?<name> (?i) (?i-m:.
    private static int GroupOpenLength(string pattern, int start)
    {
        int i = start + 1;
        if (i >= pattern.Length || pattern[i] != '?') return 1;
        i++;
        if (i >= pattern.Length) return i - start;
        char kind = pattern[i];
        if (kind is ':' or '=' or '!' or '>') return i + 1 - start;
        if (kind == '<')
        {
            if (i + 1 < pattern.Length && pattern[i + 1] is '=' or '!') return i + 2 - start;
            int close = pattern.IndexOf('>', i);
            return close < 0 ? pattern.Length - start : close + 1 - start;
        }
        while (i < pattern.Length && (char.IsAsciiLetter(pattern[i]) || pattern[i] == '-')) i++;
        return i < pattern.Length && pattern[i] is ':' or ')' ? i + 1 - start : i - start;
    }
}
