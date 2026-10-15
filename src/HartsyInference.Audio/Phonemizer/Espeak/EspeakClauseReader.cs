using System.Text;

namespace HartsyInference.Audio.Phonemizer.Espeak;

/// <summary>Cuts English text into the clauses espeak-ng's clause reader finds, with the mark each one ends on, so a
/// caller can phonemize clause by clause and put the marks back as piper_phonemize does. A mark ends a clause only
/// when a space, the end or other punctuation follows it, and a period before a lowercase word is an abbreviation dot.</summary>
internal static class EspeakClauseReader
{
    private const string MarkRun = ".,;:?!…";
    private const string Dashes = "—–⸺";
    private const char NoMark = '\0';

    /// <summary>The clauses of <paramref name="text"/> in order; a clause's text can be blank when only punctuation precedes it.</summary>
    internal static List<EspeakClause> Read(string text)
    {
        List<EspeakClause> clauses = new();
        StringBuilder buffer = new();
        int i = 0;
        while (i < text.Length)
        {
            char c = text[i];
            if (c == '…')
            {
                End(clauses, buffer, NoMark);
                i++;
            }
            else if (".,;:?!".Contains(c))
            {
                i = ReadMarkRun(text, i, clauses, buffer);
            }
            else if (Dashes.Contains(c))
            {
                i = ReadDash(text, i, clauses, buffer);
            }
            else
            {
                buffer.Append(c);
                i++;
            }
        }
        if (buffer.ToString().Trim().Length > 0) End(clauses, buffer, NoMark);
        return clauses;
    }

    private static int ReadMarkRun(string text, int start, List<EspeakClause> clauses, StringBuilder buffer)
    {
        int end = SkipMarks(text, start);
        char mark = text[start];
        if (string.CompareOrdinal(text, start, "...", 0, 3) == 0)
        {
            End(clauses, buffer, NoMark);
            return end;
        }
        if (EndsClause(text, mark, end)) End(clauses, buffer, mark);
        else if (mark is '?' or '!') End(clauses, buffer, NoMark);
        else buffer.Append(mark == '.' && start > 0 && char.IsWhiteSpace(text[start - 1]) ? " dot " : " ");
        return end;
    }

    private static int ReadDash(string text, int start, List<EspeakClause> clauses, StringBuilder buffer)
    {
        int next = start + 1;
        if (next < text.Length && char.IsLetterOrDigit(text[next]))
        {
            buffer.Append(' ');
            return next;
        }
        End(clauses, buffer, ';');
        return SkipMarks(text, next);
    }

    private static bool EndsClause(string text, char mark, int afterRun)
    {
        if (mark != '.') return afterRun >= text.Length || !char.IsLetterOrDigit(text[afterRun]);
        if (afterRun >= text.Length) return true;
        if (!char.IsWhiteSpace(text[afterRun])) return !char.IsLetterOrDigit(text[afterRun]);
        int word = afterRun;
        while (word < text.Length && char.IsWhiteSpace(text[word])) word++;
        return word >= text.Length || !char.IsLower(text[word]);
    }

    private static int SkipMarks(string text, int index)
    {
        while (index < text.Length && MarkRun.Contains(text[index])) index++;
        return index;
    }

    private static void End(List<EspeakClause> clauses, StringBuilder buffer, char mark)
    {
        clauses.Add(new EspeakClause(buffer.ToString(), mark));
        buffer.Clear();
    }
}
