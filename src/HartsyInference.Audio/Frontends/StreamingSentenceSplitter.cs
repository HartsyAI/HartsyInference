using System.Text;

namespace HartsyInference.Audio.Frontends;

/// <summary>Applies <see cref="SentenceSplitter"/>'s rules to text that arrives a token at a time.</summary>
/// <remarks>A language model streams deltas that stop anywhere — mid-word, between "Dr" and its period, between
/// "3." and the "5" that makes it a decimal. The rules for where a sentence ends are the splitter's and are not
/// repeated here; this class only decides which of its pieces are final. Every piece but the last is: a boundary
/// the splitter accepted was followed by text that started a new sentence, and nothing that arrives later can
/// undo that. The last piece is never emitted from <see cref="Push"/>, because the splitter treats end of input as
/// a sentence end and a later delta may show it was not one; it comes out of <see cref="Flush"/> when the reply is
/// complete. The sentences emitted are therefore exactly what <see cref="SentenceSplitter.Split"/> returns for the
/// joined text, for any way of cutting it into deltas.
/// <para>The first sentence may be allowed shorter than the rest, so a reply's opening words can be spoken while
/// the model is still writing the sentence after them.</para></remarks>
public sealed class StreamingSentenceSplitter
{
    private readonly StringBuilder _pending = new();
    private readonly int _minLength;
    private readonly int _firstSentenceMinLength;
    private bool _firstEmitted;

    /// <summary>Creates a splitter that emits sentences of at least <paramref name="minLength"/> characters, except
    /// the first, which may be as short as <paramref name="firstSentenceMinLength"/> (defaults to <paramref name="minLength"/>).</summary>
    public StreamingSentenceSplitter(int minLength = SentenceSplitter.MinSentenceLength, int? firstSentenceMinLength = null)
    {
        if (minLength < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(minLength), minLength, "minLength must be non-negative");
        }
        int firstMin = firstSentenceMinLength ?? minLength;
        if (firstMin < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(firstSentenceMinLength), firstMin, "firstSentenceMinLength must be non-negative");
        }
        _minLength = minLength;
        _firstSentenceMinLength = firstMin;
    }

    /// <summary>Characters received and not yet emitted.</summary>
    public int PendingLength => _pending.Length;

    /// <summary>Appends <paramref name="delta"/> and returns every sentence it completed, in order; usually none.</summary>
    public IReadOnlyList<string> Push(string? delta)
    {
        if (string.IsNullOrEmpty(delta))
        {
            return [];
        }
        _pending.Append(delta);
        List<string>? completed = null;
        string buffer = _pending.ToString();
        while (true)
        {
            IReadOnlyList<string> parts = SentenceSplitter.Split(buffer, _firstEmitted ? _minLength : _firstSentenceMinLength);
            if (parts.Count <= 1)
            {
                break;
            }
            // Only the first piece is taken per pass: once it is out, the next sentence is subject to the normal
            // minimum, which a single split with the first-sentence minimum would not have applied.
            string first = parts[0];
            (completed ??= []).Add(first);
            _firstEmitted = true;
            int end = buffer.IndexOf(first, StringComparison.Ordinal) + first.Length;
            buffer = buffer[end..];
        }
        if (completed is null)
        {
            return [];
        }
        _pending.Clear();
        _pending.Append(buffer);
        return completed;
    }

    /// <summary>Returns whatever is still pending as the final sentence, or null when nothing is, and clears it.</summary>
    /// <remarks>The first-sentence minimum is not reset: a flush ends a reply, and the next reply starts with
    /// <see cref="Reset"/>.</remarks>
    public string? Flush()
    {
        string tail = _pending.ToString().Trim();
        _pending.Clear();
        return tail.Length == 0 ? null : tail;
    }

    /// <summary>Drops pending text and arms the first-sentence minimum again for a new reply.</summary>
    public void Reset()
    {
        _pending.Clear();
        _firstEmitted = false;
    }
}
