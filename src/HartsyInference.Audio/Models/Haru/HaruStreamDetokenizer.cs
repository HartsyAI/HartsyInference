namespace HartsyInference.Audio.Models.Haru;

/// <summary>Turns the model's text stream into text as it arrives. A run of PAD and EOP (frames with no text) reads as one space, and a
/// character split over several tokens is held back until its last token has come. Where the upstream detokenizer would carry bytes still
/// waiting for their character across a gap (and let them grow without limit), this one gives them up as they stand, U+FFFD included,
/// when a gap starts or after <see cref="MaxHeldTokens"/> tokens: the text shown is never glued to the next word.</summary>
internal sealed class HaruStreamDetokenizer(Func<IReadOnlyList<int>, string> decode, int padId, int eopId)
{
    /// <summary>Most tokens held back waiting for one character; UTF-8 has at most four bytes, so more is not a character in progress.</summary>
    internal const int MaxHeldTokens = 4;

    private readonly List<int> _pending = [];
    private bool _inGap;

    /// <summary>The new text of this frame, or an empty string when it adds none.</summary>
    internal string Push(int token)
    {
        if (token == padId || token == eopId)
        {
            if (_inGap)
            {
                return string.Empty;
            }
            _inGap = true;
            return Flush() + " ";
        }
        _inGap = false;
        _pending.Add(token);
        string text = decode(_pending);
        if (text.EndsWith('\uFFFD') && _pending.Count < MaxHeldTokens)
        {
            return string.Empty;
        }
        _pending.Clear();
        return text;
    }

    private string Flush()
    {
        if (_pending.Count == 0)
        {
            return string.Empty;
        }
        string text = decode(_pending);
        _pending.Clear();
        return text;
    }
}
