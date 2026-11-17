namespace HartsyInference.Audio.Models.Haru;

/// <summary>Turns the model's text stream into text as it arrives. A run of PAD and EOP (frames with no text) reads as one space, and a
/// character split over several tokens is held back until its last token has come.</summary>
internal sealed class HaruStreamDetokenizer(Func<IReadOnlyList<int>, string> decode, int padId, int eopId)
{
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
            return " ";
        }
        _inGap = false;
        _pending.Add(token);
        string text = decode(_pending);
        if (text.EndsWith('\uFFFD'))
        {
            return string.Empty;
        }
        _pending.Clear();
        return text;
    }
}
