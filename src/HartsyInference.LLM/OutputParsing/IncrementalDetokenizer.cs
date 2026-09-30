using System.Text;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.OutputParsing;

/// <summary>Streams token ids to text in O(1) per token: bytes go through a stateful UTF-8 decoder that holds back an incomplete multibyte sequence. Tokenizers that cannot report per-token bytes decode a short window from the last emitted boundary (the tokens emitted there stay in the window as context) and hold a delta back while it ends in U+FFFD, so concatenated deltas still equal a one-shot decode.</summary>
public sealed class IncrementalDetokenizer
{
    private readonly ILlmTokenizer _tokenizer;
    private readonly bool _includeSpecial;
    private readonly Decoder _decoder = new UTF8Encoding(false, false).GetDecoder();
    private readonly List<int> _window = [];
    private char[] _chars = new char[64];
    private int _read;
    private string _prefixText = "";

    /// <summary>Creates a detokenizer; <paramref name="includeSpecial"/> keeps control tokens as their literal text instead of skipping them.</summary>
    public IncrementalDetokenizer(ILlmTokenizer tokenizer, bool includeSpecial)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        _tokenizer = tokenizer;
        _includeSpecial = includeSpecial;
    }

    /// <summary>Adds one token and returns the text that became decodable (possibly empty).</summary>
    public string Push(int id)
    {
        byte[]? bytes = _tokenizer.TokenBytes(id, _includeSpecial);
        if (bytes is null) return PushByRedecode(id);
        return DecodeBytes(bytes, flush: false);
    }

    /// <summary>Ends the stream; an unfinished multibyte sequence is emitted as U+FFFD, matching a one-shot decode.</summary>
    public string Flush() => DecodeBytes([], flush: true) + FlushWindow();

    private string DecodeBytes(byte[] bytes, bool flush)
    {
        int max = _decoder.GetCharCount(bytes, 0, bytes.Length, flush);
        if (max == 0 && bytes.Length == 0) return "";
        if (_chars.Length < max) _chars = new char[Math.Max(max, _chars.Length * 2)];
        int n = _decoder.GetChars(bytes, 0, bytes.Length, _chars, 0, flush);
        return new string(_chars, 0, n);
    }

    /// <summary>Windowed decode for tokenizers without per-token bytes. The window is the tokens emitted at the last pivot followed by every token since; the emitted prefix is subtracted so a decoder that treats its first token specially (dummy-prefix stripping) cancels out.</summary>
    private string PushByRedecode(int id)
    {
        _window.Add(id);
        string text = _tokenizer.Decode(_window);
        // A decoder that normalizes across the boundary breaks the prefix invariant; slicing at the common prefix
        // never drops text (it may repeat a normalized character). Held back while the newest character is
        // incomplete (U+FFFD from a partial sequence) or nothing new decoded yet.
        int keep = text.AsSpan().CommonPrefixLength(_prefixText);
        if (keep == text.Length || text[^1] == '�') return "";
        string delta = text[keep..];
        // Pivot only lands on a character boundary (the text above did not end in U+FFFD), so the next window's
        // decode is a clean continuation of what was emitted.
        _window.RemoveRange(0, _read);
        _read = _window.Count;
        _prefixText = _tokenizer.Decode(_window);
        return delta;
    }

    private string FlushWindow()
    {
        if (_window.Count == _read) return "";
        string text = _tokenizer.Decode(_window);
        string tail = text[text.AsSpan().CommonPrefixLength(_prefixText)..];
        _read = _window.Count;
        _prefixText = text;
        return tail;
    }
}
