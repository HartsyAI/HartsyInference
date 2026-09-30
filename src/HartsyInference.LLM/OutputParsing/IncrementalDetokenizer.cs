using System.Text;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.OutputParsing;

/// <summary>Streams token ids to text in O(1) per token: bytes go through a stateful UTF-8 decoder that holds back an incomplete multibyte sequence. Tokenizers that cannot report per-token bytes fall back to re-decoding the running id list.</summary>
public sealed class IncrementalDetokenizer
{
    private readonly ILlmTokenizer _tokenizer;
    private readonly bool _includeSpecial;
    private readonly Decoder _decoder = new UTF8Encoding(false, false).GetDecoder();
    private readonly List<int> _ids = [];
    private char[] _chars = new char[64];
    private int _emitted;

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
    public string Flush() => DecodeBytes([], flush: true);

    private string DecodeBytes(byte[] bytes, bool flush)
    {
        int max = _decoder.GetCharCount(bytes, 0, bytes.Length, flush);
        if (max == 0 && bytes.Length == 0) return "";
        if (_chars.Length < max) _chars = new char[Math.Max(max, _chars.Length * 2)];
        int n = _decoder.GetChars(bytes, 0, bytes.Length, _chars, 0, flush);
        return new string(_chars, 0, n);
    }

    private string PushByRedecode(int id)
    {
        _ids.Add(id);
        string full = _tokenizer.Decode(_ids);
        if (full.Length <= _emitted) return "";
        string delta = full[_emitted..];
        _emitted = full.Length;
        return delta;
    }
}
