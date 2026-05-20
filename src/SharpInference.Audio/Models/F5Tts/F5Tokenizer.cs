using System.Globalization;
using System.Text;

namespace SharpInference.Audio.Models.F5Tts;

/// <summary>F5-TTS character-level tokenizer. The reference repo's <c>vocab.txt</c> is
/// a flat list of single characters (one per line), one ID per line, with ID 0 reserved
/// as a filler/pad. F5-TTS does NOT require G2P — raw text characters go in as token IDs
/// and the model learns the text-to-speech mapping internally.
///
/// <para>For multilingual fine-tunes (Mandarin, Japanese, etc.) the vocab may include
/// pinyin tokens or CJK characters; this tokenizer falls back to a Unicode-codepoint
/// lookup so any character not in the trained vocab simply maps to the filler token.</para></summary>
public sealed class F5Tokenizer : IDisposable
{
    private readonly Dictionary<string, int> _vocab;
    private readonly string[] _idToToken;
    private int _disposed;

    /// <summary>Vocab size including the implicit ID 0 filler (so 1 + line-count).</summary>
    public int VocabSize { get; }

    /// <summary>Loads the tokenizer from <c>vocab.txt</c> in the model directory. The
    /// file has one token per line; the line number (1-based) becomes the token ID,
    /// because the model uses 0 internally as a filler / batch-padding token.</summary>
    public F5Tokenizer(string modelDirectory)
    {
        string path = Path.Combine(modelDirectory, "vocab.txt");
        if (!File.Exists(path))
            throw new FileNotFoundException($"F5-TTS tokenizer requires vocab.txt at '{path}'", path);

        string[] lines = File.ReadAllLines(path);
        _vocab = new Dictionary<string, int>(lines.Length, StringComparer.Ordinal);
        _idToToken = new string[lines.Length + 1];
        _idToToken[0] = string.Empty;  // filler
        for (int i = 0; i < lines.Length; i++)
        {
            string tok = lines[i];
            // Lines may contain a single char or a multi-char token (pinyin syllables, etc.).
            _vocab[tok] = i;  // 0-based id mapped from file row
            _idToToken[i + 1] = tok;
        }
        VocabSize = lines.Length + 1;  // +1 for the filler token
    }

    /// <summary>Tokenizes a string into character token IDs. Unknown characters silently
    /// map to the filler ID 0 (matching the upstream <c>list_str_to_idx</c> behavior with
    /// <c>extra_token_idx</c>=0).</summary>
    public int[] Encode(string text)
    {
        ThrowIfDisposed();
        StringInfo si = new(text);
        int n = si.LengthInTextElements;
        int[] ids = new int[n];
        for (int i = 0; i < n; i++)
        {
            string ch = si.SubstringByTextElements(i, 1);
            ids[i] = _vocab.TryGetValue(ch, out int id) ? id : 0;
        }
        return ids;
    }

    /// <summary>Decodes a sequence of token IDs to a string. Used only for sanity
    /// printing — F5-TTS is a synthesizer, never produces token IDs as output.</summary>
    public string Decode(ReadOnlySpan<int> ids)
    {
        ThrowIfDisposed();
        StringBuilder sb = new(ids.Length);
        foreach (int id in ids)
        {
            // Reference convention: vocab has 0-based IDs from file; at inference the
            // pipeline adds +1 (using 0 as filler). We accept either by clamping.
            int idx = id < 0 ? 0 : id;
            if (idx < _idToToken.Length) sb.Append(_idToToken[idx]);
        }
        return sb.ToString();
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0) throw new ObjectDisposedException(nameof(F5Tokenizer));
    }

    public void Dispose() { Interlocked.Exchange(ref _disposed, 1); }
}
