using System.Text;

namespace HartsyInference.Tokenizers;

/// <summary>The RWKV "World" tokenizer (<c>tokenizer.ggml.model == "rwkv"</c>): a fixed vocabulary of arbitrary
/// byte strings tokenized by GREEDY LONGEST-PREFIX MATCH over raw UTF-8 bytes — no BPE merges, no
/// SentencePiece unigram scoring. llama.cpp's GGUF converter stores each vocab entry as an escaped ASCII string
/// (<c>\t</c>/<c>\n</c>/<c>\r</c>/<c>\xNN</c> plus a literal backslash-escape fallback) so arbitrary/invalid-UTF8
/// byte sequences survive the GGUF string table; this class un-escapes them back to raw bytes at load and
/// matches llama.cpp's <c>llm_tokenizer_rwkv</c> byte-trie exactly (see <c>llama-vocab.cpp</c>).</summary>
public sealed class RwkvWorldTokenizer : ILlmTokenizer
{
    private const int TypeControl = 3;
    private const int TypeUserDefined = 4;

    /// <summary>Byte-trie node: one child per possible next byte, plus an optional token id when a vocab entry
    /// ends here (a token can be a byte-string that is also a strict prefix of a longer one).</summary>
    private sealed class TrieNode
    {
        public Dictionary<byte, TrieNode>? Children;
        public int TokenId = -1;
    }

    private readonly byte[][] _tokenBytes;    // id -> raw (unescaped) byte string
    private readonly TrieNode _root = new();
    private readonly Dictionary<string, int> _specialByLiteral;
    private readonly string[] _specialLiterals;   // sorted by length desc (longest-match, for chat-template literals)
    private readonly HashSet<int> _specialIds;

    public int? BosId { get; }
    public int? EosId { get; }
    public string? BosToken { get; }
    public string? EosToken { get; }
    public IReadOnlyList<int> StopIds { get; }

    /// <summary>Builds from GGUF metadata arrays. <paramref name="tokens"/> is the full vocab (index = id, each
    /// entry the llama.cpp-escaped byte string); <paramref name="tokenType"/> (optional) marks control/
    /// user-defined tokens (used only for special-literal recognition in rendered chat text — RWKV-World has no
    /// real chat template, so most callers fall back to ChatML and rarely hit this path).</summary>
    public RwkvWorldTokenizer(string[] tokens, int[]? tokenType, int? bosId, int? eosId,
        IReadOnlyList<int>? extraStopIds = null)
    {
        if (tokens is null || tokens.Length == 0) throw new ArgumentException("GGUF tokens array is empty.", nameof(tokens));

        _tokenBytes = new byte[tokens.Length][];
        _specialByLiteral = new Dictionary<string, int>(StringComparer.Ordinal);
        _specialIds = [];
        for (int id = 0; id < tokens.Length; id++)
        {
            byte[] raw = UnescapeToken(tokens[id]);
            _tokenBytes[id] = raw;
            Insert(raw, id);

            bool isSpecial = tokenType is not null && id < tokenType.Length
                ? tokenType[id] == TypeControl || tokenType[id] == TypeUserDefined
                : false;
            if (isSpecial && tokens[id].Length > 0)
            {
                _specialByLiteral[tokens[id]] = id;
                _specialIds.Add(id);
            }
        }
        _specialLiterals = _specialByLiteral.Keys.OrderByDescending(s => s.Length).ToArray();

        BosId = bosId;
        EosId = eosId;
        BosToken = bosId is int b && b >= 0 && b < tokens.Length ? tokens[b] : null;
        EosToken = eosId is int e && e >= 0 && e < tokens.Length ? tokens[e] : null;

        HashSet<int> stops = [];
        if (eosId is int eid) stops.Add(eid);
        if (extraStopIds is not null) foreach (int s in extraStopIds) stops.Add(s);
        StopIds = stops.ToArray();
    }

    /// <summary>Reverses llama.cpp's RWKV vocab escaping (<c>llama_unescape_rwkv_token</c>): <c>\t</c>/<c>\n</c>/
    /// <c>\r</c>/<c>\xHH</c> plus "backslash followed by any other char emits that char literally". Operates on
    /// the token's raw UTF-8 bytes (not .NET chars) so it exactly reproduces the original byte string, including
    /// byte sequences that are not valid UTF-8 on their own.</summary>
    private static byte[] UnescapeToken(string escaped)
    {
        byte[] src = Encoding.UTF8.GetBytes(escaped);
        List<byte> output = new(src.Length);
        bool escaping = false;
        int hexRemaining = 0;
        int hexAcc = 0;
        foreach (byte c in src)
        {
            if (hexRemaining != 0)
            {
                int value = c >= (byte)'a' ? c - (byte)'a' + 10 : c - (byte)'0';
                hexAcc = (hexAcc << 4) + value;
                hexRemaining--;
                if (hexRemaining == 0) { output.Add((byte)hexAcc); hexAcc = 0; }
                continue;
            }
            if (escaping)
            {
                if (c == (byte)'t') output.Add((byte)'\t');
                else if (c == (byte)'n') output.Add((byte)'\n');
                else if (c == (byte)'r') output.Add((byte)'\r');
                else if (c == (byte)'x') hexRemaining = 2;
                else output.Add(c);
                escaping = false;
                continue;
            }
            if (c == (byte)'\\') { escaping = true; continue; }
            output.Add(c);
        }
        return [.. output];
    }

    private void Insert(byte[] bytes, int id)
    {
        TrieNode node = _root;
        foreach (byte b in bytes)
        {
            node.Children ??= [];
            if (!node.Children.TryGetValue(b, out TrieNode? next)) { next = new TrieNode(); node.Children[b] = next; }
            node = next;
        }
        node.TokenId = id;
    }

    public int? SpecialId(string token) => _specialByLiteral.TryGetValue(token, out int id) ? id : null;

    public int[] EncodeOrdinary(string text)
    {
        if (text.Length == 0) return [];
        byte[] bytes = Encoding.UTF8.GetBytes(text);
        List<int> ids = new(bytes.Length / 2 + 4);
        int pos = 0;
        while (pos < bytes.Length)
        {
            TrieNode node = _root;
            int bestId = -1, bestLen = 0;
            int i = pos;
            while (i < bytes.Length && node.Children is not null && node.Children.TryGetValue(bytes[i], out TrieNode? next))
            {
                node = next;
                i++;
                if (node.TokenId >= 0) { bestId = node.TokenId; bestLen = i - pos; }
            }
            if (bestId < 0)
            {
                // No vocab entry starts with this byte at all — should not happen for a well-formed World
                // vocab (every raw byte 0..255 is its own token), but degrade by skipping rather than crashing.
                pos++;
                continue;
            }
            ids.Add(bestId);
            pos += bestLen;
        }
        return [.. ids];
    }

    public int[] Encode(string text, bool addSpecial)
    {
        if (!addSpecial || _specialLiterals.Length == 0) return EncodeOrdinary(text);

        List<int> ids = new(text.Length / 3 + 8);
        int i = 0;
        while (i < text.Length)
        {
            string? matched = MatchSpecialAt(text, i);
            if (matched is not null) { ids.Add(_specialByLiteral[matched]); i += matched.Length; continue; }
            int next = NextSpecialIndex(text, i);
            ids.AddRange(EncodeOrdinary(text[i..next]));
            i = next;
        }
        return [.. ids];
    }

    public string Decode(IReadOnlyList<int> ids)
    {
        List<byte> raw = new(ids.Count * 2);
        foreach (int id in ids)
        {
            if (id < 0 || id >= _tokenBytes.Length || _specialIds.Contains(id)) continue;
            raw.AddRange(_tokenBytes[id]);
        }
        return Encoding.UTF8.GetString([.. raw]);
    }

    private string? MatchSpecialAt(string text, int i)
    {
        foreach (string lit in _specialLiterals)
            if (i + lit.Length <= text.Length && string.CompareOrdinal(text, i, lit, 0, lit.Length) == 0)
                return lit;
        return null;
    }

    private int NextSpecialIndex(string text, int from)
    {
        int best = text.Length;
        foreach (string lit in _specialLiterals)
        {
            int idx = text.IndexOf(lit, from, StringComparison.Ordinal);
            if (idx >= 0 && idx < best) best = idx;
        }
        return best;
    }
}
