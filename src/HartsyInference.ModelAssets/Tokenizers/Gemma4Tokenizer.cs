using System.Text;
using System.Text.Json;

namespace HartsyInference.ModelAssets.Tokenizers;

/// <summary>Gemma 4's tokenizer, built from the HuggingFace <c>tokenizer.json</c> that LTX-2.5 ships <i>inside</i>
/// its text-encoder safetensors as the U8 <c>tokenizer_json</c> tensor. Despite the SentencePiece flavour
/// (spaces become <c>▁</c>, unknown characters fall back to <c>&lt;0xNN&gt;</c> pieces) the model is a genuine
/// rank-ordered <c>BPE</c>, so neither <see cref="HfTokenizerJson"/> (byte-level remaps the input through
/// <see cref="ByteLevelCodec"/> first) nor <see cref="SpmGgufTokenizer"/> (merges by score, not rank) can read it.</summary>
///
/// <remarks>Pipeline as declared by the real blob: normalizer <c>Replace(" " → "▁")</c>, then a
/// <c>Split(" ")</c> pre-tokenizer that is vestigial (normalization already consumed every space), so BPE runs
/// over the whole normalized string as a single word. <c>ignore_merges</c> is false and there is no
/// continuing-subword prefix or end-of-word suffix.</remarks>
public sealed class Gemma4Tokenizer
{
    /// <summary>Padding token id (<c>&lt;pad&gt;</c>).</summary>
    public const int PadTokenId = 0;

    /// <summary>End-of-sequence token id (<c>&lt;eos&gt;</c>). Never appended for conditioning.</summary>
    public const int EosTokenId = 1;

    /// <summary>Beginning-of-sequence token id (<c>&lt;bos&gt;</c>).</summary>
    public const int BosTokenId = 2;

    /// <summary>Sequence length LTX-2.5 pads Gemma 4 conditioning up to.</summary>
    public const int LtxMinLength = 1024;

    private const char SpaceMeta = '▁';

    private readonly string[] _tokens;
    private readonly Dictionary<string, int> _tokenToId;
    private readonly Dictionary<(string Left, string Right), int> _mergeRanks;
    private readonly Dictionary<byte, int> _byteToId;
    private readonly Dictionary<string, int> _specialByLiteral;
    private readonly string[] _specialLiterals;
    private readonly HashSet<int> _specialIds;
    private readonly int _unkId;

    private Gemma4Tokenizer(string[] tokens, Dictionary<string, int> tokenToId,
        Dictionary<(string, string), int> mergeRanks, Dictionary<byte, int> byteToId,
        Dictionary<string, int> specialByLiteral, int unkId)
    {
        _tokens = tokens;
        _tokenToId = tokenToId;
        _mergeRanks = mergeRanks;
        _byteToId = byteToId;
        _specialByLiteral = specialByLiteral;
        _specialLiterals = [.. specialByLiteral.Keys.OrderByDescending(s => s.Length)];
        _specialIds = [.. specialByLiteral.Values];
        _unkId = unkId;
    }

    /// <summary>Vocabulary size (highest id + 1).</summary>
    public int VocabSize => _tokens.Length;

    /// <summary>Parses a Gemma 4 <c>tokenizer.json</c>. Throws when the declared pipeline is not the one this
    /// implementation reproduces, rather than silently producing a different tokenization.</summary>
    public static Gemma4Tokenizer FromTokenizerJson(Stream json)
    {
        ArgumentNullException.ThrowIfNull(json);
        using JsonDocument doc = JsonDocument.Parse(json);
        return FromDocument(doc.RootElement);
    }

    /// <summary>Parses a Gemma 4 <c>tokenizer.json</c> held in memory — the form the LTX-2.5 checkpoint's
    /// <c>tokenizer_json</c> tensor takes.</summary>
    public static Gemma4Tokenizer FromTokenizerJson(ReadOnlySpan<byte> json)
    {
        using JsonDocument doc = JsonDocument.Parse(json.ToArray());
        return FromDocument(doc.RootElement);
    }

    private static Gemma4Tokenizer FromDocument(JsonElement root)
    {
        if (!root.TryGetProperty("model", out JsonElement model))
            throw new InvalidOperationException("Gemma 4 tokenizer.json has no 'model' object.");
        string? type = model.TryGetProperty("type", out JsonElement t) ? t.GetString() : null;
        if (!string.Equals(type, "BPE", StringComparison.Ordinal))
            throw new NotSupportedException($"Gemma4Tokenizer expects model.type 'BPE'; got '{type}'.");
        if (!(model.TryGetProperty("byte_fallback", out JsonElement bf) && bf.ValueKind == JsonValueKind.True))
            throw new NotSupportedException("Gemma4Tokenizer expects model.byte_fallback = true.");
        if (model.TryGetProperty("ignore_merges", out JsonElement im) && im.ValueKind == JsonValueKind.True)
            throw new NotSupportedException("Gemma4Tokenizer does not implement the ignore_merges shortcut.");
        ValidateNormalizer(root);

        Dictionary<string, int> tokenToId = new(StringComparer.Ordinal);
        JsonElement vocab = model.GetProperty("vocab");
        int maxId = -1;
        foreach (JsonProperty kv in vocab.EnumerateObject())
        {
            int id = kv.Value.GetInt32();
            tokenToId.TryAdd(kv.Name, id);
            if (id > maxId) maxId = id;
        }

        Dictionary<string, int> specialByLiteral = new(StringComparer.Ordinal);
        if (root.TryGetProperty("added_tokens", out JsonElement addedTokens) && addedTokens.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement added in addedTokens.EnumerateArray())
            {
                int id = added.GetProperty("id").GetInt32();
                string content = added.GetProperty("content").GetString()
                    ?? throw new InvalidOperationException("added_token has null content.");
                specialByLiteral[content] = id;
                tokenToId.TryAdd(content, id);
                if (id > maxId) maxId = id;
            }
        }

        string[] tokens = new string[maxId + 1];
        foreach (KeyValuePair<string, int> kv in tokenToId) tokens[kv.Value] = kv.Key;
        for (int i = 0; i < tokens.Length; i++) tokens[i] ??= string.Empty;

        Dictionary<(string, string), int> mergeRanks = ReadMerges(model);

        Dictionary<byte, int> byteToId = new(256);
        for (int b = 0; b < 256; b++)
        {
            if (tokenToId.TryGetValue($"<0x{b:X2}>", out int id)) byteToId[(byte)b] = id;
        }
        if (byteToId.Count != 256)
            throw new NotSupportedException($"Gemma 4 byte fallback needs all 256 <0xNN> pieces; found {byteToId.Count}.");

        string? unkToken = model.TryGetProperty("unk_token", out JsonElement unk) ? unk.GetString() : null;
        int unkId = unkToken is not null && tokenToId.TryGetValue(unkToken, out int u) ? u : -1;

        return new Gemma4Tokenizer(tokens, tokenToId, mergeRanks, byteToId, specialByLiteral, unkId);
    }

    /// <summary>Encodes text to ids with <b>no</b> special tokens added — literal special-token strings already in
    /// the text still resolve to their ids, which is what the reference tokenizer does.</summary>
    public int[] Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (text.Length == 0) return [];
        if (_specialLiterals.Length == 0) return EncodeOrdinary(text);

        List<int> ids = new(text.Length / 3 + 8);
        int i = 0;
        while (i < text.Length)
        {
            string? matched = MatchSpecialAt(text, i);
            if (matched is not null)
            {
                ids.Add(_specialByLiteral[matched]);
                i += matched.Length;
                continue;
            }
            int next = NextSpecialIndex(text, i);
            ids.AddRange(EncodeOrdinary(text[i..next]));
            i = next;
        }
        return [.. ids];
    }

    /// <summary>Encodes a prompt into the LTX-2.5 conditioning sequence: BOS exactly once, no EOS, right-padded
    /// with <see cref="PadTokenId"/> to at least <paramref name="minLength"/>.</summary>
    public int[] EncodeForConditioning(string text, int minLength = LtxMinLength) =>
        BuildConditioningSequence(Encode(text), minLength);

    /// <summary>Prepends BOS unless <paramref name="ids"/> already starts with it (upstream both dropped and
    /// duplicated it at different times), never appends EOS, and right-pads to <paramref name="minLength"/>.
    /// Sequences already longer than <paramref name="minLength"/> are returned unpadded — the reference caps
    /// length at <c>min_length</c>, not a maximum.</summary>
    public static int[] BuildConditioningSequence(ReadOnlySpan<int> ids, int minLength = LtxMinLength,
        int bosId = BosTokenId, int padId = PadTokenId)
    {
        if (minLength < 0) throw new ArgumentOutOfRangeException(nameof(minLength), "Minimum length cannot be negative.");
        bool hasBos = ids.Length > 0 && ids[0] == bosId;
        int contentLength = hasBos ? ids.Length : ids.Length + 1;
        int[] output = new int[Math.Max(contentLength, minLength)];
        int cursor = 0;
        if (!hasBos) output[cursor++] = bosId;
        for (int i = 0; i < ids.Length; i++) output[cursor++] = ids[i];
        for (int i = cursor; i < output.Length; i++) output[i] = padId;
        return output;
    }

    /// <summary>Decodes ids back to text, skipping special tokens and reassembling <c>&lt;0xNN&gt;</c> fallbacks.</summary>
    public string Decode(IReadOnlyList<int> ids)
    {
        ArgumentNullException.ThrowIfNull(ids);
        List<byte> bytes = new(ids.Count * 2);
        foreach (int id in ids)
        {
            if (id < 0 || id >= _tokens.Length || _specialIds.Contains(id)) continue;
            if (TryParseByteToken(_tokens[id], out byte raw)) { bytes.Add(raw); continue; }
            bytes.AddRange(Encoding.UTF8.GetBytes(_tokens[id].Replace(SpaceMeta, ' ')));
        }
        return Encoding.UTF8.GetString([.. bytes]);
    }

    /// <summary>Normalizes (space → <c>▁</c>) then runs rank-ordered BPE over the whole segment as one word.</summary>
    private int[] EncodeOrdinary(string text)
    {
        if (text.Length == 0) return [];
        string normalized = text.Replace(' ', SpaceMeta);

        List<string> symbols = [];
        for (int i = 0; i < normalized.Length;)
        {
            int len = char.IsHighSurrogate(normalized[i]) && i + 1 < normalized.Length ? 2 : 1;
            string piece = normalized.Substring(i, len);
            i += len;
            if (_tokenToId.ContainsKey(piece)) { symbols.Add(piece); continue; }
            // Byte fallback happens before merging, so the <0xNN> pieces can themselves participate in merges.
            foreach (byte b in Encoding.UTF8.GetBytes(piece))
                symbols.Add(_tokens[_byteToId[b]]);
        }

        MergeSymbols(symbols);

        List<int> ids = new(symbols.Count);
        foreach (string symbol in symbols)
        {
            if (_tokenToId.TryGetValue(symbol, out int id)) ids.Add(id);
            else if (_unkId >= 0) ids.Add(_unkId);
        }
        return [.. ids];
    }

    /// <summary>Applies BPE merges lowest-rank-first over a doubly-linked view of <paramref name="symbols"/>.
    /// A stale queue entry is detected by re-checking adjacency instead of eagerly purging the queue.</summary>
    private void MergeSymbols(List<string> symbols)
    {
        int count = symbols.Count;
        if (count < 2) return;

        string[] text = [.. symbols];
        int[] prev = new int[count];
        int[] next = new int[count];
        bool[] alive = new bool[count];
        for (int i = 0; i < count; i++)
        {
            prev[i] = i - 1;
            next[i] = i + 1 < count ? i + 1 : -1;
            alive[i] = true;
        }

        PriorityQueue<(int Left, int Right, string Merged), long> queue = new();
        for (int i = 0; i + 1 < count; i++) TryEnqueue(queue, text, i, i + 1);

        while (queue.TryDequeue(out (int Left, int Right, string Merged) pair, out long _))
        {
            int left = pair.Left, right = pair.Right;
            if (!alive[left] || !alive[right] || next[left] != right) continue;
            if (!string.Equals(text[left] + text[right], pair.Merged, StringComparison.Ordinal)) continue;

            text[left] = pair.Merged;
            alive[right] = false;
            int after = next[right];
            next[left] = after;
            if (after >= 0) prev[after] = left;

            int before = prev[left];
            if (before >= 0) TryEnqueue(queue, text, before, left);
            if (after >= 0) TryEnqueue(queue, text, left, after);
        }

        symbols.Clear();
        for (int i = 0; i >= 0 && i < count; i = next[i]) symbols.Add(text[i]);
    }

    private void TryEnqueue(PriorityQueue<(int, int, string), long> queue, string[] text, int left, int right)
    {
        if (!_mergeRanks.TryGetValue((text[left], text[right]), out int rank)) return;
        // Ties on rank are impossible (ranks are unique), so the left index only breaks ordering between
        // distinct ranks that never actually collide; it keeps the pop order deterministic regardless.
        queue.Enqueue((left, right, text[left] + text[right]), ((long)rank << 32) | (uint)left);
    }

    private static Dictionary<(string, string), int> ReadMerges(JsonElement model)
    {
        if (!model.TryGetProperty("merges", out JsonElement merges) || merges.ValueKind != JsonValueKind.Array)
            throw new InvalidOperationException("Gemma 4 tokenizer.json has no 'model.merges' array.");
        Dictionary<(string, string), int> ranks = new(merges.GetArrayLength());
        int rank = 0;
        foreach (JsonElement entry in merges.EnumerateArray())
        {
            string left, right;
            if (entry.ValueKind == JsonValueKind.Array && entry.GetArrayLength() == 2)
            {
                left = entry[0].GetString()!;
                right = entry[1].GetString()!;
            }
            else if (entry.ValueKind == JsonValueKind.String)
            {
                string raw = entry.GetString()!;
                int space = raw.IndexOf(' ');
                if (space <= 0) throw new InvalidOperationException($"Unparseable merge entry at rank {rank}.");
                left = raw[..space];
                right = raw[(space + 1)..];
            }
            else
            {
                throw new InvalidOperationException($"Unexpected merge entry at rank {rank}.");
            }
            ranks.TryAdd((left, right), rank);
            rank++;
        }
        return ranks;
    }

    private static void ValidateNormalizer(JsonElement root)
    {
        if (!root.TryGetProperty("normalizer", out JsonElement normalizer) || normalizer.ValueKind != JsonValueKind.Object)
            throw new NotSupportedException("Gemma4Tokenizer expects a Replace normalizer mapping ' ' to '▁'.");
        bool isReplace = normalizer.TryGetProperty("type", out JsonElement type) && type.GetString() == "Replace";
        bool fromSpace = normalizer.TryGetProperty("pattern", out JsonElement pattern)
            && pattern.TryGetProperty("String", out JsonElement patternString) && patternString.GetString() == " ";
        bool toMeta = normalizer.TryGetProperty("content", out JsonElement content) && content.GetString() == "▁";
        if (!isReplace || !fromSpace || !toMeta)
            throw new NotSupportedException("Gemma4Tokenizer expects a Replace normalizer mapping ' ' to '▁'.");
    }

    private static bool TryParseByteToken(string token, out byte value)
    {
        value = 0;
        if (token.Length != 6 || token[0] != '<' || token[1] != '0' || token[2] != 'x' || token[5] != '>')
            return false;
        return byte.TryParse(token.AsSpan(3, 2), System.Globalization.NumberStyles.HexNumber, null, out value);
    }

    private string? MatchSpecialAt(string text, int index)
    {
        foreach (string literal in _specialLiterals)
        {
            if (index + literal.Length <= text.Length
                && string.CompareOrdinal(text, index, literal, 0, literal.Length) == 0)
            {
                return literal;
            }
        }
        return null;
    }

    private int NextSpecialIndex(string text, int from)
    {
        int best = text.Length;
        foreach (string literal in _specialLiterals)
        {
            int index = text.IndexOf(literal, from, StringComparison.Ordinal);
            if (index >= 0 && index < best) best = index;
        }
        return best;
    }
}
