using System.Text;
using System.Text.Json;

namespace HartsyInference.Audio.Models.FishSpeech;

/// <summary>Fish-Speech / OpenAudio tiktoken-style byte-level BPE tokenizer. The vocab is asset-supplied
/// (102048 entries for fish-speech-1.5) and parsed at load time from one of two layouts:
/// <list type="bullet">
///   <item>a HuggingFace <c>tokenizer.json</c> (<c>model.vocab</c> token→id map + <c>model.merges</c> +
///   <c>added_tokens</c>), or</item>
///   <item>a tiktoken-style file (one <c>&lt;base64-bytes&gt; &lt;rank&gt;</c> per line) paired with a JSON
///   specials map.</item>
/// </list>
/// Nothing is hardcoded: the fish special tokens (<c>&lt;|im_start|&gt;</c>, <c>&lt;|im_end|&gt;</c>,
/// <c>&lt;|semantic:i|&gt;</c> range, <c>&lt;|audio_start|&gt;</c>/<c>&lt;|audio_end|&gt;</c>) are resolved from the
/// loaded vocab by string→id lookup. <see cref="SemanticBeginId"/> / <see cref="ImEndId"/> expose the ids the
/// DualAR pipeline needs.</summary>
public sealed class FishSpeechTokenizer
{
    public const string ImStart = "<|im_start|>";
    public const string ImEnd = "<|im_end|>";
    public const string AudioStart = "<|audio_start|>";
    public const string AudioEnd = "<|audio_end|>";
    public const string SemanticZero = "<|semantic:0|>";
    public const string PlainSemantic = "<|semantic|>";

    private readonly Dictionary<string, int> _tokenToId = new(StringComparer.Ordinal);
    private readonly Dictionary<int, string> _idToToken = new();
    private readonly Dictionary<(string, string), int> _mergeRanks = new();
    private readonly Dictionary<string, int> _specials = new(StringComparer.Ordinal);
    private readonly List<string> _specialsByLength = [];
    private readonly Dictionary<int, byte> _byteDecoder = new();
    private readonly Dictionary<byte, char> _byteEncoder;
    private int _semanticBeginId = -1;
    private bool _loaded;

    public FishSpeechTokenizer()
    {
        _byteEncoder = BuildByteEncoder();
        foreach (KeyValuePair<byte, char> kv in _byteEncoder) _byteDecoder[kv.Value] = kv.Key;
    }

    /// <summary>Token id of <c>&lt;|im_end|&gt;</c> (DualAR stop token), or -1 if the asset lacks it.</summary>
    public int ImEndId { get; private set; } = -1;

    /// <summary>Token id of <c>&lt;|im_start|&gt;</c>, or -1.</summary>
    public int ImStartId { get; private set; } = -1;

    /// <summary>Id of the first semantic token (<c>&lt;|semantic:0|&gt;</c>, or the plain <c>&lt;|semantic|&gt;</c>
    /// sentinel when the checkpoint uses a single semantic token). Semantic token <c>i</c> resolves to
    /// <c>SemanticBeginId + i</c> for the ranged variant.</summary>
    public int SemanticBeginId => _semanticBeginId;

    /// <summary>Total vocabulary size including special tokens.</summary>
    public int VocabSize => _tokenToId.Count;

    /// <summary>Loads vocab + merges from <paramref name="vocabPath"/> — a HF <c>tokenizer.json</c> or a plain
    /// token→id <c>vocab.json</c>. When a sibling <c>merges.txt</c> exists (or <paramref name="mergesPath"/> is
    /// given) it is parsed for BPE ranks.</summary>
    public void Load(string vocabPath, string? mergesPath = null)
    {
        if (string.IsNullOrWhiteSpace(vocabPath)) throw new ArgumentException("vocabPath required.", nameof(vocabPath));
        if (!File.Exists(vocabPath)) throw new FileNotFoundException("tokenizer asset not found.", vocabPath);
        using FileStream fs = File.OpenRead(vocabPath);
        using JsonDocument doc = JsonDocument.Parse(fs);
        JsonElement root = doc.RootElement;
        if (root.TryGetProperty("model", out JsonElement model) && model.TryGetProperty("vocab", out JsonElement vocab))
        {
            LoadHfTokenizerJson(root, model, vocab);
        }
        else
        {
            LoadFlatVocabJson(root);
        }
        if (mergesPath is not null && File.Exists(mergesPath)) LoadMerges(File.ReadLines(mergesPath));
        else
        {
            string sibling = Path.Combine(Path.GetDirectoryName(vocabPath) ?? ".", "merges.txt");
            if (File.Exists(sibling)) LoadMerges(File.ReadLines(sibling));
        }
        ResolveSpecials();
    }

    /// <summary>Loads from an in-memory token→id map (and optional merges + specials). Used for tests and for
    /// checkpoints that ship the vocab as a parsed dictionary.</summary>
    public void LoadFromVocab(IReadOnlyDictionary<string, int> vocab, IEnumerable<(string, string)>? merges = null,
        IEnumerable<string>? specials = null)
    {
        if (vocab is null || vocab.Count == 0) throw new ArgumentException("vocab must be non-empty.", nameof(vocab));
        foreach (KeyValuePair<string, int> kv in vocab) AddToken(kv.Key, kv.Value);
        if (specials is not null) foreach (string s in specials) RegisterSpecial(s);
        if (merges is not null)
        {
            int rank = 0;
            foreach ((string a, string b) in merges) _mergeRanks[(a, b)] = rank++;
        }
        ResolveSpecials();
    }

    /// <summary>Resolves a special token string to its id, or -1 if absent.</summary>
    public int SpecialId(string token) => _specials.TryGetValue(token, out int id) ? id : -1;

    /// <summary>Id of the <c>i</c>-th semantic token (<c>&lt;|semantic:i|&gt;</c>).</summary>
    public int SemanticId(int i)
    {
        if (_semanticBeginId < 0) throw new InvalidOperationException("No semantic tokens in this vocab.");
        return _semanticBeginId + i;
    }

    /// <summary>Encodes text to ids: greedily peels off registered special tokens, byte-level BPE on the rest.</summary>
    public int[] Encode(string text)
    {
        EnsureLoaded();
        List<int> ids = [];
        int pos = 0, runStart = 0;
        while (pos < text.Length)
        {
            string? hit = MatchSpecial(text, pos);
            if (hit is not null)
            {
                if (pos > runStart) EncodeChunk(text[runStart..pos], ids);
                ids.Add(_specials[hit]);
                pos += hit.Length;
                runStart = pos;
            }
            else pos++;
        }
        if (runStart < text.Length) EncodeChunk(text[runStart..], ids);
        return [.. ids];
    }

    /// <summary>Decodes ids back to text, dropping special tokens (so generated audio/semantic markers don't
    /// surface in rendered text).</summary>
    public string Decode(IReadOnlyList<int> ids)
    {
        EnsureLoaded();
        List<byte> bytes = new(ids.Count * 2);
        foreach (int id in ids)
        {
            if (!_idToToken.TryGetValue(id, out string? tok)) continue;
            if (_specials.ContainsKey(tok)) continue;
            foreach (char c in tok)
                if (_byteDecoder.TryGetValue(c, out byte b)) bytes.Add(b);
        }
        return Encoding.UTF8.GetString(bytes.ToArray());
    }

    private void LoadHfTokenizerJson(JsonElement root, JsonElement model, JsonElement vocab)
    {
        foreach (JsonProperty p in vocab.EnumerateObject()) AddToken(p.Name, p.Value.GetInt32());
        if (model.TryGetProperty("merges", out JsonElement merges) && merges.ValueKind == JsonValueKind.Array)
        {
            int rank = 0;
            foreach (JsonElement m in merges.EnumerateArray())
            {
                if (m.ValueKind == JsonValueKind.String)
                {
                    string s = m.GetString()!;
                    int sp = s.IndexOf(' ');
                    if (sp > 0) _mergeRanks[(s[..sp], s[(sp + 1)..])] = rank++;
                }
                else if (m.ValueKind == JsonValueKind.Array && m.GetArrayLength() == 2)
                {
                    _mergeRanks[(m[0].GetString()!, m[1].GetString()!)] = rank++;
                }
            }
        }
        if (root.TryGetProperty("added_tokens", out JsonElement added) && added.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement a in added.EnumerateArray())
            {
                if (!a.TryGetProperty("content", out JsonElement content)) continue;
                string tok = content.GetString()!;
                int id = a.TryGetProperty("id", out JsonElement idEl) ? idEl.GetInt32() : -1;
                if (id >= 0) AddToken(tok, id);
                RegisterSpecial(tok);
            }
        }
    }

    private void LoadFlatVocabJson(JsonElement root)
    {
        if (root.ValueKind != JsonValueKind.Object)
            throw new InvalidDataException("vocab.json must be a token→id object.");
        foreach (JsonProperty p in root.EnumerateObject()) AddToken(p.Name, p.Value.GetInt32());
    }

    private void LoadMerges(IEnumerable<string> lines)
    {
        int rank = _mergeRanks.Count;
        foreach (string line in lines)
        {
            if (line.Length == 0 || line.StartsWith('#')) continue;
            int sp = line.IndexOf(' ');
            if (sp <= 0) continue;
            (string, string) key = (line[..sp], line[(sp + 1)..]);
            if (!_mergeRanks.ContainsKey(key)) _mergeRanks[key] = rank++;
        }
    }

    private void AddToken(string token, int id)
    {
        _tokenToId[token] = id;
        _idToToken[id] = token;
    }

    private void RegisterSpecial(string token)
    {
        if (!_tokenToId.TryGetValue(token, out int id)) return;
        _specials[token] = id;
    }

    private void ResolveSpecials()
    {
        // Auto-detect specials by the fish <|...|> convention so a flat vocab.json without an added_tokens block
        // still exposes the markers.
        foreach (KeyValuePair<string, int> kv in _tokenToId)
            if (kv.Key.Length >= 4 && kv.Key.StartsWith("<|", StringComparison.Ordinal) &&
                kv.Key.EndsWith("|>", StringComparison.Ordinal))
                _specials[kv.Key] = kv.Value;

        _specialsByLength.Clear();
        _specialsByLength.AddRange(_specials.Keys);
        _specialsByLength.Sort((a, b) => b.Length.CompareTo(a.Length));

        ImStartId = SpecialId(ImStart);
        ImEndId = SpecialId(ImEnd);
        _semanticBeginId = SpecialId(SemanticZero);
        if (_semanticBeginId < 0) _semanticBeginId = SpecialId(PlainSemantic);
        _loaded = true;
    }

    private string? MatchSpecial(string text, int pos)
    {
        if (text[pos] != '<') return null;
        foreach (string sp in _specialsByLength)
            if (pos + sp.Length <= text.Length && string.CompareOrdinal(text, pos, sp, 0, sp.Length) == 0)
                return sp;
        return null;
    }

    private void EncodeChunk(string chunk, List<int> ids)
    {
        // Byte-level mapping then greedy BPE over the byte-symbols.
        byte[] raw = Encoding.UTF8.GetBytes(chunk);
        List<string> parts = new(raw.Length);
        foreach (byte b in raw) parts.Add(_byteEncoder[b].ToString());
        while (parts.Count > 1)
        {
            int bestRank = int.MaxValue, bestIdx = -1;
            for (int i = 0; i < parts.Count - 1; i++)
                if (_mergeRanks.TryGetValue((parts[i], parts[i + 1]), out int r) && r < bestRank)
                {
                    bestRank = r; bestIdx = i;
                }
            if (bestIdx < 0) break;
            parts[bestIdx] += parts[bestIdx + 1];
            parts.RemoveAt(bestIdx + 1);
        }
        foreach (string p in parts)
        {
            if (_tokenToId.TryGetValue(p, out int id)) ids.Add(id);
            else foreach (char ch in p) if (_tokenToId.TryGetValue(ch.ToString(), out int cid)) ids.Add(cid);
        }
    }

    private void EnsureLoaded()
    {
        if (!_loaded) throw new InvalidOperationException("FishSpeechTokenizer.Load must be called first.");
    }

    /// <summary>GPT-2 / tiktoken byte→unicode table — maps every byte to a printable codepoint so BPE merges
    /// operate on strings while staying byte-reversible.</summary>
    private static Dictionary<byte, char> BuildByteEncoder()
    {
        List<int> bs = [];
        for (int i = '!'; i <= '~'; i++) bs.Add(i);
        for (int i = '¡'; i <= '¬'; i++) bs.Add(i);
        for (int i = '®'; i <= 'ÿ'; i++) bs.Add(i);
        List<int> cs = [.. bs];
        int n = 0;
        Dictionary<byte, char> map = new();
        for (int b = 0; b < 256; b++)
        {
            if (!bs.Contains(b))
            {
                bs.Add(b);
                cs.Add(256 + n);
                n++;
            }
        }
        for (int i = 0; i < bs.Count; i++) map[(byte)bs[i]] = (char)cs[i];
        return map;
    }
}
