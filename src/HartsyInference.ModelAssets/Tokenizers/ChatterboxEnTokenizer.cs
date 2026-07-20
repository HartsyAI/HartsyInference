using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace HartsyInference.ModelAssets.Tokenizers;

/// <summary>Text tokenizer for Resemble AI's Chatterbox TTS (the upstream <c>EnTokenizer</c>). The model's T3 LM
/// consumes a tiny plain-character BPE vocabulary (704 English entries, 265 merges) rather than a byte-level BPE,
/// so this reproduces the HuggingFace <c>tokenizers</c> pipeline for that specific config: no normalizer, a
/// <c>Whitespace</c> pre-tokenizer (<c>\w+|[^\w\s]+</c>), and BPE over the character vocab. Spaces are mapped to
/// the literal <c>[SPACE]</c> token before encoding (the upstream convention), and the bracketed special tokens
/// (<c>[SPACE]</c>, <c>[START]</c>, <c>[STOP]</c>, sound-effect tags, ...) are matched whole.
///
/// <para>The vocabulary is embedded (<see cref="EmbeddedTokenizerResources.ChatterboxTokenizerJsonName"/>) so the
/// model ships self-contained. Validated to bit-exact parity against the reference <c>tokenizers</c> library.</para></summary>
public sealed class ChatterboxEnTokenizer
{
    private static readonly Regex WhitespaceSplit = new(@"\w+|[^\w\s]+", RegexOptions.Compiled);

    private readonly Dictionary<string, int> _vocab;
    private readonly Dictionary<(string, string), int> _ranks;
    private readonly Dictionary<string, int> _special;
    private readonly string[] _specialsByLength;

    /// <summary>Id of <c>[UNK]</c> (1), emitted for characters outside the vocab.</summary>
    public int UnkId { get; }

    /// <summary>Id of <c>[SPACE]</c> (2), the inter-word marker.</summary>
    public int SpaceId { get; }

    /// <summary>Id of <c>[START]</c> (255), the T3 begin-of-text marker.</summary>
    public int StartId { get; }

    /// <summary>Id of <c>[STOP]</c> (0), the T3 end-of-text marker.</summary>
    public int StopId { get; }

    /// <summary>Loads the embedded Chatterbox tokenizer.</summary>
    public ChatterboxEnTokenizer()
        : this(EmbeddedTokenizerResources.OpenChatterboxTokenizerJson(), ownsStream: true) { }

    /// <summary>Loads from a <c>tokenizer.json</c> stream. When <paramref name="ownsStream"/> is true the stream
    /// is disposed after parsing.</summary>
    public ChatterboxEnTokenizer(Stream tokenizerJson, bool ownsStream = false)
    {
        ArgumentNullException.ThrowIfNull(tokenizerJson);
        try
        {
            using JsonDocument doc = JsonDocument.Parse(tokenizerJson);
            JsonElement root = doc.RootElement;
            JsonElement model = root.GetProperty("model");

            _vocab = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (JsonProperty kv in model.GetProperty("vocab").EnumerateObject())
            {
                _vocab[kv.Name] = kv.Value.GetInt32();
            }

            _ranks = new Dictionary<(string, string), int>();
            int rank = 0;
            foreach (JsonElement m in model.GetProperty("merges").EnumerateArray())
            {
                (string a, string b) pair;
                if (m.ValueKind == JsonValueKind.String)
                {
                    string[] parts = m.GetString()!.Split(' ');
                    pair = (parts[0], parts[1]);
                }
                else
                {
                    pair = (m[0].GetString()!, m[1].GetString()!);
                }
                _ranks[pair] = rank++;
            }

            _special = new Dictionary<string, int>(StringComparer.Ordinal);
            if (root.TryGetProperty("added_tokens", out JsonElement added) && added.ValueKind == JsonValueKind.Array)
            {
                foreach (JsonElement t in added.EnumerateArray())
                {
                    _special[t.GetProperty("content").GetString()!] = t.GetProperty("id").GetInt32();
                }
            }
            // Match longest specials first so e.g. "[SPACE]" is not partially matched.
            _specialsByLength = _special.Keys.OrderByDescending(s => s.Length).ToArray();

            UnkId = _special.GetValueOrDefault("[UNK]", 1);
            SpaceId = _special.GetValueOrDefault("[SPACE]", 2);
            StartId = _special.GetValueOrDefault("[START]", 255);
            StopId = _special.GetValueOrDefault("[STOP]", 0);
        }
        finally
        {
            if (ownsStream)
            {
                tokenizerJson.Dispose();
            }
        }
    }

    /// <summary>Encodes text into Chatterbox token ids, reproducing the upstream <c>EnTokenizer.encode</c>: spaces
    /// become <c>[SPACE]</c>, then the HuggingFace Whitespace+BPE pipeline runs. Does not add <c>[START]</c>/
    /// <c>[STOP]</c> (the T3 caller wraps the sequence).</summary>
    public int[] Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        string prepared = text.Replace(" ", "[SPACE]");
        List<int> ids = new();

        int i = 0;
        int n = prepared.Length;
        while (i < n)
        {
            string? special = MatchSpecialAt(prepared, i);
            if (special is not null)
            {
                ids.Add(_special[special]);
                i += special.Length;
                continue;
            }
            // Accumulate a normal run up to the next special, then Whitespace + BPE it.
            int start = i;
            while (i < n && MatchSpecialAt(prepared, i) is null)
            {
                i++;
            }
            EncodeNormalRun(prepared.Substring(start, i - start), ids);
        }
        return ids.ToArray();
    }

    /// <summary>Convenience: encode and wrap with <c>[START]</c> ... <c>[STOP]</c> for direct T3 input.</summary>
    public int[] EncodeWithStartStop(string text)
    {
        int[] core = Encode(text);
        int[] outIds = new int[core.Length + 2];
        outIds[0] = StartId;
        Array.Copy(core, 0, outIds, 1, core.Length);
        outIds[^1] = StopId;
        return outIds;
    }

    private string? MatchSpecialAt(string s, int pos)
    {
        // Specials all start with '[' — cheap reject before scanning the list.
        if (s[pos] != '[')
        {
            return null;
        }
        foreach (string sp in _specialsByLength)
        {
            if (pos + sp.Length <= s.Length && string.CompareOrdinal(s, pos, sp, 0, sp.Length) == 0)
            {
                return sp;
            }
        }
        return null;
    }

    private void EncodeNormalRun(string run, List<int> ids)
    {
        foreach (Match m in WhitespaceSplit.Matches(run))
        {
            BpeEncode(m.Value, ids);
        }
    }

    private void BpeEncode(string token, List<int> ids)
    {
        // Decompose into Unicode-scalar symbols.
        List<string> symbols = new(token.Length);
        foreach (Rune r in token.EnumerateRunes())
        {
            symbols.Add(r.ToString());
        }
        if (symbols.Count == 0)
        {
            return;
        }

        // Canonical BPE: repeatedly merge every occurrence of the lowest-ranked adjacent pair.
        while (symbols.Count > 1)
        {
            int bestRank = int.MaxValue;
            int bestPairFirst = -1;
            for (int i = 0; i < symbols.Count - 1; i++)
            {
                if (_ranks.TryGetValue((symbols[i], symbols[i + 1]), out int r) && r < bestRank)
                {
                    bestRank = r;
                    bestPairFirst = i;
                }
            }
            if (bestPairFirst < 0)
            {
                break;
            }
            string a = symbols[bestPairFirst];
            string b = symbols[bestPairFirst + 1];
            List<string> merged = new(symbols.Count);
            int j = 0;
            while (j < symbols.Count)
            {
                if (j < symbols.Count - 1 && symbols[j] == a && symbols[j + 1] == b)
                {
                    merged.Add(a + b);
                    j += 2;
                }
                else
                {
                    merged.Add(symbols[j]);
                    j++;
                }
            }
            symbols = merged;
        }

        foreach (string sym in symbols)
        {
            ids.Add(_vocab.TryGetValue(sym, out int id) ? id : UnkId);
        }
    }
}
