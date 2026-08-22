using System.Text;
using System.Text.Json;

namespace HartsyInference.Audio.Models.Moonshine;

/// <summary>Minimal Moonshine tokenizer that reads a HuggingFace <c>tokenizer.json</c> directly and supports decode only — the decoder prompt is just <c>[BOS]</c>, so BPE encode isn't needed for the v1 STT path.</summary>
// SentencePiece byte-fallback BPE (Llama-2 style): token 0 = <unk>, 1 = <s>, 2 = </s>; IDs 3..258 are
// byte-fallback tokens <0x00>..<0xFF>; IDs 259..31999 are BPE merges over ▁-prefixed sequences
// (SentencePiece "Metaspace" encoding where U+2581 stands in for space); IDs 32000..32767 are reserved
// special tokens (<<ST_0>>..<<ST_767>>).
// Decode walks the IDs, accumulates runs of byte-fallback tokens as raw bytes (UTF-8 multi-byte
// sequences span multiple BPE tokens), and flushes runs with the standard ▁ → space replacement.
public sealed class MoonshineTokenizer : IDisposable
{
    private readonly string?[] _idToToken;
    private readonly bool[] _isByteFallback;
    private readonly byte[] _byteFallbackValue;
    private readonly int _vocabSize;
    private int _disposed;

    /// <summary>Returns the total vocab size (includes special tokens + byte-fallback).</summary>
    public int VocabSize => _vocabSize;

    /// <summary>Loads the tokenizer from a HuggingFace-format <c>tokenizer.json</c> living next to <c>model.safetensors</c> in the model directory.</summary>
    public MoonshineTokenizer(string modelDirectory)
    {
        string path = Path.Combine(modelDirectory, "tokenizer.json");
        if (!File.Exists(path))
            throw new FileNotFoundException($"Moonshine tokenizer requires tokenizer.json at '{path}'", path);

        using FileStream fs = File.OpenRead(path);
        using JsonDocument doc = JsonDocument.Parse(fs);
        JsonElement root = doc.RootElement;

        // Parse model.vocab and added_tokens to build a flat id → token-string table.
        JsonElement model = root.GetProperty("model");
        JsonElement vocab = model.GetProperty("vocab");
        int maxId = -1;
        // Walk to find max ID for sizing.
        foreach (JsonProperty kv in vocab.EnumerateObject())
            if (kv.Value.GetInt32() > maxId) maxId = kv.Value.GetInt32();
        if (root.TryGetProperty("added_tokens", out JsonElement added) && added.ValueKind == JsonValueKind.Array)
            foreach (JsonElement at in added.EnumerateArray())
                if (at.GetProperty("id").GetInt32() > maxId) maxId = at.GetProperty("id").GetInt32();

        _vocabSize = maxId + 1;
        _idToToken = new string?[_vocabSize];
        _isByteFallback = new bool[_vocabSize];
        _byteFallbackValue = new byte[_vocabSize];

        foreach (JsonProperty kv in vocab.EnumerateObject())
        {
            int id = kv.Value.GetInt32();
            string tok = kv.Name;
            _idToToken[id] = tok;
            // Byte-fallback tokens are formatted as "<0xHH>". Detect and pre-decode.
            if (tok.Length == 6 && tok[0] == '<' && tok[1] == '0' && tok[2] == 'x' && tok[5] == '>')
            {
                if (byte.TryParse(tok.AsSpan(3, 2), System.Globalization.NumberStyles.HexNumber, null, out byte b))
                {
                    _isByteFallback[id] = true;
                    _byteFallbackValue[id] = b;
                }
            }
        }
        if (root.TryGetProperty("added_tokens", out JsonElement added2) && added2.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement at in added2.EnumerateArray())
            {
                int id = at.GetProperty("id").GetInt32();
                string content = at.GetProperty("content").GetString() ?? string.Empty;
                _idToToken[id] = content;
            }
        }
    }

    /// <summary>Drops special tokens (<c>&lt;s&gt;</c>, <c>&lt;/s&gt;</c>, <c>&lt;unk&gt;</c>, the <c>&lt;&lt;ST_x&gt;&gt;</c> reserved range) unless <paramref name="includeSpecial"/> is true.</summary>
    public string Decode(ReadOnlySpan<int> tokenIds, bool includeSpecial = false)
    {
        ThrowIfDisposed();
        StringBuilder sb = new(tokenIds.Length * 4);
        List<byte> byteRun = new(8);

        foreach (int id in tokenIds)
        {
            if ((uint)id >= (uint)_vocabSize) continue;
            string? tok = _idToToken[id];
            if (tok is null) continue;

            // Accumulate consecutive byte-fallback tokens, then flush as UTF-8.
            if (_isByteFallback[id])
            {
                byteRun.Add(_byteFallbackValue[id]);
                continue;
            }
            if (byteRun.Count > 0)
            {
                sb.Append(Encoding.UTF8.GetString(byteRun.ToArray()));
                byteRun.Clear();
            }

            if (IsSpecial(id, tok))
            {
                if (includeSpecial) sb.Append(tok);
                continue;
            }

            // Metaspace decode: U+2581 → ASCII space.
            for (int i = 0; i < tok.Length; i++)
            {
                char c = tok[i];
                sb.Append(c == '▁' ? ' ' : c);
            }
        }

        // Flush any trailing byte-fallback run.
        if (byteRun.Count > 0) sb.Append(Encoding.UTF8.GetString(byteRun.ToArray()));

        return sb.ToString();
    }

    /// <summary>Returns the text for a single token ID, for streaming code that emits per-token output; byte-fallback singletons return the bytes as a raw Latin-1 string — the caller must buffer until a complete UTF-8 character lands.</summary>
    public string DecodeOne(int id)
    {
        ThrowIfDisposed();
        if ((uint)id >= (uint)_vocabSize) return string.Empty;
        string? tok = _idToToken[id];
        if (tok is null) return string.Empty;
        if (_isByteFallback[id]) return Encoding.Latin1.GetString([_byteFallbackValue[id]]);
        StringBuilder sb = new(tok.Length);
        foreach (char c in tok) sb.Append(c == '▁' ? ' ' : c);
        return sb.ToString();
    }

    private static bool IsSpecial(int id, string tok)
    {
        // 0=<unk>, 1=<s>, 2=</s>.
        if (id <= 2) return true;
        // <<ST_x>> reserved range (32000..32767 for vocab_size=32768).
        if (tok.StartsWith("<<", StringComparison.Ordinal) && tok.EndsWith(">>", StringComparison.Ordinal))
            return true;
        return false;
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0) throw new ObjectDisposedException(nameof(MoonshineTokenizer));
    }

    public void Dispose() { Interlocked.Exchange(ref _disposed, 1); }
}
