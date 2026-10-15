using System.Text.Json;

namespace HartsyInference.ModelAssets.Tokenizers;

/// <summary>Builds a Hugging Face-exact Qwen byte-level BPE (<see cref="GgufTokenizer"/>) from a checkpoint's own
/// <c>vocab.json</c> and <c>merges.txt</c>, with Qwen's pre-tokenizer split regex. Use it when a model ships a Qwen
/// vocabulary that differs from the embedded Qwen3 one (CosyVoice 2's CosyVoice-BlankEN, for example, has different
/// CJK merges), so <c>tokenizer.json</c> is not available but the split rules are Qwen's.</summary>
public static class QwenBpe
{
    /// <summary>Qwen2/Qwen2.5/Qwen3 pre-tokenizer split pattern (<c>tokenizer.json</c> <c>pre_tokenizer.Split</c>); the
    /// engine's single copy lives on <see cref="GgufTokenizer.Qwen2PreTokenRegex"/>.</summary>
    public const string SplitPattern = GgufTokenizer.Qwen2PreTokenRegex;

    /// <summary>Loads the BPE from a checkpoint's <c>vocab.json</c> and <c>merges.txt</c>. Ids missing from <c>vocab.json</c>
    /// (added tokens) get <c>&lt;|unused_N|&gt;</c> placeholder names that no merge produces; callers map their own
    /// added-token literals before encoding.</summary>
    public static GgufTokenizer FromFiles(string vocabJsonPath, string mergesPath)
    {
        using Stream vocab = File.OpenRead(vocabJsonPath);
        using Stream merges = File.OpenRead(mergesPath);
        return FromStreams(vocab, merges);
    }

    /// <summary>Loads the BPE from <c>vocab.json</c> and <c>merges.txt</c> streams, with the same contract as
    /// <see cref="FromFiles"/>. The result has no added-token table, so a literal such as <c>&lt;think&gt;</c> encodes as
    /// ordinary characters. The caller keeps ownership of both streams.</summary>
    public static GgufTokenizer FromStreams(Stream vocabJson, Stream mergesTxt)
    {
        Dictionary<string, int> vocab = new(StringComparer.Ordinal);
        using (JsonDocument doc = JsonDocument.Parse(vocabJson))
        {
            if (doc.RootElement.ValueKind != JsonValueKind.Object)
                throw new InvalidDataException("vocab.json is not a token-to-id object");
            foreach (JsonProperty entry in doc.RootElement.EnumerateObject())
            {
                int id = entry.Value.GetInt32();
                if (id < 0) throw new InvalidDataException($"vocab.json has negative id {id} for '{entry.Name}'");
                vocab[entry.Name] = id;
            }
        }
        if (vocab.Count == 0) throw new InvalidDataException("vocab.json is empty");
        int size = 0;
        foreach (int id in vocab.Values) if (id + 1 > size) size = id + 1;
        string[] tokens = new string[size];
        for (int i = 0; i < size; i++) tokens[i] = $"<|unused_{i}|>";
        foreach (KeyValuePair<string, int> kv in vocab) tokens[kv.Value] = kv.Key;

        List<string> merges = [];
        using (StreamReader reader = new(mergesTxt, leaveOpen: true))
        {
            string? line;
            while ((line = reader.ReadLine()) is not null)
            {
                if (line.Length == 0 || line.StartsWith("#version", StringComparison.Ordinal)) continue;
                merges.Add(line);
            }
        }
        return new GgufTokenizer(tokens, [.. merges], tokenType: null, bosId: null, eosId: null, preTokenizerRegex: SplitPattern);
    }
}
