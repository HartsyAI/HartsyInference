using System.Text.Json;

namespace HartsyInference.ModelAssets.Tokenizers;

/// <summary>Builds a Hugging Face-exact Qwen byte-level BPE (<see cref="GgufTokenizer"/>) from a checkpoint's own
/// <c>vocab.json</c> and <c>merges.txt</c>, with Qwen's pre-tokenizer split regex. Use it when a model ships a Qwen
/// vocabulary that differs from the embedded Qwen3 one (CosyVoice 2's CosyVoice-BlankEN, for example, has different
/// CJK merges), so <c>tokenizer.json</c> is not available but the split rules are Qwen's.</summary>
public static class QwenBpe
{
    /// <summary>Qwen2/Qwen2.5/Qwen3 pre-tokenizer split pattern (<c>tokenizer.json</c> <c>pre_tokenizer.Split</c>).</summary>
    public const string SplitPattern =
        @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    /// <summary>Loads the BPE. Ids missing from <c>vocab.json</c> (added tokens) are left as empty placeholders; callers
    /// map their own added-token literals before encoding.</summary>
    public static GgufTokenizer FromFiles(string vocabJsonPath, string mergesPath)
    {
        Dictionary<string, int> vocab;
        using (Stream s = File.OpenRead(vocabJsonPath))
            vocab = JsonSerializer.Deserialize<Dictionary<string, int>>(s)
                ?? throw new InvalidDataException($"Empty vocabulary: {vocabJsonPath}");
        int size = 0;
        foreach (int id in vocab.Values) if (id + 1 > size) size = id + 1;
        string[] tokens = new string[size];
        for (int i = 0; i < size; i++) tokens[i] = $"<|unused_{i}|>";
        foreach (KeyValuePair<string, int> kv in vocab) tokens[kv.Value] = kv.Key;

        List<string> merges = [];
        foreach (string line in File.ReadLines(mergesPath))
        {
            if (line.Length == 0 || line.StartsWith("#version", StringComparison.Ordinal)) continue;
            merges.Add(line);
        }
        return new GgufTokenizer(tokens, [.. merges], tokenType: null, bosId: null, eosId: null, preTokenizerRegex: SplitPattern);
    }
}
