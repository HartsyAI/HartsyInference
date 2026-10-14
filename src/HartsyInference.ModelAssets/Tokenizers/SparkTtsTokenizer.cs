using System.Text;
using System.Text.Json;

namespace HartsyInference.ModelAssets.Tokenizers;

/// <summary>Spark-TTS prompt tokenizer, matching the HF <c>Qwen2Tokenizer</c> Spark loads: NFC normalization, the Qwen2
/// split regex, byte-level BPE over the checkpoint's <c>vocab.json</c> + <c>merges.txt</c> (151,643 entries), and the
/// ~13.5k control/audio tokens of <c>added_tokens.json</c> (<c>&lt;|task_tts|&gt;</c>, <c>&lt;|bicodec_global_0|&gt;</c>,
/// …) matched atomically before BPE. Built on the shared <see cref="GgufTokenizer"/> core.</summary>
public sealed class SparkTtsTokenizer
{
    private const int TypeControl = 3;

    private readonly GgufTokenizer _tokenizer;

    /// <summary>Builds from the Spark <c>LLM/</c> directory files: <c>vocab.json</c>, <c>merges.txt</c>, <c>added_tokens.json</c> (a <c>{ "&lt;|token|&gt;": id }</c> map).</summary>
    public SparkTtsTokenizer(string vocabPath, string mergesPath, string addedTokensPath)
    {
        Dictionary<int, string> byId = new(180_000);
        HashSet<int> added = [];
        ReadTokenMap(vocabPath, byId, null);
        ReadTokenMap(addedTokensPath, byId, added);
        int maxId = byId.Keys.Max();
        string[] tokens = new string[maxId + 1];
        int[] tokenType = new int[maxId + 1];
        for (int i = 0; i <= maxId; i++)
        {
            if (!byId.TryGetValue(i, out string? token))
                throw new InvalidDataException($"Spark tokenizer has no token with id {i}.");
            tokens[i] = token;
            tokenType[i] = added.Contains(i) ? TypeControl : 1;
        }
        string[] merges = File.ReadLines(mergesPath)
            .Where(l => l.Length > 0 && !l.StartsWith("#version", StringComparison.Ordinal)).ToArray();
        _tokenizer = new GgufTokenizer(tokens, merges, tokenType, bosId: null, eosId: null,
            preTokenizerRegex: GgufTokenizer.Qwen2PreTokenRegex);
    }

    /// <summary>Convenience: load from a Spark <c>LLM/</c> directory.</summary>
    public static SparkTtsTokenizer FromDirectory(string llmDir) => new(
        Path.Combine(llmDir, "vocab.json"), Path.Combine(llmDir, "merges.txt"),
        Path.Combine(llmDir, "added_tokens.json"));

    /// <summary>Encodes a prompt that may interleave plain text with atomic added-token literals, as HF
    /// <c>tokenizer(prompt)</c> does for Spark (no BOS/EOS is added).</summary>
    public int[] Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        string normalized = text.IsNormalized(NormalizationForm.FormC) ? text : text.Normalize(NormalizationForm.FormC);
        return _tokenizer.Encode(normalized, addSpecial: true);
    }

    private static void ReadTokenMap(string path, Dictionary<int, string> byId, HashSet<int>? ids)
    {
        using FileStream s = File.OpenRead(path);
        using JsonDocument doc = JsonDocument.Parse(s);
        foreach (JsonProperty p in doc.RootElement.EnumerateObject())
        {
            int id = p.Value.GetInt32();
            if (!byId.TryAdd(id, p.Name))
                throw new InvalidDataException($"Spark tokenizer id {id} is assigned twice ('{byId[id]}', '{p.Name}').");
            ids?.Add(id);
        }
    }
}
