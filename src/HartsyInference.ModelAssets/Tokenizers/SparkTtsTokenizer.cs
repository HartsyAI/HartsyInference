using System.Text.Json;
using Microsoft.ML.Tokenizers;

namespace HartsyInference.ModelAssets.Tokenizers;

/// <summary>Spark-TTS prompt tokenizer. Spark's LM is a Qwen2.5-0.5B whose base byte-level BPE vocab
/// (151,643 entries shipped as the checkpoint's <c>vocab.json</c> + <c>merges.txt</c>) is extended with
/// ~13.5k atomic control/audio tokens listed in <c>added_tokens.json</c> (<c>&lt;|task_tts|&gt;</c>,
/// <c>&lt;|bicodec_global_0|&gt;</c>, <c>&lt;|gender_0|&gt;</c>, …). This class composes the shared
/// <see cref="BpeTokenizer"/> + <see cref="ByteLevelCodec"/> (the same byte-level path validated for
/// Qwen3) and only adds the Spark-specific pieces: the added-token map and the <c>&lt;|…|&gt;</c> split.
/// Added tokens are matched atomically (never BPE-merged); the text between them is byte-level BPE'd.</summary>
public sealed class SparkTtsTokenizer
{
    private readonly Tokenizer _bpe;
    private readonly IReadOnlyDictionary<string, int> _added;

    /// <summary>Builds from the Spark <c>LLM/</c> directory files: <c>vocab.json</c>, <c>merges.txt</c>,
    /// <c>added_tokens.json</c> (a <c>{ "&lt;|token|&gt;": id }</c> map).</summary>
    public SparkTtsTokenizer(string vocabPath, string mergesPath, string addedTokensPath)
    {
        using FileStream vocab = File.OpenRead(vocabPath);
        using FileStream merges = File.OpenRead(mergesPath);
        _bpe = BpeTokenizer.Create(vocab, merges);

        Dictionary<string, int> added = new(StringComparer.Ordinal);
        using (FileStream s = File.OpenRead(addedTokensPath))
        using (JsonDocument doc = JsonDocument.Parse(s))
            foreach (JsonProperty p in doc.RootElement.EnumerateObject()) added[p.Name] = p.Value.GetInt32();
        _added = added;
    }

    /// <summary>Convenience: load from a Spark <c>LLM/</c> directory.</summary>
    public static SparkTtsTokenizer FromDirectory(string llmDir) => new(
        Path.Combine(llmDir, "vocab.json"), Path.Combine(llmDir, "merges.txt"),
        Path.Combine(llmDir, "added_tokens.json"));

    /// <summary>Encodes a prompt that may interleave plain text with atomic <c>&lt;|…|&gt;</c> added tokens.
    /// Added tokens map to their ids; runs of plain text are GPT-2 byte-level BPE'd (so leading spaces map to
    /// <c>Ġ</c> exactly, via <see cref="ByteLevelCodec"/>).</summary>
    public int[] Encode(string text)
    {
        List<int> ids = new(text.Length / 2 + 8);
        int i = 0, segStart = 0;
        while (i < text.Length)
        {
            if (text[i] == '<' && i + 1 < text.Length && text[i + 1] == '|')
            {
                int close = text.IndexOf("|>", i + 2, StringComparison.Ordinal);
                if (close >= 0 && _added.TryGetValue(text[i..(close + 2)], out int id))
                {
                    if (i > segStart) EncodeBase(ids, text[segStart..i]);
                    ids.Add(id);
                    i = close + 2;
                    segStart = i;
                    continue;
                }
            }
            i++;
        }
        if (segStart < text.Length) EncodeBase(ids, text[segStart..]);
        return [.. ids];
    }

    private void EncodeBase(List<int> dst, string segment)
    {
        foreach (int id in _bpe.EncodeToIds(ByteLevelCodec.Encode(segment))) dst.Add(id);
    }
}
