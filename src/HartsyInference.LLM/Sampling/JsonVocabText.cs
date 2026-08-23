using System.Runtime.CompilerServices;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Sampling;

/// <summary>Per-tokenizer cache of each vocab id's standalone decoded text, shared by every JSON-grammar sampler step (<see cref="JsonGrammarStep"/>, <see cref="SentinelJsonGrammarStep"/>) so the O(vocab size) decode pass happens once per loaded model, not once per request.</summary>
internal static class JsonVocabText
{
    private static readonly ConditionalWeakTable<ILlmTokenizer, string[]> Cache = new();

    public static string[] GetOrBuild(ILlmTokenizer tokenizer, int vocabSize) =>
        Cache.GetValue(tokenizer, t => Build(t, vocabSize));

    /// <summary>Bounds-safe lookup: ids past the end of <paramref name="table"/> (a logits span wider than the decoded vocab) read as empty text, which the grammar states treat as a no-op rather than a parse failure.</summary>
    public static string TokenText(string[] table, int id) => (uint)id < (uint)table.Length ? table[id] : string.Empty;

    private static string[] Build(ILlmTokenizer tokenizer, int vocabSize)
    {
        string[] table = new string[vocabSize];
        for (int i = 0; i < vocabSize; i++)
        {
            try { table[i] = tokenizer.Decode([i]); }
            catch { table[i] = string.Empty; } // some ids (partial byte-fallback pieces etc.) may not decode standalone — treat conservatively as empty rather than throwing
        }
        return table;
    }
}
