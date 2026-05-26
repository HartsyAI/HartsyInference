using System.Text.Json;

namespace SharpInference.Audio.Models.Kokoro;

/// <summary>Loads and applies the Kokoro phoneme vocab. Kokoro takes IPA-phoneme strings
/// (the upstream pipeline uses <c>misaki</c> as a G2P front-end to turn English text into
/// IPA), so this tokenizer is a simple symbol-to-int map. The vocab lives directly inside
/// the model's <c>config.json</c> under the <c>"vocab"</c> key as <c>{symbol: id}</c>.
///
/// <para>The vocab is sparse: ~114 actual symbols spread across IDs <c>[1, 177]</c>, with
/// gaps where symbols were planned but not used. ID 0 is reserved for BOS/EOS padding —
/// every encoded sequence is wrapped <c>[0, t₁, ..., tₙ, 0]</c> before being fed to PLBERT.</para>
///
/// <para>Unknown symbols are skipped silently (rather than mapping to an UNK id) because
/// the misaki G2P should never produce a symbol outside the vocab; if it does, the safe
/// fallback is to drop it. A diagnostic logger could be added later.</para></summary>
public sealed class KokoroPhonemeTokenizer
{
    /// <summary>BOS/EOS padding ID, prepended and appended to every encoded sequence.</summary>
    public const int PadId = 0;

    private readonly Dictionary<string, int> _vocab;
    private readonly int _maxId;

    /// <summary>Total number of vocab IDs (one past the maximum ID, including gaps).
    /// Matches <see cref="KokoroConfig.NumTokens"/> when loaded from a v1 config.</summary>
    public int VocabSize => _maxId + 1;

    private KokoroPhonemeTokenizer(Dictionary<string, int> vocab, int maxId)
    {
        _vocab = vocab;
        _maxId = maxId;
    }

    /// <summary>Loads the vocab from <paramref name="configJsonPath"/> (Kokoro's
    /// <c>config.json</c>). Reads the <c>"vocab"</c> object as a flat
    /// <c>{symbol: id}</c> map.</summary>
    public static KokoroPhonemeTokenizer LoadFromConfig(string configJsonPath)
    {
        using FileStream fs = File.OpenRead(configJsonPath);
        using JsonDocument doc = JsonDocument.Parse(fs);
        if (!doc.RootElement.TryGetProperty("vocab", out JsonElement vocabEl))
            throw new InvalidOperationException($"Kokoro config '{configJsonPath}' missing 'vocab' key.");

        Dictionary<string, int> vocab = new(StringComparer.Ordinal);
        int maxId = 0;
        foreach (JsonProperty p in vocabEl.EnumerateObject())
        {
            int id = p.Value.GetInt32();
            vocab[p.Name] = id;
            if (id > maxId) maxId = id;
        }
        return new KokoroPhonemeTokenizer(vocab, maxId);
    }

    /// <summary>Encodes <paramref name="phonemes"/> into Kokoro token IDs. Each Unicode
    /// code point is looked up directly — Kokoro's vocab is per-codepoint, not per-byte
    /// or per-grapheme, so a multi-byte symbol like ʃ is one ID. Unknown symbols are
    /// silently dropped.
    ///
    /// <para>BOS/EOS padding <see cref="PadId"/> is automatically prepended and appended,
    /// so the returned array length is <c>(visible symbol count) + 2</c>.</para></summary>
    public int[] Encode(string phonemes)
    {
        List<int> ids = new(phonemes.Length + 2) { PadId };
        // Iterate by code point so that multi-byte IPA symbols (each a single rune in
        // the vocab) map to a single ID. .NET strings are UTF-16, so a non-BMP code
        // point would take two chars — Kokoro's vocab is entirely BMP so iterating by
        // System.Text.Rune still works in a single pass.
        for (int i = 0; i < phonemes.Length;)
        {
            int cp = char.ConvertToUtf32(phonemes, i);
            string sym = char.ConvertFromUtf32(cp);
            if (_vocab.TryGetValue(sym, out int id)) ids.Add(id);
            i += char.IsSurrogatePair(phonemes, i) ? 2 : 1;
        }
        ids.Add(PadId);
        return ids.ToArray();
    }
}
