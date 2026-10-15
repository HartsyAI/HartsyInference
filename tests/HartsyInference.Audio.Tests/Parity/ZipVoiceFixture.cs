using System.Text.Json;
using HartsyInference.Audio.Models.ZipVoice;
using HartsyInference.Audio.Phonemizer;

namespace HartsyInference.Audio.Tests.Parity;

/// <summary>The upstream dump in <c>Fixtures/zipvoice_frontend_parity.json</c> and the stand-ins the ZipVoice frontend tests share.</summary>
internal static class ZipVoiceFixture
{
    private static readonly Lazy<JsonDocument> Document = new(() => JsonDocument.Parse(File.ReadAllText(
        Path.Combine(AppContext.BaseDirectory, "Fixtures", "zipvoice_frontend_parity.json"))));

    internal static JsonElement Root => Document.Value.RootElement;

    /// <summary>A tokenizer whose vocabulary holds only the pad symbol; <see cref="ZipVoiceTokenizer.Tokenize"/> needs no more.</summary>
    internal static ZipVoiceTokenizer PadOnlyTokenizer()
    {
        string path = Path.Combine(Path.GetTempPath(), $"zipvoice_tokens_{Guid.NewGuid():N}.txt");
        File.WriteAllText(path, "_\t0\n");
        try
        {
            return ZipVoiceTokenizer.FromFile(path);
        }
        finally
        {
            File.Delete(path);
        }
    }

    /// <summary>Phonemizes each word from a lexicon and joins them with spaces, so only the clause structure is under test.</summary>
    internal sealed class LexiconPhonemizer(IReadOnlyDictionary<string, string> lexicon) : IPhonemizer
    {
        public string PhonemizeToIpa(string text, string language)
        {
            List<string> words = new();
            foreach (string word in text.Split(' ', StringSplitOptions.RemoveEmptyEntries))
            {
                string letters = new(word.Where(char.IsLetter).ToArray());
                if (letters.Length > 0) words.Add(lexicon[letters.ToLowerInvariant()]);
            }
            return string.Join(' ', words);
        }

        public IReadOnlyList<string> PhonemizeToMnemonics(string text, string language) => throw new NotSupportedException();

        public int[] PhonemizeToIds(string text, string language, PhonemeIdMap idMap) => throw new NotSupportedException();
    }
}
