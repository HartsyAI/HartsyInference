using HartsyInference.Audio.Frontends.Espeak;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Structural parser tests for the compiled espeak-ng dictionary reader. Gated on a real
/// <c>en_dict</c> being present (set <c>ESPEAK_DATA_DIR</c> to an <c>espeak-ng-data</c> directory). Validates that
/// <see cref="EspeakDictFile"/> walks the entire word-list hash table and the rule-group section of a genuine
/// v1.50 dictionary without error and produces sane indices.</summary>
public sealed class EspeakDictFileTests
{
    private static string? DictPath()
    {
        string? dir = Environment.GetEnvironmentVariable("ESPEAK_DATA_DIR");
        if (string.IsNullOrEmpty(dir)) return null;
        string path = Path.Combine(dir, "en_dict");
        return File.Exists(path) ? path : null;
    }

    [Fact]
    public void ParsesRealEnglishDictionary()
    {
        string? path = DictPath();
        if (path is null) return; // gated: no real dictionary available

        EspeakDictFile dict = EspeakDictFile.Load(path);

        Assert.Equal(1024, dict.HashStart.Length);
        Assert.True(dict.RulesOffset > 0 && dict.RulesOffset < dict.Data.Length);

        // Every hash bucket start must land inside the word-list region (before the rules section).
        foreach (int start in dict.HashStart)
            Assert.InRange(start, 8, dict.RulesOffset);

        // English exercises single-letter rule groups for the whole alphabet.
        for (char c = 'a'; c <= 'z'; c++)
            Assert.True(dict.Groups1[c] > 0, $"missing single-letter rule group for '{c}'");

        // Two-letter groups exist (e.g. 'ch', 'th', 'sh' digraphs).
        Assert.True(dict.NumGroups2 > 0);
        Assert.Equal(dict.NumGroups2, dict.Groups2Name.Length);

        // Every rule-group offset points into the rules section.
        foreach (int g in dict.Groups1)
            if (g >= 0) Assert.InRange(g, dict.RulesOffset, dict.Data.Length - 1);
    }
}
