using System.Text.Json;
using System.Text.RegularExpressions;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.ModelAssets.Tokenizers.Tests;

/// <summary>Split semantics of <see cref="PreTokenizerPipeline"/> checked against HuggingFace <c>tokenizers</c> 0.23.2 pre_tokenize_str output; the expectations come from tests/python-reference/deepseek_v41/dump_encoder_reference.py and inline HF dumps.</summary>
public sealed class PreTokenizerPipelineTests
{
    private static string ReferencePath => Path.Combine(
        RepoRoot.Path, "tests", "python-reference", "deepseek_v41", "encoder_reference", "encoder_reference.json");

    private static PreTokenizerPipeline Build(string json)
    {
        using JsonDocument doc = JsonDocument.Parse(json);
        return PreTokenizerPipeline.FromJson(doc.RootElement.Clone());
    }

    private static string SplitJson(string behavior, bool invert, string regex) =>
        JsonSerializer.Serialize(new
        {
            type = "Sequence",
            pretokenizers = new object[]
            {
                new { type = "Split", pattern = new { Regex = regex }, behavior, invert },
            },
        });

    [Theory]
    [InlineData("Removed", false, "\\d", "a1b22c", new string[] { "a", "b", "c" })]
    [InlineData("Removed", false, "\\d+", "12ab345", new string[] { "ab" })]
    [InlineData("Removed", false, " ", "a b  c ", new string[] { "a", "b", "c" })]
    [InlineData("Removed", false, "\\d+", "", new string[] { })]
    [InlineData("Removed", true, "\\d", "a1b22c", new string[] { "1", "2", "2" })]
    [InlineData("Removed", true, "\\d+", "12ab345", new string[] { "12", "345" })]
    [InlineData("Removed", true, " ", "a b  c ", new string[] { " ", " ", " ", " " })]
    [InlineData("Removed", true, "\\d+", "", new string[] { })]
    [InlineData("Isolated", false, "\\d", "a1b22c", new string[] { "a", "1", "b", "2", "2", "c" })]
    [InlineData("Isolated", false, "\\d+", "12ab345", new string[] { "12", "ab", "345" })]
    [InlineData("Isolated", false, " ", "a b  c ", new string[] { "a", " ", "b", " ", " ", "c", " " })]
    [InlineData("Isolated", false, "\\d+", "", new string[] { })]
    [InlineData("Isolated", true, "\\d", "a1b22c", new string[] { "a", "1", "b", "2", "2", "c" })]
    [InlineData("Isolated", true, "\\d+", "12ab345", new string[] { "12", "ab", "345" })]
    [InlineData("Isolated", true, " ", "a b  c ", new string[] { "a", " ", "b", " ", " ", "c", " " })]
    [InlineData("Isolated", true, "\\d+", "", new string[] { })]
    [InlineData("MergedWithPrevious", false, "\\d", "a1b22c", new string[] { "a1", "b2", "2", "c" })]
    [InlineData("MergedWithPrevious", false, "\\d+", "12ab345", new string[] { "12", "ab345" })]
    [InlineData("MergedWithPrevious", false, " ", "a b  c ", new string[] { "a ", "b ", " ", "c " })]
    [InlineData("MergedWithPrevious", false, "\\d+", "", new string[] { })]
    [InlineData("MergedWithPrevious", true, "\\d", "a1b22c", new string[] { "a", "1b", "2", "2c" })]
    [InlineData("MergedWithPrevious", true, "\\d+", "12ab345", new string[] { "12ab", "345" })]
    [InlineData("MergedWithPrevious", true, " ", "a b  c ", new string[] { "a", " b", " ", " c", " " })]
    [InlineData("MergedWithPrevious", true, "\\d+", "", new string[] { })]
    [InlineData("MergedWithNext", false, "\\d", "a1b22c", new string[] { "a", "1b", "2", "2c" })]
    [InlineData("MergedWithNext", false, "\\d+", "12ab345", new string[] { "12ab", "345" })]
    [InlineData("MergedWithNext", false, " ", "a b  c ", new string[] { "a", " b", " ", " c", " " })]
    [InlineData("MergedWithNext", false, "\\d+", "", new string[] { })]
    [InlineData("MergedWithNext", true, "\\d", "a1b22c", new string[] { "a1", "b2", "2", "c" })]
    [InlineData("MergedWithNext", true, "\\d+", "12ab345", new string[] { "12", "ab345" })]
    [InlineData("MergedWithNext", true, " ", "a b  c ", new string[] { "a ", "b ", " ", "c " })]
    [InlineData("MergedWithNext", true, "\\d+", "", new string[] { })]
    [InlineData("Contiguous", false, "\\d", "a1b22c", new string[] { "a", "1", "b", "22", "c" })]
    [InlineData("Contiguous", false, "\\d+", "12ab345", new string[] { "12", "ab", "345" })]
    [InlineData("Contiguous", false, " ", "a b  c ", new string[] { "a", " ", "b", "  ", "c", " " })]
    [InlineData("Contiguous", false, "\\d+", "", new string[] { })]
    [InlineData("Contiguous", true, "\\d", "a1b22c", new string[] { "a", "1", "b", "22", "c" })]
    [InlineData("Contiguous", true, "\\d+", "12ab345", new string[] { "12", "ab", "345" })]
    [InlineData("Contiguous", true, " ", "a b  c ", new string[] { "a", " ", "b", "  ", "c", " " })]
    [InlineData("Contiguous", true, "\\d+", "", new string[] { })]
    public void SingleSplit_MatchesHuggingFace(string behavior, bool invert, string regex, string text, string[] expected)
    {
        PreTokenizerPipeline pipeline = Build(SplitJson(behavior, invert, regex));
        Assert.Equal(expected, pipeline.Split(text));
    }

    [Fact]
    public void Sequence_AppliesStagesInOrder()
    {
        const string json = """
            {"type":"Sequence","pretokenizers":[
              {"type":"Split","pattern":{"Regex":"\\d{1,3}"},"behavior":"Isolated","invert":false},
              {"type":"Split","pattern":{"Regex":"[a-z]+"},"behavior":"Removed","invert":false}]}
            """;
        Assert.Equal(new[] { "123", "45", "67", " " }, Build(json).Split("ab12345cd67 ef"));
    }

    [Fact]
    public void MissingPreTokenizer_UsesGpt2Split()
    {
        Assert.Same(PreTokenizerPipeline.Gpt2, PreTokenizerPipeline.FromJson(null));
        Assert.Equal(new[] { "Hello", " world", "'s", " 123", "!" }, PreTokenizerPipeline.Gpt2.Split("Hello world's 123!"));
    }

    [Fact]
    public void ByteLevelWithoutRegex_AddsNoSplit()
    {
        const string json = """{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}""";
        Assert.Equal(new[] { "a b  c" }, Build(json).Split("a b  c"));
    }

    [Fact]
    public void UnsupportedStage_Throws()
    {
        Assert.Throws<NotSupportedException>(() => Build("""{"type":"Whitespace"}"""));
        Assert.Throws<NotSupportedException>(() => Build(
            """{"type":"Split","pattern":{"String":" "},"behavior":"Isolated","invert":false}"""));
    }

    [Fact]
    public void DeepSeekV41Stages_MatchHuggingFaceOnStressCorpus()
    {
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(ReferencePath));
        PreTokenizerPipeline pipeline = PreTokenizerPipeline.FromJson(doc.RootElement.GetProperty("pre_tokenizer").Clone());
        int checkedCount = 0;
        foreach (JsonElement entry in doc.RootElement.GetProperty("stress").EnumerateArray())
        {
            string text = entry.GetProperty("text").GetString()!;
            string[] expected = entry.GetProperty("pieces").EnumerateArray().Select(p => p.GetString()!).ToArray();
            string[] actual = pipeline.Split(text).Select(ByteLevelCodec.Encode).ToArray();
            Assert.True(expected.SequenceEqual(actual), $"pre-tokenization differs for text #{checkedCount}: {text}");
            checkedCount++;
        }
        Assert.True(checkedCount > 50);
    }

    [Fact]
    public void EmbeddedLlama3AndQwen3_MatchLegacyFirstSplitRegex()
    {
        using JsonDocument reference = JsonDocument.Parse(File.ReadAllText(ReferencePath));
        List<string> corpus = reference.RootElement.GetProperty("stress").EnumerateArray()
            .Select(e => e.GetProperty("text").GetString()!)
            .Where(t => !t.Any(char.IsSurrogate))
            .ToList();
        corpus.Add("It's 12345 o'clock, isn't it? IT'S \r\n\r\n  x\t\ty   ");

        int compared = 0;
        foreach ((bool present, Func<Stream> open) in new (bool, Func<Stream>)[]
        {
            (EmbeddedTokenizerResources.HasLlama3TokenizerJson, EmbeddedTokenizerResources.OpenLlama3TokenizerJson),
            (EmbeddedTokenizerResources.HasQwen3TokenizerJson, EmbeddedTokenizerResources.OpenQwen3TokenizerJson),
        })
        {
            if (!present) continue;
            using Stream stream = open();
            using JsonDocument doc = JsonDocument.Parse(stream);
            JsonElement preTokenizer = doc.RootElement.GetProperty("pre_tokenizer");
            string pattern = preTokenizer.GetProperty("pretokenizers")[0].GetProperty("pattern").GetProperty("Regex").GetString()!;
            Regex legacy = new(pattern, RegexOptions.CultureInvariant);
            PreTokenizerPipeline pipeline = PreTokenizerPipeline.FromJson(preTokenizer.Clone());
            foreach (string text in corpus)
            {
                string[] expected = legacy.Matches(text).Select(m => m.Value).ToArray();
                Assert.True(expected.SequenceEqual(pipeline.Split(text)), $"pipeline differs from legacy regex: {text}");
                compared++;
            }
        }
        Assert.True(compared > 0 || !EmbeddedTokenizerResources.HasLlama3TokenizerJson);
    }
}
