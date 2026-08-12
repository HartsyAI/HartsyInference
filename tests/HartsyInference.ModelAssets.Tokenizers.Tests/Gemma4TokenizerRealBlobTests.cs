using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ModelAssets.Tokenizers.Tests;

/// <summary>Bit-exact cross-check of <see cref="Gemma4Tokenizer"/> against the HuggingFace <c>tokenizers</c>
/// library on the <b>real</b> 262k-vocab Gemma 4 tokenizer. The blob is the 32 MB <c>tokenizer_json</c> tensor
/// inside the LTX-2.5 text encoder and is not committed, so this skips unless the reference directory produced by
/// <c>tests/python-reference/dump_gemma4_tokenizer_reference.py</c> is present.</summary>
[Trait("Category", "Integration")]
public sealed class Gemma4TokenizerRealBlobTests
{
    private static string ReferenceDir =>
        Environment.GetEnvironmentVariable("GEMMA4_TOKENIZER_REFERENCE_DIR")
        ?? Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "python-reference", "gemma4_tokenizer_reference");

    private readonly ITestOutputHelper _output;
    public Gemma4TokenizerRealBlobTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void RealTokenizer_MatchesHuggingFaceIds()
    {
        string tokenizerPath = Path.Combine(ReferenceDir, "tokenizer.json");
        string expectedPath = Path.Combine(ReferenceDir, "expected.json");
        if (!File.Exists(tokenizerPath) || !File.Exists(expectedPath))
        {
            _output.WriteLine($"SKIPPED: Gemma 4 tokenizer reference not found at {ReferenceDir}.");
            _output.WriteLine("Generate it with: python tests/python-reference/dump_gemma4_tokenizer_reference.py --tokenizer-json <path>");
            return;
        }

        Expected expected = JsonSerializer.Deserialize<Expected>(File.ReadAllText(expectedPath))
            ?? throw new InvalidDataException("Gemma 4 tokenizer expected.json malformed.");

        using FileStream stream = File.OpenRead(tokenizerPath);
        Gemma4Tokenizer tokenizer = Gemma4Tokenizer.FromTokenizerJson(stream);
        Assert.Equal(262144, tokenizer.VocabSize);

        foreach (Case testCase in expected.Cases)
        {
            int[] actual = tokenizer.Encode(testCase.Text);
            Assert.Equal(testCase.Ids, actual);
            Assert.Equal(testCase.Text, tokenizer.Decode(actual));
        }
        _output.WriteLine($"{expected.Cases.Count} prompts matched HuggingFace ids exactly.");
    }

    private sealed record Expected
    {
        [JsonPropertyName("cases")] public List<Case> Cases { get; init; } = [];
    }

    private sealed record Case
    {
        [JsonPropertyName("text")] public string Text { get; init; } = "";
        [JsonPropertyName("ids")] public int[] Ids { get; init; } = [];
    }
}
