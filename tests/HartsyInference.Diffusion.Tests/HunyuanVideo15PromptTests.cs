using System.Text.Json;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The HunyuanVideo-1.5 / MovieRoll prompt helpers against strings dumped from the reference functions
/// (<c>tests/python-reference/movieroll_reference/dump_hy15_text.py</c>, MovieRoll 0b5f73e): the video system prompt byte for byte, and the glyph
/// text extraction and formatting. Token ids against the HuggingFace Qwen2.5-VL tokenizer are not covered: its tokenizer files have not been
/// fetched, so <c>Tokenize</c> is pinned structurally and through the engine tokenizer's own chat template.</summary>
public sealed class HunyuanVideo15PromptTests
{
    private static JsonElement Fixture()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "Fixtures", "hy15_text.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
        return document.RootElement.Clone();
    }

    [Fact]
    public void SystemPromptMatchesTheReferenceByteForByte()
    {
        Assert.Equal(HunyuanVideo15Prompt.SystemPrompt, Fixture().GetProperty("system_prompt").GetString());
    }

    [Fact]
    public void GlyphTextExtractionAndFormattingMatchTheReference()
    {
        foreach (JsonElement testCase in Fixture().GetProperty("glyph_cases").EnumerateArray())
        {
            string prompt = testCase.GetProperty("prompt").GetString()!;
            string[] expectedTexts = testCase.GetProperty("texts").EnumerateArray().Select(t => t.GetString()!).ToArray();
            string? expectedFormatted = testCase.GetProperty("formatted").GetString();

            Assert.Equal(expectedTexts, HunyuanVideo15Prompt.ExtractGlyphTexts(prompt));
            Assert.Equal(expectedFormatted, HunyuanVideo15Prompt.FormatGlyphText(prompt));
        }
    }

    [Fact]
    public void NullPromptHasNoGlyphText()
    {
        Assert.Empty(HunyuanVideo15Prompt.ExtractGlyphTexts(null));
        Assert.Null(HunyuanVideo15Prompt.FormatGlyphText(null));
    }

    // Qwen2.5 ids: <|im_start|> 151644, "user" 872, newline 198.
    private const int UserId = 872;
    private const int NewlineId = 198;

    [Fact]
    public void TokenizeKeepsEveryNewlineAndCropsRightAfterTheUserMarker()
    {
        Qwen2Tokenizer tokenizer = new();

        (int[] ids, int crop) = HunyuanVideo15Prompt.Tokenize(tokenizer, "A red fox runs through snow.");

        Assert.Equal(Qwen2Tokenizer.ImStartId, ids[crop - 3]);
        Assert.Equal(UserId, ids[crop - 2]);
        Assert.Equal(NewlineId, ids[crop - 1]);
        Assert.Equal(Qwen2Tokenizer.ImStartId, ids[0]);
        // Regression pin from the exact tokenizer (the reviewer's probe also gives 120 ids and a crop of 108); not yet compared with HF ids.
        Assert.Equal(120, ids.Length);
        Assert.Equal(108, crop);
        // system, user and assistant turns each have a newline after the role, plus one after each <|im_end|>: five newline ids.
        Assert.Equal(5, ids.Count(id => id == NewlineId));
    }

    [Fact]
    public void TokenizeEndsWithTheGenerationPrompt()
    {
        Qwen2Tokenizer tokenizer = new();

        (int[] ids, _) = HunyuanVideo15Prompt.Tokenize(tokenizer, "A red fox runs through snow.");
        int[] assistant = tokenizer.EncodeExact("assistant\n");

        Assert.Equal(assistant, ids[^assistant.Length..]);
        Assert.Equal(Qwen2Tokenizer.ImStartId, ids[^(assistant.Length + 1)]);
        Assert.Equal(NewlineId, ids[^(assistant.Length + 2)]);
        Assert.Equal(Qwen2Tokenizer.ImEndId, ids[^(assistant.Length + 3)]);
    }

    [Fact]
    public void CropPointIsTheSameForEveryPrompt()
    {
        Qwen2Tokenizer tokenizer = new();

        int first = HunyuanVideo15Prompt.Tokenize(tokenizer, "short").CropStart;
        int second = HunyuanVideo15Prompt.Tokenize(tokenizer, "a much longer prompt with several words in it, and a comma").CropStart;

        Assert.Equal(first, second);
        Assert.True(first > 100);
    }

    [Fact]
    public void LongPromptsAreTruncatedOnTheRightToTheWindowPlusTheTemplatePrefix()
    {
        Qwen2Tokenizer tokenizer = new();
        string prompt = string.Join(' ', Enumerable.Repeat("word", 1500));

        (int[] ids, int crop) = HunyuanVideo15Prompt.Tokenize(tokenizer, prompt);

        Assert.Equal(HunyuanVideo15Prompt.MaxTextTokens + crop, ids.Length);
        Assert.Equal(UserId, ids[crop - 2]);
        // The end of the template, generation prompt included, is cut off, as the reference's truncation=True does.
        Assert.NotEqual(NewlineId, ids[^1]);
    }

    [Fact]
    public void FindCropStartUsesTheThreeIdMarker()
    {
        Qwen2Tokenizer tokenizer = new();
        int[] ids = [5, Qwen2Tokenizer.ImStartId, UserId, NewlineId, 9];

        Assert.Equal(4, HunyuanVideo15Prompt.FindCropStart(ids, tokenizer));
        // Without the newline id the marker is incomplete, which is what the legacy tokenizer path produced.
        Assert.Equal(0, HunyuanVideo15Prompt.FindCropStart([5, Qwen2Tokenizer.ImStartId, UserId, 9], tokenizer));
    }

    [Fact]
    public void CropStartIsZeroWithoutTheUserMarker()
    {
        Qwen2Tokenizer tokenizer = new();

        Assert.Equal(0, HunyuanVideo15Prompt.FindCropStart([1, 2, 3], tokenizer));
        Assert.Equal(0, HunyuanVideo15Prompt.FindCropStart([], tokenizer));
    }

    [Fact]
    public void CropStartTakesTheFirstMarker()
    {
        Qwen2Tokenizer tokenizer = new();
        int[] user = tokenizer.EncodeExact("user\n");
        List<int> ids = [7, Qwen2Tokenizer.ImStartId];
        ids.AddRange(user);
        int firstEnd = ids.Count;
        ids.AddRange([9, Qwen2Tokenizer.ImStartId]);
        ids.AddRange(user);

        Assert.Equal(firstEnd, HunyuanVideo15Prompt.FindCropStart(ids.ToArray(), tokenizer));
    }
}
