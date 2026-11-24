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

    [Fact]
    public void TokenizeUsesTheChatTemplateWithAGenerationPrompt()
    {
        Qwen2Tokenizer tokenizer = new();
        const string prompt = "A red fox runs through snow.";

        (int[] ids, int crop) = HunyuanVideo15Prompt.Tokenize(tokenizer, prompt);

        Assert.Equal(tokenizer.EncodeChat(prompt, HunyuanVideo15Prompt.SystemPrompt, addGenerationPrompt: true), ids);
        Assert.True(crop > 0 && crop < ids.Length);
        Assert.Equal(Qwen2Tokenizer.ImStartId, ids[crop - 1 - tokenizer.EncodeRaw("user\n").Count]);
        Assert.Equal(tokenizer.EncodeRaw("A")[0], ids[crop]);
    }

    [Fact]
    public void CropPointIsTheSameForEveryPrompt()
    {
        Qwen2Tokenizer tokenizer = new();

        int first = HunyuanVideo15Prompt.Tokenize(tokenizer, "short").CropStart;
        int second = HunyuanVideo15Prompt.Tokenize(tokenizer, "a much longer prompt with several words in it, and a comma").CropStart;

        Assert.Equal(first, second);
    }

    [Fact]
    public void LongPromptsAreTruncatedOnTheRightToTheWindowPlusTheTemplatePrefix()
    {
        Qwen2Tokenizer tokenizer = new();
        string prompt = string.Join(' ', Enumerable.Repeat("word", 1500));

        (int[] ids, int crop) = HunyuanVideo15Prompt.Tokenize(tokenizer, prompt);

        Assert.Equal(HunyuanVideo15Prompt.MaxTextTokens + crop, ids.Length);
        Assert.Equal(Qwen2Tokenizer.ImStartId, ids[crop - 1 - tokenizer.EncodeRaw("user\n").Count]);
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
        IReadOnlyList<int> user = tokenizer.EncodeRaw("user\n");
        List<int> ids = [7, Qwen2Tokenizer.ImStartId];
        ids.AddRange(user);
        int firstEnd = ids.Count;
        ids.AddRange([9, Qwen2Tokenizer.ImStartId]);
        ids.AddRange(user);

        Assert.Equal(firstEnd, HunyuanVideo15Prompt.FindCropStart(ids.ToArray(), tokenizer));
    }
}
