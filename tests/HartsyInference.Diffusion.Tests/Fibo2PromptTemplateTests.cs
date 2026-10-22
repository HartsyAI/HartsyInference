using HartsyInference.Diffusion.Prompting;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Fibo 2.0 prompt template: the Qwen3 user turn with no think block, the null caption, and the glyph refusal.</summary>
public sealed class Fibo2PromptTemplateTests
{
    private const string NullCaptionText =
        "{\"short_description\":\"\",\"objects\":[],\"background_setting\":\"\",\"lighting\":{},\"aesthetics\":{},"
        + "\"photographic_characteristics\":{},\"style_medium\":\"\",\"context\":\"\",\"artistic_style\":\"\"}";

    [Fact]
    public void NullCaption_IsTheComfyUiEmptyStructuredPrompt()
    {
        Assert.Equal(NullCaptionText, Fibo2PromptTemplate.NullCaption);
    }

    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("\n\t")]
    public void CaptionFor_EmptyPrompt_UsesTheNullCaption(string prompt)
    {
        Assert.Equal(Fibo2PromptTemplate.NullCaption, Fibo2PromptTemplate.CaptionFor(prompt));
    }

    [Fact]
    public void CaptionFor_NonEmptyPrompt_IsThePrompt()
    {
        Assert.Equal("a red bicycle", Fibo2PromptTemplate.CaptionFor("a red bicycle"));
    }

    /// <summary>The ids are the template spliced around the caption: <c>&lt;|im_start|&gt;</c>, the user turn (one BPE call with
    /// the caption, as the tokenizer merges it), <c>&lt;|im_end|&gt;</c>, a newline, <c>&lt;|im_start|&gt;</c>, and the assistant
    /// prefix. No think block follows.</summary>
    [Fact]
    public void Tokenize_WrapsTheCaptionInTheUserTurn_WithoutAThinkBlock()
    {
        using Qwen3Tokenizer tokenizer = new();
        const string caption = "a red bicycle";

        int[] ids = Fibo2PromptTemplate.Tokenize(tokenizer, caption);

        List<int> expected = [Qwen3Tokenizer.ImStartId];
        expected.AddRange(tokenizer.EncodeRaw("user\n" + caption));
        expected.Add(Qwen3Tokenizer.ImEndId);
        expected.AddRange(tokenizer.EncodeRaw("\n"));
        expected.Add(Qwen3Tokenizer.ImStartId);
        expected.AddRange(tokenizer.EncodeRaw("assistant\n"));
        Assert.Equal(expected, ids);
        Assert.DoesNotContain(Qwen3Tokenizer.ThinkStartId, ids);
        Assert.DoesNotContain(Qwen3Tokenizer.ThinkEndId, ids);
    }

    [Fact]
    public void Tokenize_EmptyPrompt_EncodesTheNullCaption()
    {
        using Qwen3Tokenizer tokenizer = new();

        int[] fromEmpty = Fibo2PromptTemplate.Tokenize(tokenizer, Fibo2PromptTemplate.CaptionFor(string.Empty));
        int[] fromNull = Fibo2PromptTemplate.Tokenize(tokenizer, Fibo2PromptTemplate.NullCaption);

        Assert.Equal(fromNull, fromEmpty);
    }

    [Fact]
    public void Tokenize_GlyphTextEntry_IsRefusedNotDropped()
    {
        using Qwen3Tokenizer tokenizer = new();
        const string caption = "{\"short_description\":\"a sign\",\"text_render\":[{\"text\":\"OPEN\"}]}";

        NotSupportedException error = Assert.Throws<NotSupportedException>(() => Fibo2PromptTemplate.Tokenize(tokenizer, caption));

        Assert.Contains("glyph text images are not implemented yet for Fibo 2.0", error.Message, StringComparison.Ordinal);
    }
}
