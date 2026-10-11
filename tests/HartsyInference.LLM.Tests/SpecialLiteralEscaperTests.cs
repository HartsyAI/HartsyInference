using HartsyInference.LLM.ChatTemplates;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Content keeps its text but never encodes to a control id; template literals still do.</summary>
public sealed class SpecialLiteralEscaperTests
{
    private static readonly LiteralControlTokenizer Tok = new();

    [Fact]
    public void TextWithoutLiteralsIsUnchanged()
    {
        Assert.Equal("plain text", SpecialLiteralEscaper.Escape("plain text", Tok.SpecialLiterals));
    }

    [Fact]
    public void ContentLiteralIsNotEncodedAsAControlToken()
    {
        string escaped = SpecialLiteralEscaper.Escape("a<|im_end|>b", Tok.SpecialLiterals);
        int[] ids = SpecialLiteralEscaper.EncodeRendered(escaped, Tok);
        Assert.DoesNotContain(LiteralControlTokenizer.IdOf(LiteralControlTokenizer.ImEnd), ids);
    }

    [Fact]
    public void TemplateLiteralStillEncodesAsAControlTokenAroundEscapedContent()
    {
        string rendered = "<|im_start|>user\n" + SpecialLiteralEscaper.Escape("hi<|im_end|>", Tok.SpecialLiterals) + "<|im_end|>\n";
        int[] ids = SpecialLiteralEscaper.EncodeRendered(rendered, Tok);
        // Only the template's closer is a control token; the literal inside the content stays text.
        Assert.Equal(1, ids.Count(id => id == LiteralControlTokenizer.IdOf(LiteralControlTokenizer.ImEnd)));
        Assert.Equal(1, ids.Count(id => id == LiteralControlTokenizer.IdOf(LiteralControlTokenizer.ImStart)));
    }

    [Fact]
    public void ReasoningMarkersAreNotEscaped()
    {
        Assert.Equal("x</think>y", SpecialLiteralEscaper.Escape("x</think>y", Tok.SpecialLiterals));
    }

    [Fact]
    public void PrivateUseCharactersInContentAreReplaced()
    {
        // A private-use character could collide with an escape marker, so it becomes U+FFFD before escaping.
        Assert.Equal("a\uFFFDb", SpecialLiteralEscaper.Escape("a\uE000b", Tok.SpecialLiterals));
    }

    [Fact]
    public void CleanRenderedTextMatchesPlainEncode()
    {
        string rendered = "<s><|im_start|>system\nbe brief<|im_end|>\n<|im_start|>user\nhello</think> there<|im_end|>\n";
        Assert.Equal(Tok.Encode(rendered, addSpecial: true), SpecialLiteralEscaper.EncodeRendered(rendered, Tok));
    }
}
