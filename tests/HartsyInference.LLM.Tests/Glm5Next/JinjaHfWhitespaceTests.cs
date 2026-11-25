using HartsyInference.LLM.ChatTemplates;
using Xunit;

namespace HartsyInference.LLM.Tests.Glm5Next;

/// <summary>The engine's Jinja features the Sage template relies on, pinned on small templates whose expected output is what transformers renders
/// (trim_blocks and lstrip_blocks on, as the Hugging Face environment sets them).</summary>
public sealed class JinjaHfWhitespaceTests
{
    private static string Render(string template, bool hfWhitespace, Dictionary<string, object?> context)
    {
        Dictionary<string, object?> full = new(context) { ["bos_token"] = string.Empty, ["eos_token"] = string.Empty };
        return new JinjaEngine(template, hfWhitespace).Render(full);
    }

    [Fact]
    public void LstripBlocks_RemovesIndentationBeforeABlockTag()
    {
        const string template = "Example: {\"a\": 1}\n    {% if x %}y{% endif %}\n";
        Dictionary<string, object?> ctx = new() { ["x"] = true };

        Assert.Equal("Example: {\"a\": 1}\ny", Render(template, hfWhitespace: true, ctx));
        Assert.Equal("Example: {\"a\": 1}\n    y", Render(template, hfWhitespace: false, ctx));
    }

    [Fact]
    public void LstripBlocks_StillAppliesWhenALiteralBraceSitsBeforeTheTag()
    {
        // The literal "{" that does not open a tag must not stop the indentation strip before the block tag on the next line.
        const string template = "a {b\n    {% if x %}y{% endif %}";
        Dictionary<string, object?> ctx = new() { ["x"] = true };

        Assert.Equal("a {b\ny", Render(template, hfWhitespace: true, ctx));
    }

    [Fact]
    public void ExpressionBraceIsNotStrippedAsABlock()
    {
        Dictionary<string, object?> ctx = new() { ["x"] = "v" };

        Assert.Equal("  v", Render("  {{ x }}", hfWhitespace: true, ctx));
    }

    [Fact]
    public void IntegerSubscript_ReadsAListItem()
    {
        Dictionary<string, object?> ctx = new() { ["xs"] = new List<object?> { "first", "second" } };

        Assert.Equal("second", Render("{{ xs.1 }}", hfWhitespace: false, ctx));
    }

    [Fact]
    public void Capitalize_UpperCasesOnlyTheFirstCharacter()
    {
        Assert.Equal("Hello world", Render("{{ 'hELLO wORLD' | capitalize }}", hfWhitespace: false, new Dictionary<string, object?>()));
        Assert.Equal("", Render("{{ '' | capitalize }}", hfWhitespace: false, new Dictionary<string, object?>()));
    }
}
