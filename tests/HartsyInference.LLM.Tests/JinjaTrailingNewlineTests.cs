using HartsyInference.LLM.ChatTemplates;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Jinja2's default <c>keep_trailing_newline=False</c>: one newline at the end of a template's source is dropped, a second one is kept.</summary>
public sealed class JinjaTrailingNewlineTests
{
    private static string Render(string template) => new JinjaEngine(template).Render([]);

    [Theory]
    [InlineData("a\n", "a")]
    [InlineData("a\n\n", "a\n")]
    [InlineData("a\r\n", "a")]
    [InlineData("a", "a")]
    [InlineData("{% if true %}{{ 'x' }}{% endif %}\n", "x")]
    [InlineData("{{ 'x\n' }}\n", "x\n")]
    public void DropsOneTrailingNewlineOfTheSource(string template, string expected) => Assert.Equal(expected, Render(template));
}
