using HartsyInference.LLM.ChatTemplates;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Python-style tuple literals: a comma inside parentheses builds a sequence, a bare parenthesized expression stays grouping.</summary>
public sealed class JinjaTupleLiteralTests
{
    private static string Render(string template, Dictionary<string, object?>? ctx = null) => new JinjaEngine(template).Render(ctx ?? []);

    [Theory]
    [InlineData("{{ 'high' in ('high', 'low') }}", "True")]
    [InlineData("{{ 'mid' in ('high', 'low') }}", "False")]
    [InlineData("{% if 'mid' not in ('high', 'low') %}refused{% endif %}", "refused")]
    [InlineData("{{ 'a' in ('a',) }}", "True")]
    [InlineData("{{ 'a' in ('a', 'b',) }}", "True")]
    [InlineData("{{ 'a' in () }}", "False")]
    public void TuplesBehaveAsSequences(string template, string expected) => Assert.Equal(expected, Render(template));

    [Fact]
    public void ParenthesesStillGroup() => Assert.Equal("6", Render("{{ (1 + 2) * 2 }}"));
}
