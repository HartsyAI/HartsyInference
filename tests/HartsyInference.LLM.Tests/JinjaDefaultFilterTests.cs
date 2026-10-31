using HartsyInference.LLM.ChatTemplates;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Jinja2's <c>default</c> filter replaces an undefined value only; a defined false, zero or empty string stays unless the second argument is true.</summary>
public sealed class JinjaDefaultFilterTests
{
    private static string Render(string template, Dictionary<string, object?> ctx) => new JinjaEngine(template).Render(ctx);

    [Fact]
    public void KeepsADefinedFalse() => Assert.Equal("false", Render("{{ 'true' if (x | default(true)) else 'false' }}", new() { ["x"] = false }));

    [Fact]
    public void UsesTheDefaultForAnUndefinedValue() => Assert.Equal("true", Render("{{ 'true' if (x | default(true)) else 'false' }}", []));

    [Fact]
    public void KeepsAnEmptyStringUnlessBooleanIsSet() =>
        Assert.Equal("[][fallback]", Render("[{{ s | default('fallback') }}][{{ s | default('fallback', true) }}]", new() { ["s"] = "" }));
}
