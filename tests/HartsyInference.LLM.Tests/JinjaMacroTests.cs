using HartsyInference.LLM.ChatTemplates;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Template-defined macros and the character-set form of strip/lstrip/rstrip. The engine used to skip every <c>{% macro %}</c>, so a call such as
/// Qwen3.5's <c>render_content(...)</c> threw "Unsupported Jinja call expression" and the whole template fell back to ChatML.</summary>
public sealed class JinjaMacroTests
{
    private static string Render(string template, Dictionary<string, object?>? ctx = null) => new JinjaEngine(template).Render(ctx ?? []);

    [Fact]
    public void Macro_PositionalArguments()
        => Assert.Equal("<b>hi</b>!", Render("{% macro bold(t) %}<b>{{ t }}</b>{% endmacro %}{{ bold('hi') }}!"));

    [Fact]
    public void Macro_DefaultsAndKeywordArguments()
    {
        const string t = "{% macro tag(text, name='p', cls=none) %}<{{ name }}{% if cls %} class=\"{{ cls }}\"{% endif %}>{{ text }}</{{ name }}>{% endmacro %}";
        Assert.Equal("<p>a</p>", Render(t + "{{ tag('a') }}"));
        Assert.Equal("<h1>a</h1>", Render(t + "{{ tag('a', 'h1') }}"));
        Assert.Equal("<p class=\"x\">a</p>", Render(t + "{{ tag('a', cls='x') }}"));
        Assert.Equal("<h2 class=\"y\">a</h2>", Render(t + "{{ tag(text='a', cls='y', name='h2') }}"));
    }

    [Fact]
    public void Macro_DefaultMayUseAnEarlierParameter()
        => Assert.Equal("3-3", Render("{% macro f(a, b=a) %}{{ a }}-{{ b }}{% endmacro %}{{ f(3) }}"));

    [Fact]
    public void Macro_ResultTakesFilters_AndWhitespaceControlStillApplies()
        => Assert.Equal("[x]", Render("{%- macro pad(t) %}\n  {{ t }}\n{% endmacro -%}[{{ pad('x')|trim }}]"));

    [Fact]
    public void Macro_SetInsideStaysInsideTheCall()
        => Assert.Equal("a1|a", Render("{% set v = 'a' %}{% macro f() %}{% set v = 'inner' %}1{% endmacro %}{{ v }}{{ f() }}|{{ v }}"));

    [Fact]
    public void Macro_CanCallAnEarlierMacro_AndRunInsideALoop()
    {
        const string t = "{% macro item(x) %}({{ x }}){% endmacro %}{% macro row(xs) %}{% for x in xs %}{{ item(x) }}{% endfor %}{% endmacro %}{{ row([1, 2, 3]) }}";
        Assert.Equal("(1)(2)(3)", Render(t));
    }

    [Fact]
    public void Macro_ReadsVariablesDefinedBeforeTheCall()
        => Assert.Equal("hello, world", Render("{% set greeting = 'hello' %}{% macro f(who) %}{{ greeting }}, {{ who }}{% endmacro %}{{ f('world') }}"));

    [Fact]
    public void Macro_UnknownKeywordOrTooManyArguments_Throw()
    {
        const string def = "{% macro f(a) %}{{ a }}{% endmacro %}";
        Assert.Throws<NotSupportedException>(() => Render(def + "{{ f(1, b=2) }}"));
        Assert.Throws<NotSupportedException>(() => Render(def + "{{ f(1, 2) }}"));
    }

    [Fact]
    public void Macro_BodyIsVisibleToStaticRoleInspection()
        => Assert.True(new JinjaEngine("{% macro m(x) %}{% if x.role == 'observation' %}o{% endif %}{% endmacro %}").ComparesRoleTo("observation"));

    [Theory]
    [InlineData("{{ '\n\n x \n\n'.strip('\n') }}", " x ")]
    [InlineData("{{ '\n\n x \n\n'.lstrip('\n') }}", " x \n\n")]
    [InlineData("{{ '\n\n x \n\n'.rstrip('\n') }}", "\n\n x ")]
    [InlineData("{{ 'xxaxx'.strip('x') }}", "a")]
    [InlineData("{{ '  a  '.strip() }}", "a")]
    [InlineData("{{ '  a  '.lstrip() }}", "a  ")]
    [InlineData("{{ '  a  '.rstrip() }}", "  a")]
    public void Strip_WithACharacterSet_StripsOnlyThoseCharacters(string template, string expected)
        => Assert.Equal(expected, Render(template));

    [Fact]
    public void ForOverAMapping_YieldsItsKeys()
    {
        Dictionary<string, object?> ctx = new() { ["d"] = new Dictionary<string, object?> { ["a"] = 1L, ["b"] = 2L } };
        Assert.Equal("a=1;b=2;", Render("{% for k in d %}{{ k }}={{ d[k] }};{% endfor %}", ctx));
        Assert.Equal("a,b", Render("{{ d|join(',') }}", ctx));
        Assert.Equal("2", Render("{{ d|length }}", ctx));
    }

    [Fact]
    public void Loop_ExposesNeighbouringItemsAndReverseIndexes()
    {
        const string t = "{% for x in [10, 20, 30] %}[{{ loop.revindex }}/{{ loop.revindex0 }}|{% if loop.previtem %}{{ loop.previtem }}{% else %}-{% endif %}<{{ x }}>{% if not loop.last %}{{ loop.nextitem }}{% else %}-{% endif %}]{% endfor %}";
        Assert.Equal("[3/2|-<10>20][2/1|10<20>30][1/0|20<30>-]", Render(t));
    }
}
