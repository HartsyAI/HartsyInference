using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tests.MemoryManagement;
using Xunit;

namespace HartsyInference.Core.Tests.Configuration;

/// <summary>Pins the two boolean grammars a knob can carry, so migrating a call site cannot silently change what an exported value means.</summary>
/// <remarks>The engine grew six inconsistent spellings of "on". Two survive as <see cref="BoolGrammar"/>, and they
/// genuinely disagree: on a default-ON knob the historic <c>!= "0"</c> test reads <c>false</c> as <b>true</b>,
/// while <c>EnvSwitch.IsEnabled</c> reads it as false. C2 moves knobs, it does not reconcile grammars — so both
/// behaviors are asserted here rather than unified.</remarks>
[Collection(EnvironmentSensitiveCollection.Name)]
public sealed class KnobGrammarTests
{
    private const string Var = "HARTSY_KNOB_GRAMMAR_TEST";

    private static void With(string? value, Action body)
    {
        string? previous = Environment.GetEnvironmentVariable(Var);
        try
        {
            Environment.SetEnvironmentVariable(Var, value);
            body();
        }
        finally
        {
            Environment.SetEnvironmentVariable(Var, previous);
        }
    }

    private static Knob<bool> Declare(bool defaultValue, BoolGrammar grammar, string id)
        => new(id, Var, defaultValue, KnobScope.Runtime, KnobDomain.Numerics, "test knob", grammar);

    /// <summary>The historic <c>== "1"</c> call sites: only a literal 1 turns it on, so <c>true</c> stays OFF.</summary>
    [Theory]
    [InlineData("1", true)]
    [InlineData("0", false)]
    [InlineData("true", false)]
    [InlineData("TRUE", false)]
    [InlineData("", false)]
    [InlineData(null, false)]
    [InlineData("nonsense", false)]
    public void ExactGrammar_DefaultOff_OnlyLiteralOneEnables(string? value, bool expected)
    {
        Knob<bool> knob = Declare(false, BoolGrammar.Exact, $"test.exactOff.{value ?? "null"}");
        With(value, () => Assert.Equal(expected, knob.Value));
    }

    /// <summary>The historic <c>!= "0"</c> call sites. Note <c>"false"</c> resolves to <b>true</b> — surprising, but exactly what that code did.</summary>
    [Theory]
    [InlineData("0", false)]
    [InlineData("1", true)]
    [InlineData("false", true)]
    [InlineData("", true)]
    [InlineData(null, true)]
    [InlineData("nonsense", true)]
    public void ExactGrammar_DefaultOn_OnlyLiteralZeroDisables(string? value, bool expected)
    {
        Knob<bool> knob = Declare(true, BoolGrammar.Exact, $"test.exactOn.{value ?? "null"}");
        With(value, () => Assert.Equal(expected, knob.Value));
    }

    /// <summary>The <c>EnvSwitch.IsEnabled</c> convention, asserted against both defaults so the fallback arm is covered.</summary>
    [Theory]
    [InlineData("1", false, true)]
    [InlineData("true", false, true)]
    [InlineData("TRUE", false, true)]
    [InlineData("0", true, false)]
    [InlineData("false", true, false)]
    [InlineData("False", true, false)]
    [InlineData("nonsense", true, true)]
    [InlineData("nonsense", false, false)]
    [InlineData(null, true, true)]
    [InlineData(null, false, false)]
    public void TriStateGrammar_MatchesEnvSwitchIsEnabled(string? value, bool defaultOn, bool expected)
    {
        Knob<bool> knob = Declare(defaultOn, BoolGrammar.TriState, $"test.tri.{value ?? "null"}.{defaultOn}");
        With(value, () => Assert.Equal(expected, knob.Value));
    }

    /// <summary>An explicit override beats the environment in both directions, which is what makes CLI <c>--set</c> and the API authoritative in C4.</summary>
    [Fact]
    public void Override_BeatsTheEnvironment()
    {
        Knob<bool> knob = Declare(false, BoolGrammar.Exact, "test.override");
        With("1", () =>
        {
            Assert.True(knob.Value);
            KnobStore.Set(knob, false);
            try
            {
                Assert.False(knob.Value);
            }
            finally
            {
                KnobStore.Clear(knob);
            }
            Assert.True(knob.Value);
        });
    }

    /// <summary>Numeric knobs fall back to the declared default when the value will not parse, matching <c>EnvSwitch.GetInt/GetLong/GetFloat</c>.</summary>
    [Theory]
    [InlineData("4096", 4096L)]
    [InlineData("notanumber", 3072L)]
    [InlineData("", 3072L)]
    [InlineData(null, 3072L)]
    public void NumericKnob_FallsBackOnUnparsableValues(string? value, long expected)
    {
        Knob<long> knob = new($"test.long.{value ?? "null"}", Var, 3072L, KnobScope.Runtime, KnobDomain.Vram, "test knob");
        With(value, () => Assert.Equal(expected, knob.Value));
    }

    /// <summary>A string knob yields the raw value; unset yields its default, which is usually null for dump directories.</summary>
    [Theory]
    [InlineData("/tmp/dump", "/tmp/dump")]
    [InlineData(null, null)]
    [InlineData("", null)]
    public void StringKnob_YieldsRawValue(string? value, string? expected)
    {
        Knob<string?> knob = new($"test.str.{value ?? "null"}", Var, null, KnobScope.Runtime, KnobDomain.Diagnostics, "test knob");
        With(value, () => Assert.Equal(expected, knob.Value));
    }
}
