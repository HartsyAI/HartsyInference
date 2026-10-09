using HartsyInference.Cli.Dispatch;
using Xunit;

namespace HartsyInference.Cli.Tests;

/// <summary><c>--reasoning-effort</c> is read by the parser the chat API uses: a named effort or an integer in [1, 100]. An integer outside the range, including one too
/// large for an int, is refused as out of range rather than as an unknown name.</summary>
public sealed class ReasoningEffortCliTests
{
    [Theory]
    [InlineData("low", 50)]
    [InlineData("max", 100)]
    [InlineData("1", 1)]
    [InlineData("100", 100)]
    public void Names_And_Integers_In_Range_Parse(string value, int expected) =>
        Assert.Equal(expected, GenerationDispatch.ParseReasoningEffort(value));

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    public void Unset_Is_Null(string? value) => Assert.Null(GenerationDispatch.ParseReasoningEffort(value));

    [Theory]
    [InlineData("0")]
    [InlineData("101")]
    [InlineData("99999999999")]
    public void An_Integer_Outside_1_To_100_Is_Refused_As_Out_Of_Range(string value)
    {
        ArgumentException ex = Assert.Throws<ArgumentException>(() => GenerationDispatch.ParseReasoningEffort(value));

        Assert.Equal($"Reasoning effort {value} is out of range; use an integer in [1, 100], or low, high or max.", ex.Message);
    }

    [Theory]
    [InlineData("medium")]
    [InlineData("Max")]
    [InlineData("-5")]
    public void Anything_Else_Is_Refused_As_Unknown(string value)
    {
        ArgumentException ex = Assert.Throws<ArgumentException>(() => GenerationDispatch.ParseReasoningEffort(value));

        Assert.Equal($"Unknown reasoning effort '{value}'; use low, high, max or an integer in [1, 100].", ex.Message);
    }
}
