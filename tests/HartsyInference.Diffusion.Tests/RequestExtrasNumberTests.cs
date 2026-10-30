using System.Text.Json;
using HartsyInference.Engine.Features;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Numeric request extras as they arrive: a JSON request body hands them over as <see cref="JsonElement"/>, which
/// <see cref="RequestExtras.Number"/> must read rather than drop to its fallback.</summary>
public sealed class RequestExtrasNumberTests
{
    [Theory]
    [InlineData("0.6", 0.6)]
    [InlineData("786432", 786432)]
    [InlineData("\"4.5\"", 4.5)]
    public void Number_ReadsJsonNumbersAndNumericStrings(string json, double expected)
    {
        using JsonDocument document = JsonDocument.Parse(json);
        Dictionary<string, object> extra = new() { ["k"] = document.RootElement.Clone() };
        Assert.Equal(expected, RequestExtras.Number(extra, "k", -1));
    }

    [Theory]
    [InlineData("\"abc\"")]
    [InlineData("\"NaN\"")]
    [InlineData("\"Infinity\"")]
    [InlineData("true")]
    [InlineData("null")]
    public void Number_FallsBackOnANonNumericJsonValue(string json)
    {
        using JsonDocument document = JsonDocument.Parse(json);
        Dictionary<string, object> extra = new() { ["k"] = document.RootElement.Clone() };
        Assert.Equal(-1, RequestExtras.Number(extra, "k", -1));
    }
}
