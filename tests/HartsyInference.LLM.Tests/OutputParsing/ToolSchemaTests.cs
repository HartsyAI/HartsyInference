using System.Text.Json;
using HartsyInference.LLM.ChatTemplates;
using Xunit;

namespace HartsyInference.LLM.Tests.OutputParsing;

/// <summary>Which tool parameters take raw text: the cases follow llama.cpp's schema reading (<c>common/json-schema.cpp</c>,
/// <c>may_be_string</c>) keyword by keyword.</summary>
public sealed class ToolSchemaTests
{
    [Theory]
    [InlineData("""{"type":"string"}""", true)]
    [InlineData("""{"type":["integer","string"]}""", true)]
    [InlineData("""{"type":["integer","null"]}""", false)]
    [InlineData("""{"type":"integer"}""", false)]
    [InlineData("""{"type":"object"}""", false)]
    [InlineData("""{}""", false)]
    [InlineData("""{"description":"anything"}""", false)]
    [InlineData("""{"const":"fixed"}""", true)]
    [InlineData("""{"const":3}""", false)]
    [InlineData("""{"enum":[1,"two"]}""", true)]
    [InlineData("""{"enum":[1,2]}""", false)]
    [InlineData("""{"type":"string","enum":[1,2]}""", false)]
    [InlineData("""{"anyOf":[{"type":"integer"},{"type":"string"}]}""", true)]
    [InlineData("""{"oneOf":[{"type":"integer"}],"anyOf":[{"type":"string"}]}""", false)]
    [InlineData("""{"allOf":[{},{"type":"string"}]}""", true)]
    [InlineData("""{"allOf":[{"type":"string"},{"type":"integer"}]}""", false)]
    [InlineData("""{"allOf":[{}]}""", false)]
    [InlineData("""{"pattern":"^[a-z]+$"}""", true)]
    [InlineData("""{"format":"date-time"}""", true)]
    [InlineData("""{"format":"email"}""", false)]
    [InlineData("""{"items":{"type":"string"}}""", false)]
    [InlineData("""{"$ref":"#/$defs/Name"}""", true)]
    [InlineData("""{"$ref":"#/$defs/Loop"}""", false)]
    [InlineData("""{"$ref":"https://example.com/remote"}""", false)]
    public void MayBeString_FollowsLlamaCpp(string schema, bool expected)
    {
        using JsonDocument root = JsonDocument.Parse("""{"$defs":{"Name":{"type":"string"},"Loop":{"$ref":"#/$defs/Loop"}}}""");
        using JsonDocument node = JsonDocument.Parse(schema);

        Assert.Equal(expected, ToolSchema.MayBeString(node.RootElement, root.RootElement));
    }

    [Fact]
    public void StringParameters_MapsEachFunctionsParameters()
    {
        ToolSpec weather = ToolSpec.FromJson("""
            {"type":"function","function":{"name":"get_weather","parameters":{"type":"object","$defs":{"City":{"type":"string"}},
            "properties":{"city":{"$ref":"#/$defs/City"},"days":{"type":"integer"}}}}}
            """);
        ToolSpec ping = ToolSpec.FromJson("""{"type":"function","function":{"name":"ping"}}""");

        IReadOnlyDictionary<string, IReadOnlyDictionary<string, bool>> map = ToolSchema.StringParameters([weather, ping]);

        Assert.True(map["get_weather"]["city"]);
        Assert.False(map["get_weather"]["days"]);
        Assert.Empty(map["ping"]);
        Assert.Empty(ToolSchema.StringParameters(null));
    }
}
