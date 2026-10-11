using HartsyInference.Engine.Requests;
using HartsyInference.Tools.Parsing;
using Xunit;

namespace HartsyInference.Tools.Tests.Parsing;

/// <summary>The markup dialects: Qwen's XML parameters, GLM's arg pairs and DeepSeek-R1's fenced blocks, parsed at many split points with their values typed by the offered schemas.</summary>
public sealed class ToolCallMarkupParserTests
{
    private static readonly ToolDefinition GetTime = new()
    {
        Name = "get_time",
        Description = "time",
        JsonSchema = "{\"type\":\"object\",\"properties\":{\"tz\":{\"type\":\"string\"},\"n\":{\"type\":\"integer\"},\"strict\":{\"type\":\"boolean\"}}}",
    };

    private static readonly ToolDefinition HangUp = new() { Name = "hang_up", Description = "end", JsonSchema = "{\"type\":\"object\",\"properties\":{}}" };

    private static readonly ToolDefinition[] Tools = [GetTime, HangUp];

    private static (string Forwarded, List<NativeToolCall> Calls) RunTools(ToolCallFormat format, string completion, int seed = 1)
        => ParserDriver.Drive(ToolCallParser.ForTools(format, Tools, "call_1_"), ParserDriver.Pieces(completion, new Random(seed), 5));

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(9)]
    [InlineData(31)]
    public void QwenXml_SingleCallWithTypedValues(int seed)
    {
        const string completion = "<tool_call>\n<function=get_time>\n<parameter=tz>\nUTC\n</parameter>\n<parameter=n>\n3\n</parameter>\n<parameter=strict>\ntrue\n</parameter>\n</function>\n</tool_call>";
        (string forwarded, List<NativeToolCall> calls) = RunTools(ToolCallFormat.QwenXml, completion, seed);
        NativeToolCall call = Assert.Single(calls);
        Assert.Equal("get_time", call.Name);
        Assert.Equal("call_1_0", call.Id);
        Assert.Equal("{\"tz\":\"UTC\",\"n\":3,\"strict\":true}", call.Arguments);
        Assert.Equal("", forwarded.Trim());
    }

    [Fact]
    public void QwenXml_MultilineParameterKeepsItsInnerNewlines()
    {
        const string completion = "<tool_call><function=get_time><parameter=tz>\nline one\nline two\n</parameter></function></tool_call>";
        (_, List<NativeToolCall> calls) = RunTools(ToolCallFormat.QwenXml, completion);
        Assert.Equal("{\"tz\":\"line one\\nline two\"}", Assert.Single(calls).Arguments);
    }

    [Fact]
    public void QwenXml_BareFunctionFormWithoutTheWrapper()
    {
        (_, List<NativeToolCall> calls) = RunTools(ToolCallFormat.QwenXml, "<function=hang_up>\n</function>\n");
        Assert.Equal("hang_up", Assert.Single(calls).Name);
    }

    [Fact]
    public void QwenXml_UnclosedSpanIsForwardedAtFlush()
    {
        (string forwarded, List<NativeToolCall> calls) = RunTools(ToolCallFormat.QwenXml, "<tool_call><function=hang_up>");
        Assert.Empty(calls);
        Assert.Equal("<tool_call><function=hang_up>", forwarded);
    }

    [Fact]
    public void QwenXml_BareUnknownToolIsNotACall()
    {
        (string forwarded, List<NativeToolCall> calls) = RunTools(ToolCallFormat.QwenXml, "<function=delete_all></function>");
        Assert.Empty(calls);
        Assert.Equal("<function=delete_all></function>", forwarded);
    }

    [Fact]
    public void GlmXml_ArgKeyPairsBecomeArguments()
    {
        const string completion = "<tool_call>get_time\n<arg_key>tz</arg_key>\n<arg_value>UTC</arg_value>\n<arg_key>n</arg_key>\n<arg_value>5</arg_value>\n</tool_call>";
        (_, List<NativeToolCall> calls) = RunTools(ToolCallFormat.GlmXml, completion, seed: 4);
        NativeToolCall call = Assert.Single(calls);
        Assert.Equal("get_time", call.Name);
        Assert.Equal("{\"tz\":\"UTC\",\"n\":5}", call.Arguments);
    }

    [Fact]
    public void GlmXml_ValueIsTypedBySchemaNotByShape()
    {
        // "5" is a string value unless the schema says integer; here it is, so it becomes a number.
        (_, List<NativeToolCall> calls) = RunTools(ToolCallFormat.GlmXml, "<tool_call>get_time\n<arg_key>tz</arg_key>\n<arg_value>007</arg_value>\n</tool_call>");
        Assert.Equal("{\"tz\":\"007\"}", Assert.Single(calls).Arguments);
    }

    [Fact]
    public void DeepSeekR1_TwoCallsInOneBlock()
    {
        const string completion =
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_time\n```json\n{\"tz\":\"UTC\"}\n```<｜tool▁call▁end｜>"
            + "<｜tool▁call▁begin｜>function<｜tool▁sep｜>hang_up\n```json\n{}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>";
        (string forwarded, List<NativeToolCall> calls) = RunTools(ToolCallFormat.DeepSeekR1, completion, seed: 6);
        Assert.Equal(["get_time", "hang_up"], calls.Select(c => c.Name));
        Assert.Equal("{\"tz\":\"UTC\"}", calls[0].Arguments);
        Assert.Equal("", forwarded.Trim());
    }

    [Fact]
    public void DeepSeekR1_SingleCallMarkerForm()
    {
        (_, List<NativeToolCall> calls) = RunTools(ToolCallFormat.DeepSeekR1, "<｜tool▁call▁begin｜>function<｜tool▁sep｜>hang_up\n```json\n{}\n```<｜tool▁call▁end｜>");
        Assert.Equal("hang_up", Assert.Single(calls).Name);
    }

    [Fact]
    public void DeepSeekR1_InvalidFencedJsonIsForwarded()
    {
        (string forwarded, List<NativeToolCall> calls) = RunTools(ToolCallFormat.DeepSeekR1, "<｜tool▁call▁begin｜>function<｜tool▁sep｜>hang_up\n```json\nnot json\n```<｜tool▁call▁end｜>");
        Assert.Empty(calls);
        Assert.Contains("not json", forwarded, StringComparison.Ordinal);
    }

    [Fact]
    public void Hermes_NamedLineFormWithNewlineForAnOfferedName()
    {
        // GLM-4-0414 writes the bare call as "name" on one line and its JSON object on the next.
        (_, List<NativeToolCall> calls) = RunTools(ToolCallFormat.Hermes, "get_time\n{\"tz\": \"UTC\"}");
        NativeToolCall call = Assert.Single(calls);
        Assert.Equal("get_time", call.Name);
    }

    [Fact]
    public void Hermes_NamedLineFormDoesNotHoldProse()
    {
        (string forwarded, List<NativeToolCall> calls) = RunTools(ToolCallFormat.Hermes, "hang_up\nThat is all.");
        Assert.Empty(calls);
        Assert.Equal("hang_up\nThat is all.", forwarded);
    }
}
