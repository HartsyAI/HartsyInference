using HartsyInference.Engine.Requests;
using HartsyInference.Tools.Parsing;
using Xunit;

namespace HartsyInference.Tools.Tests.Parsing;

/// <summary>The incremental parser per format: markers split across deltas, the bare payload forms the GGUF detokenizer leaves behind (marker tokens are special and dropped), several calls per turn, and every malformed shape coming back as plain text rather than an exception.</summary>
public sealed class ToolCallParserTests
{
    private const string HermesTagged = "Sure. <tool_call>\n{\"name\": \"hang_up\", \"arguments\": {\"reason\": \"done\"}}\n</tool_call>";

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    public void HermesTaggedCallSplitAcrossDeltas(int seed)
    {
        (string forwarded, List<NativeToolCall> calls) = ParserDriver.Drive(ToolCallFormat.Hermes, HermesTagged + " Bye.", seed);
        Assert.Equal("Sure. Bye.", forwarded);
        NativeToolCall call = Assert.Single(calls);
        Assert.Equal("call_0", call.Id);
        Assert.Equal("hang_up", call.Name);
        Assert.Equal("{\"reason\": \"done\"}", call.Arguments);
    }

    [Fact]
    public void HermesCallEndsAtTheBalancedObjectWhenTheClosingTagIsMissing()
    {
        (string forwarded, List<NativeToolCall> calls) = ParserDriver.Drive(ToolCallFormat.Hermes, "<tool_call>{\"name\": \"hang_up\", \"arguments\": {}}");
        Assert.Equal("", forwarded);
        Assert.Equal("hang_up", Assert.Single(calls).Name);
    }

    [Theory]
    [InlineData(1)]
    public void HermesBarePayloadWithMarkerTokensDroppedByTheDetokenizer(int seed)
    {
        // What the filter really sees for a Qwen3 GGUF: <tool_call> and </tool_call> are user-defined tokens and never reach the text.
        (string forwarded, List<NativeToolCall> calls) = ParserDriver.Drive(ToolCallFormat.Hermes, "\n{\"name\": \"hang_up\", \"arguments\": {}}\n", seed, maxPiece: 3);
        Assert.Equal("", forwarded.Trim());
        NativeToolCall call = Assert.Single(calls);
        Assert.Equal("hang_up", call.Name);
        Assert.Equal("{}", call.Arguments);
    }

    [Fact]
    public void TwoBareCallsOnSeparateLinesGetSequentialIds()
    {
        string text = "{\"name\": \"a\", \"arguments\": {\"x\": 1}}\n\n{\"name\": \"b\", \"arguments\": {\"y\": 2}}\n";
        (string forwarded, List<NativeToolCall> calls) = ParserDriver.Drive(ToolCallFormat.Hermes, text, 7);
        Assert.Equal("", forwarded.Trim());
        Assert.Equal(["call_0", "call_1"], calls.Select(c => c.Id));
        Assert.Equal(["a", "b"], calls.Select(c => c.Name));
    }

    [Fact]
    public void MalformedJsonInsideTagsIsForwardedAsText()
    {
        const string text = "<tool_call>{\"name\": \"x\", \"arguments\": {oops}}</tool_call>";
        (string forwarded, List<NativeToolCall> calls) = ParserDriver.Drive(ToolCallFormat.Hermes, text, 2);
        Assert.Empty(calls);
        Assert.Equal(text, forwarded);
    }

    [Fact]
    public void SpanPastTheCapIsForwardedAsText()
    {
        string text = "{\"name\": \"x\", \"arguments\": {\"blob\": \"" + new string('z', 200) + "\"}}";
        (string forwarded, List<NativeToolCall> calls) = ParserDriver.Drive(ToolCallFormat.Hermes, text, 1, maxPiece: 8, maxSpanChars: 64);
        Assert.Empty(calls);
        Assert.Equal(text, forwarded);
    }

    [Fact]
    public void OpenAiFunctionNestingIsUnwrapped()
    {
        (_, List<NativeToolCall> calls) = ParserDriver.Drive(ToolCallFormat.Hermes, "{\"function\": {\"name\": \"x\", \"arguments\": {\"a\": 1}}}");
        NativeToolCall call = Assert.Single(calls);
        Assert.Equal("x", call.Name);
        Assert.Equal("{\"a\": 1}", call.Arguments);
    }

    [Fact]
    public void JsonStringsContainingBracesAndTagsDoNotEndTheSpan()
    {
        const string text = "<tool_call>{\"name\": \"say\", \"arguments\": {\"text\": \"} </tool_call> \\\" {\"}}</tool_call>";
        (string forwarded, List<NativeToolCall> calls) = ParserDriver.Drive(ToolCallFormat.Hermes, text, 6);
        Assert.Equal("", forwarded);
        Assert.Equal("{\"text\": \"} </tool_call> \\\" {\"}", Assert.Single(calls).Arguments);
    }

    [Fact]
    public void ResetClearsIdsAndFlushKeepsThem()
    {
        ToolCallParser parser = new(ToolCallFormat.Hermes);
        parser.Push("{\"name\": \"a\", \"arguments\": {}}");
        parser.Flush();
        Assert.Equal(1, parser.CompletedCalls);
        Assert.Equal("call_1", parser.Push("{\"name\": \"b\", \"arguments\": {}}").Call!.Id);
        parser.Reset();
        Assert.Equal(0, parser.CompletedCalls);
        Assert.Equal("call_0", parser.Push("{\"name\": \"c\", \"arguments\": {}}").Call!.Id);
    }

    [Theory]
    [InlineData(1)]
    public void Llama3PythonTagAndParametersKey(int seed)
    {
        const string text = "<|python_tag|>{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}}";
        (string forwarded, List<NativeToolCall> calls) = ParserDriver.Drive(ToolCallFormat.Llama3, text, seed);
        Assert.Equal("", forwarded);
        NativeToolCall call = Assert.Single(calls);
        Assert.Equal("get_weather", call.Name);
        Assert.Equal("{\"city\": \"Paris\"}", call.Arguments);
    }

    [Theory]
    [InlineData(1)]
    public void MistralArrayYieldsEveryCall(int seed)
    {
        const string text = "[TOOL_CALLS][{\"name\": \"a\", \"arguments\": {\"x\": 1}}, {\"name\": \"b\", \"arguments\": {}}]";
        (string forwarded, List<NativeToolCall> calls) = ParserDriver.Drive(ToolCallFormat.Mistral, text, seed);
        Assert.Equal("", forwarded);
        Assert.Equal(["call_0", "call_1"], calls.Select(c => c.Id));
        Assert.Equal(["a", "b"], calls.Select(c => c.Name));
        Assert.Equal("{\"x\": 1}", calls[0].Arguments);
    }

    [Fact]
    public void MistralBareArrayWhenTheMarkerTokenWasDropped()
    {
        (string forwarded, List<NativeToolCall> calls) = ParserDriver.Drive(ToolCallFormat.Mistral, "[{\"name\": \"a\", \"arguments\": {}}]", 4);
        Assert.Equal("", forwarded);
        Assert.Equal("a", Assert.Single(calls).Name);
    }

    [Theory]
    [InlineData(1)]
    public void MistralNamedFormAtLineStart(int seed)
    {
        (string forwarded, List<NativeToolCall> calls) = ParserDriver.Drive(ToolCallFormat.Mistral, "get_weather{\"city\": \"Paris\"}", seed);
        Assert.Equal("", forwarded);
        NativeToolCall call = Assert.Single(calls);
        Assert.Equal("get_weather", call.Name);
        Assert.Equal("{\"city\": \"Paris\"}", call.Arguments);
    }

    [Fact]
    public void MistralProseAndMarkdownListsAreNotCalls()
    {
        const string text = "Paris is lovely.\n[x] done\n[ ] todo\n";
        (string forwarded, List<NativeToolCall> calls) = ParserDriver.Drive(ToolCallFormat.Mistral, text, 5);
        Assert.Empty(calls);
        Assert.Equal(text, forwarded);
    }

    [Theory]
    [InlineData(1)]
    public void GemmaBlockWithMarkersAndQuoteTokens(int seed)
    {
        const string text = "<|tool_call>call:get_weather{city:<|\"|>Paris, FR<|\"|>,days:3,metric:true}<tool_call|>";
        (string forwarded, List<NativeToolCall> calls) = ParserDriver.Drive(ToolCallFormat.Gemma, text, seed);
        Assert.Equal("", forwarded);
        NativeToolCall call = Assert.Single(calls);
        Assert.Equal("get_weather", call.Name);
        Assert.Equal("{\"city\":\"Paris, FR\",\"days\":3,\"metric\":true}", call.Arguments);
    }

    [Fact]
    public void GemmaNestedObjectsArraysAndEmptyArguments()
    {
        (_, List<NativeToolCall> calls) = ParserDriver.Drive(ToolCallFormat.Gemma, "call:f{opts:{a:[1,2.5],b:<|\"|>x,y<|\"|>,n:null}}\ncall:hang_up{}", 8);
        Assert.Equal(2, calls.Count);
        Assert.Equal("{\"opts\":{\"a\":[1,2.5],\"b\":\"x,y\",\"n\":null}}", calls[0].Arguments);
        Assert.Equal("hang_up", calls[1].Name);
        Assert.Equal("{}", calls[1].Arguments);
    }

    [Theory]
    [InlineData(null, ToolCallFormat.Hermes)]
    [InlineData("Qwen3-4B-Q4_K_M.gguf", ToolCallFormat.Hermes)]
    [InlineData("llama-3.2-1b-instruct-q8_0.gguf", ToolCallFormat.Llama3)]
    [InlineData("Mistral-7B-Instruct-v0.3-Q4_K_M.gguf", ToolCallFormat.Mistral)]
    [InlineData("gemma-4-E2B-it", ToolCallFormat.Gemma)]
    [InlineData("{%- if tools %}[AVAILABLE_TOOLS]{{ tools }}[/AVAILABLE_TOOLS]{%- endif %}[TOOL_CALLS]", ToolCallFormat.Mistral)]
    [InlineData("deepseek-v3.1", ToolCallFormat.Hermes)] // not "...-qwen-..." — would hit the Qwen branch instead
    public void DetectPicksTheFamilyFormat(string? hint, ToolCallFormat expected)
        => Assert.Equal(expected, ToolCallFormats.Detect(hint));


    [Theory]
    [InlineData(1)]
    public void BareObjectNamingAnUnknownToolIsTextWhenTheOfferedToolsAreKnown(int seed)
    {
        const string text = "Here is the record:\n{\"name\": \"Bob\", \"age\": 3}\nand {\"name\": \"hang_up\", \"arguments\": {}}";
        (string forwarded, List<NativeToolCall> calls) = ParserDriver.Drive(ToolCallFormat.Hermes, text, seed, knownTools: ["hang_up"]);
        Assert.Equal("Here is the record:\n{\"name\": \"Bob\", \"age\": 3}\nand ", forwarded);
        Assert.Equal("hang_up", Assert.Single(calls).Name);
    }

    [Fact]
    public void LineStartObjectNotOpeningWithNameIsReleasedAtItsFirstKey()
    {
        ToolCallParser parser = new(ToolCallFormat.Hermes);
        const string json = "{\"answer\": 42, \"unit\":";
        Assert.Equal(json, parser.Push(json).ForwardText);
        Assert.False(parser.InCall);
        const string code = "\n{\n  int x = 1;\n";
        Assert.Equal(code, parser.Push(code).ForwardText);
        Assert.False(parser.InCall);
        Assert.Equal("}", parser.Push("}").ForwardText);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void MistralNamedFormOpensOnlyOnACompleteKnownName(int seed)
    {
        const string text = "get{\"city\": \"Paris\"}\nget_weather{\"city\": \"Rome\"}";
        (string forwarded, List<NativeToolCall> calls) = ParserDriver.Drive(ToolCallFormat.Mistral, text, seed, knownTools: ["get_weather"]);
        Assert.Equal("get{\"city\": \"Paris\"}\n", forwarded);
        NativeToolCall call = Assert.Single(calls);
        Assert.Equal("get_weather", call.Name);
        Assert.Equal("{\"city\": \"Rome\"}", call.Arguments);
    }

    [Theory]
    [InlineData(ToolCallFormat.Hermes, "{\"na me\": \"x\", ")]
    [InlineData(ToolCallFormat.Mistral, "[{\"na me\": \"x\", ")]
    public void WhitespaceInsideTheProbedKeyReleasesTheSpanAtOnce(ToolCallFormat format, string text)
    {
        ToolCallParser parser = new(format);
        ToolCallParseResult result = parser.Push(text);
        Assert.False(parser.InCall);
        Assert.Null(result.Calls);
        Assert.Equal(text, result.ForwardText);
    }

}
