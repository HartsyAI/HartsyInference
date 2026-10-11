using HartsyInference.Engine.Requests;
using HartsyInference.Tools.Parsing;
using Xunit;

namespace HartsyInference.Tools.Tests.Parsing;

/// <summary>Verbatim completions captured from real checkpoints (the decoded text the filter saw, markers included), parsed at many split points. A regression here means a real model's output stopped parsing.</summary>
public sealed class CapturedOutputParserTests
{
    private static string Captured(string name) => File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", "CapturedOutput", name));

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(7)]
    [InlineData(42)]
    public void Qwen3SmallMarkerFormCompletionParsesToOneHangUp(int seed)
    {
        string completion = Captured("qwen3-0.6b-markers.txt");
        (string forwarded, List<NativeToolCall> calls) = ParserDriver.Drive(ToolCallFormat.Hermes, completion, seed, maxPiece: 5, knownTools: ["hang_up"]);
        NativeToolCall call = Assert.Single(calls);
        Assert.Equal("hang_up", call.Name);
        Assert.Equal("{}", call.Arguments.Replace(" ", "").Replace("\n", ""));
        Assert.Equal("", forwarded.Trim());
    }

    [Fact]
    public void MarkerLiteralsAreWhatTheFilterAsksTheEngineToSurface()
    {
        // The engine decodes exactly these as text; the completion above contains each of them.
        IReadOnlyCollection<string> literals = ToolCallFormats.RulesFor(ToolCallFormat.Hermes).MarkerLiterals;
        Assert.Contains("<tool_call>", literals);
        Assert.Contains("</tool_call>", literals);
        Assert.DoesNotContain("{\"name\"", literals);
        Assert.Contains("<tool_call>", Captured("qwen3-0.6b-markers.txt"));
    }
}
