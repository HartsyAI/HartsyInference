using HartsyInference.LLM.OutputParsing;

namespace HartsyInference.LLM.Tests.OutputParsing;

internal static class ParserTestHelpers
{
    public static DeepSeekV41OutputParser NewParser(PieceTokenizer tokenizer, bool thinking) =>
        new(tokenizer, thinking ? OutputParserState.Reasoning : OutputParserState.Content);

    /// <summary>Compares a parse against the encoding.py result and the event replay against the parse.</summary>
    public static void AssertMatches(ParserCase expected, ParsedAssistant got, ReplayedTurn replay, string where)
    {
        Xunit.Assert.True(expected.Reasoning == got.Reasoning, $"{expected.Name} {where}: reasoning '{got.Reasoning}' != '{expected.Reasoning}'");
        Xunit.Assert.True(expected.Content == got.Content, $"{expected.Name} {where}: content '{got.Content}' != '{expected.Content}'");
        Xunit.Assert.True(expected.Calls.Count == got.ToolCalls.Count,
            $"{expected.Name} {where}: {got.ToolCalls.Count} calls != {expected.Calls.Count}");
        for (int i = 0; i < expected.Calls.Count; i++)
        {
            Xunit.Assert.Equal(expected.Calls[i].Name, got.ToolCalls[i].Name);
            Xunit.Assert.Equal(expected.Calls[i].Namespace, got.ToolCalls[i].Namespace);
            Xunit.Assert.Equal(expected.Calls[i].Arguments, got.ToolCalls[i].ArgumentsJson);
        }
        Xunit.Assert.False(got.Malformed, $"{expected.Name} {where}: flagged malformed");
        AssertReplayMatches(got, replay, $"{expected.Name} {where}");
    }

    public static void AssertReplayMatches(ParsedAssistant got, ReplayedTurn replay, string where)
    {
        Xunit.Assert.True(got.Reasoning == replay.Reasoning.ToString(), $"{where}: reasoning deltas differ from Result");
        Xunit.Assert.True(got.Content == replay.Content.ToString(), $"{where}: content deltas differ from Result");
        Xunit.Assert.True(got.ToolCalls.Count == replay.Calls.Count(c => c.Ended), $"{where}: ended calls differ from Result");
        for (int i = 0; i < got.ToolCalls.Count; i++)
        {
            Xunit.Assert.True(got.ToolCalls[i].ArgumentsJson == replay.Calls[i].Args.ToString(), $"{where}: args deltas of call {i} differ");
            Xunit.Assert.True(replay.Calls[i].Ended, $"{where}: call {i} never ended");
        }
        Xunit.Assert.Equal(1, replay.Stops);
        Xunit.Assert.Equal(got.Malformed, replay.Malformed > 0);
    }
}
