using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using Xunit;

namespace HartsyInference.Engine.Tests.Services;

/// <summary>How a vision answer's stop reason and reasoning are read, shared by the Qwen3.5-VL, Mllama and generic vision paths.</summary>
public sealed class VisionAnswerTests
{
    [Fact]
    public void AStopTokenIsANaturalStopAndASpentBudgetIsALengthStop()
    {
        Assert.Equal(StopReason.Stop, TextService.VisionStopReason(stoppedOnToken: true));
        Assert.Equal(StopReason.Length, TextService.VisionStopReason(stoppedOnToken: false));
    }

    [Fact]
    public void ThinkingOffKeepsTheWholeAnswerAsContent()
    {
        (string reasoning, string content) = TextService.SplitVisionReasoning("<think>\nplan\n</think>\n\nAnswer.", enableThinking: false, stoppedOnToken: true);
        Assert.Equal("", reasoning);
        Assert.Equal("<think>\nplan\n</think>\n\nAnswer.", content);
    }

    [Fact]
    public void ThinkingOnSplitsTheReasoningFromTheContent()
    {
        (string reasoning, string content) = TextService.SplitVisionReasoning("<think>\nThe bus is blue.\n</think>\n\nA blue bus on a street.", enableThinking: null, stoppedOnToken: true);
        Assert.Equal("The bus is blue.", reasoning);
        Assert.Equal("A blue bus on a street.", content);
    }

    [Fact]
    public void ThinkingOnWithNoMarkerAndANaturalStopIsAllContent()
    {
        (string reasoning, string content) = TextService.SplitVisionReasoning("A blue bus.", enableThinking: null, stoppedOnToken: true);
        Assert.Equal("", reasoning);
        Assert.Equal("A blue bus.", content);
    }

    [Fact]
    public void ThinkingOnThatRunsOutInsideTheReasoningBlockIsAllReasoning()
    {
        // The budget ran out before </think>: no answer was given, so nothing is passed off as content.
        (string reasoning, string content) = TextService.SplitVisionReasoning("<think>\nLet me describe the bus", enableThinking: null, stoppedOnToken: false);
        Assert.Equal("Let me describe the bus", reasoning);
        Assert.Equal("", content);
    }
}
