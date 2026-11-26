using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using Xunit;

namespace HartsyInference.Engine.Tests.Services;

/// <summary>How a vision answer's stop reason and reasoning are read. The split only applies when the prompt opened a reasoning block.</summary>
public sealed class VisionAnswerTests
{
    [Fact]
    public void AStopTokenIsANaturalStopAndASpentBudgetIsALengthStop()
    {
        Assert.Equal(StopReason.Stop, TextService.VisionStopReason(stoppedOnToken: true));
        Assert.Equal(StopReason.Length, TextService.VisionStopReason(stoppedOnToken: false));
    }

    [Fact]
    public void WithoutAnOpenReasoningBlockTheWholeAnswerIsContent()
    {
        (string reasoning, string content) = TextService.SplitVisionReasoning("<think>\nplan\n</think>\n\nAnswer.", reasoningOpen: false, stoppedOnToken: true);
        Assert.Equal("", reasoning);
        Assert.Equal("<think>\nplan\n</think>\n\nAnswer.", content);
    }

    [Fact]
    public void ANonReasoningModelThatSpendsItsBudgetKeepsItsAnswerAsContent()
    {
        // The regression: the generic and Mllama paths never open a reasoning block, so a cut-off answer must not be sent as reasoning.
        (string reasoning, string content) = TextService.SplitVisionReasoning("A blue bus on a stre", reasoningOpen: false, stoppedOnToken: false);
        Assert.Equal("", reasoning);
        Assert.Equal("A blue bus on a stre", content);
    }

    [Fact]
    public void AnOpenReasoningBlockIsSplitAtItsClosingMarker()
    {
        (string reasoning, string content) = TextService.SplitVisionReasoning("The bus is blue.\n</think>\n\nA blue bus on a street.", reasoningOpen: true, stoppedOnToken: true);
        Assert.Equal("The bus is blue.", reasoning);
        Assert.Equal("A blue bus on a street.", content);
    }

    [Fact]
    public void AnOpenBlockWithNoMarkerAndANaturalStopIsAllContent()
    {
        (string reasoning, string content) = TextService.SplitVisionReasoning("A blue bus.", reasoningOpen: true, stoppedOnToken: true);
        Assert.Equal("", reasoning);
        Assert.Equal("A blue bus.", content);
    }

    [Fact]
    public void AnOpenBlockThatRunsOutBeforeItsMarkerIsAllReasoning()
    {
        (string reasoning, string content) = TextService.SplitVisionReasoning("Let me describe the bus", reasoningOpen: true, stoppedOnToken: false);
        Assert.Equal("Let me describe the bus", reasoning);
        Assert.Equal("", content);
    }
}
