using HartsyInference.LLM.Multimodal;
using Xunit;

namespace HartsyInference.LLM.Tests.Ssm;

/// <summary>The turn the Qwen3.5-VL prompt ends with: with --no-thinking the shipped template's empty think block follows the assistant header.</summary>
public sealed class Qwen35VlAssistantTurnTests
{
    [Fact]
    public void ThinkingUnsetOrOnLeavesTheAssistantTurnOpenForTheModelsOwnReasoning()
    {
        Assert.Equal("<|im_end|>\n<|im_start|>assistant\n", Qwen35VlGenerator.AssistantTurn(null));
        Assert.Equal("<|im_end|>\n<|im_start|>assistant\n", Qwen35VlGenerator.AssistantTurn(true));
    }

    [Fact]
    public void ThinkingOffClosesAnEmptyThinkBlockAfterTheAssistantHeader()
    {
        Assert.Equal("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", Qwen35VlGenerator.AssistantTurn(false));
    }
}
