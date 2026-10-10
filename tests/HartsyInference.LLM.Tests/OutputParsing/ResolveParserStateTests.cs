using HartsyInference.LLM.ChatTemplates;
using Xunit;

namespace HartsyInference.LLM.Tests.OutputParsing;

public sealed class ResolveParserStateTests
{
    private readonly DeepSeekV41Encoder _encoder = new();

    [Theory]
    [InlineData(true, true)]
    [InlineData(false, false)]
    public void ThinkingModeDecidesWhetherTheCompletionStartsInReasoning(bool thinking, bool reasoningOpen)
    {
        List<ChatMessage> messages = [new ChatMessage("user", "hi")];
        ParserInitialState state = _encoder.ResolveParserState(messages, new EncodeOptions { Thinking = thinking });
        Assert.Equal(reasoningOpen, state.ReasoningOpen);
    }

}
