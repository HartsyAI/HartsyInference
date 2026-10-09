using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.LLM.Generation;
using Xunit;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>Pre-tokenized ids on a text request reach the engine request unchanged and win over the messages.</summary>
public sealed class TextRequestRawTokenIdsTests
{
    // The raw path never reads the tokenizer, so none is supplied.
    [Fact]
    public void RawTokenIds_ReplaceTheMessagesOnTheEngineRequest()
    {
        TextRequest request = new()
        {
            Messages = [new TextMessage { Role = TextRole.User, Content = "The capital of France is" }],
            RawTokenIds = [0, 671, 6102],
            MaxTokens = 17,
        };

        GenerationRequest engine = TextService.BuildRequest(request, rawCompletion: false, tokenizer: null!);

        Assert.Equal(new[] { 0, 671, 6102 }, engine.RawTokenIds);
        Assert.Null(engine.Messages);
        Assert.Equal(17, engine.MaxTokens);
    }

    [Fact]
    public void WithoutRawTokenIds_TheMessagesAreKept()
    {
        TextRequest request = new()
        {
            Messages = [new TextMessage { Role = TextRole.User, Content = "hi" }],
            MaxTokens = 4,
        };

        GenerationRequest engine = TextService.BuildRequest(request, rawCompletion: false, tokenizer: null!);

        Assert.Null(engine.RawTokenIds);
        Assert.NotNull(engine.Messages);
    }
}
