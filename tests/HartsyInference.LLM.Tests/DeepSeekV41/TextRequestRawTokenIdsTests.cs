using HartsyInference.Core.Exceptions;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
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

    [Fact]
    public async Task EmptyRawTokenIds_AreRefusedBeforeAnyModelWork()
    {
        using InferenceEngine engine = new("cpu");
        TextRequest request = new() { Messages = [], RawTokenIds = [], MaxTokens = 4 };

        HartsyInferenceException ex = await Assert.ThrowsAsync<HartsyInferenceException>(() => engine.Text.GenerateAsync(MissingModel(), request));

        Assert.Contains("RawTokenIds", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task EmptyRawTokenIds_EndTheStreamWithAnErrorStop()
    {
        using InferenceEngine engine = new("cpu");
        TextRequest request = new() { Messages = [], RawTokenIds = [], MaxTokens = 4 };
        List<TextChunk> chunks = [];

        await foreach (TextChunk chunk in engine.Text.StreamAsync(MissingModel(), request))
            chunks.Add(chunk);

        // The stream pump reports a failure as a terminal error stop, so the refusal arrives as that stop's text.
        TextChunk terminal = Assert.Single(chunks);
        Assert.Equal(TextChunkKind.StopReason, terminal.Kind);
        Assert.Equal(StopReason.Error, terminal.Stop);
        Assert.Contains("RawTokenIds", terminal.Text, StringComparison.Ordinal);
    }

    /// <summary>A path that does not exist: a refusal that names RawTokenIds shows the request was rejected before the model was read.</summary>
    private static ModelSpec MissingModel() => new() { Requested = "missing", Modality = Modality.Text, LocalPath = "/nonexistent/no-such-model" };
}
