using System.Text.Json;
using HartsyInference.API;
using HartsyInference.API.Endpoints;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Http.HttpResults;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>What <c>/ready?model=</c> answers: 200 only while a deployment of the model is Ready, 503 otherwise. The <c>hartsy.status</c> frame is a named SSE event, so
/// a standard OpenAI stream is unchanged.</summary>
public sealed class HealthReadyTests
{
    /// <summary>A text service that reports only the deployments it is given. Nothing else is used by the readiness check.</summary>
    private sealed class DeploymentsOnlyText(IReadOnlyList<DeploymentStatus> deployments) : ITextService
    {
        public IReadOnlyList<DeploymentStatus> Deployments => deployments;
        public Task<TextResult> GenerateAsync(ModelSpec spec, TextRequest request, CancellationToken cancel = default) => throw new NotSupportedException();
        public IAsyncEnumerable<TextChunk> StreamAsync(ModelSpec spec, TextRequest request, CancellationToken cancel = default) => throw new NotSupportedException();
        public int CountTokens(ModelSpec spec, string text) => 0;
        public bool Unload(string? device = null) => false;
    }

    private static int StatusOf(IResult result) => Assert.IsAssignableFrom<IStatusCodeHttpResult>(result).StatusCode
        ?? throw new InvalidOperationException("the readiness result carries no status code");

    [Fact]
    public void A_Ready_Deployment_Of_The_Model_Answers_200()
    {
        DeploymentsOnlyText text = new([new DeploymentStatus("chat", "llm-a", "cpu", DeploymentState.Ready, null)]);

        Assert.Equal(StatusCodes.Status200OK, StatusOf(HealthEndpoints.ReadyFor("llm-a", text)));
    }

    [Fact]
    public void A_Deployment_Still_Loading_Answers_503()
    {
        DeploymentsOnlyText text = new([new DeploymentStatus("chat", "llm-a", "cpu", DeploymentState.Loading, null)]);

        Assert.Equal(StatusCodes.Status503ServiceUnavailable, StatusOf(HealthEndpoints.ReadyFor("llm-a", text)));
    }

    [Fact]
    public void A_Model_With_No_Deployment_Answers_503()
    {
        DeploymentsOnlyText text = new([]);

        Assert.Equal(StatusCodes.Status503ServiceUnavailable, StatusOf(HealthEndpoints.ReadyFor("llm-a", text)));
    }

    [Fact]
    public void A_Ready_Deployment_Is_Preferred_Over_An_Older_Loading_One_Of_The_Same_Model()
    {
        DeploymentsOnlyText text = new(
        [
            new DeploymentStatus("chat-old", "llm-a", "cpu", DeploymentState.Loading, null),
            new DeploymentStatus("chat-new", "llm-a", "cpu", DeploymentState.Ready, null),
        ]);

        Assert.Equal(StatusCodes.Status200OK, StatusOf(HealthEndpoints.ReadyFor("llm-a", text)));
    }

    [Fact]
    public void A_Newer_Loading_Deployment_Is_Reported_Over_A_Stale_Unloaded_One()
    {
        DeploymentsOnlyText text = new(
        [
            new DeploymentStatus("chat-old", "llm-a", "cpu", DeploymentState.Unloaded, null),
            new DeploymentStatus("chat-new", "llm-a", "cpu", DeploymentState.Loading, null),
        ]);

        IResult result = HealthEndpoints.ReadyFor("llm-a", text);

        // Both are 503; the stale record used to answer, hiding the load in progress.
        Assert.Equal(StatusCodes.Status503ServiceUnavailable, StatusOf(result));
        JsonElement body = JsonSerializer.SerializeToElement(Assert.IsAssignableFrom<IValueHttpResult>(result).Value);
        Assert.Equal("chat-new", body.GetProperty("deployment_id").GetString());
        Assert.Equal("loading", body.GetProperty("state").GetString());
    }
}

/// <summary>The frame a streamed reply sends for a queued request: a named <c>hartsy.status</c> event, whose data is the status in snake_case.</summary>
public sealed class HartsyStatusFrameTests
{
    [Fact]
    public void A_Queued_Status_Is_A_Named_Frame_With_Its_Queue_Position()
    {
        string frame = SseHelpers.Event("hartsy.status", HartsyStatusDto.From(new TextStatus("queued", 2)), new JsonSerializerOptions(JsonSerializerDefaults.Web));

        Assert.Equal("event: hartsy.status\ndata: {\"phase\":\"queued\",\"queue_position\":2,\"prefill_done\":0,\"prefill_total\":0}\n\n", frame);
    }
}
