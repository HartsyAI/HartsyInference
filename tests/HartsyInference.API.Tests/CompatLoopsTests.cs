using System.Text.Json;
using HartsyInference.API.Endpoints;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>The <c>loops</c> body field of the OpenAI-compatible chat route: the recurrent-block pass count a looped model (Dory) runs for the request.</summary>
public sealed class CompatLoopsTests
{
    [Fact]
    public void Loops_IsReadFromTheRequestBody()
    {
        ChatCompletionRequest req = JsonSerializer.Deserialize<ChatCompletionRequest>("""{"model":"dory","messages":[],"loops":3}""")!;
        Assert.Equal(3, req.Loops);
        Assert.Equal(3, CompatEndpoints.ParseLoops(req.Loops));
    }

    [Fact]
    public void Loops_AbsentLeavesTheCheckpointDefault()
    {
        ChatCompletionRequest req = JsonSerializer.Deserialize<ChatCompletionRequest>("""{"model":"dory","messages":[]}""")!;
        Assert.Null(req.Loops);
        Assert.Null(CompatEndpoints.ParseLoops(req.Loops));
    }

    [Fact]
    public void Loops_BelowOne_IsRefused()
    {
        Assert.Throws<ArgumentException>(() => CompatEndpoints.ParseLoops(0));
    }
}
