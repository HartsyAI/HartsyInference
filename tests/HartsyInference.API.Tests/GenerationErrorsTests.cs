using HartsyInference.API.Endpoints;
using HartsyInference.LLM.Generation;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Http.HttpResults;
using Microsoft.Extensions.DependencyInjection;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>The status the generation routes return for a request the scheduler could not finish. A stopped scheduler means its model was unloaded or
/// reloaded: the client should retry (503), not read it as a server fault (500).</summary>
public sealed class GenerationErrorsTests
{
    [Fact]
    public void A_Stopped_Scheduler_Maps_To_503_Service_Unavailable()
    {
        IResult result = GenerationErrors.Map(new SchedulerStoppedException());
        Assert.Equal(StatusCodes.Status503ServiceUnavailable, Assert.IsAssignableFrom<IStatusCodeHttpResult>(result).StatusCode);
    }

    [Fact]
    public async Task A_Full_Scheduler_Queue_Maps_To_429_With_Retry_After()
    {
        IResult result = GenerationErrors.Map(new SchedulerQueueFullException(4));
        // The problem body is written through the app's JSON options, so the context needs them as services.
        ServiceCollection services = new();
        services.AddOptions();
        services.AddLogging();
        services.Configure<Microsoft.AspNetCore.Http.Json.JsonOptions>(_ => { });
        DefaultHttpContext ctx = new() { RequestServices = services.BuildServiceProvider() };

        await result.ExecuteAsync(ctx);

        Assert.Equal(StatusCodes.Status429TooManyRequests, ctx.Response.StatusCode);
        Assert.Equal("1", ctx.Response.Headers["Retry-After"]);
    }
}
