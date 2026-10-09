using HartsyInference.API.Endpoints;
using HartsyInference.LLM.Generation;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Http.HttpResults;
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
}
