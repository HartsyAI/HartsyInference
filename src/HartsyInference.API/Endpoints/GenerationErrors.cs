using System.Globalization;
using HartsyInference.Core.Exceptions;
using HartsyInference.Engine;
using HartsyInference.LLM.Generation;

namespace HartsyInference.API.Endpoints;

/// <summary>Maps the exceptions <see cref="HartsyInference.Engine.Services.IInferenceEngine"/>'s typed services can throw to HTTP status codes, shared by every generation route's non-streaming path.</summary>
internal static class GenerationErrors
{
    public static IResult Map(Exception ex) => ex switch
    {
        QueueFullException => WithRetryAfter(HartsyInferenceServiceExtensions.Problem(StatusCodes.Status429TooManyRequests, ex.Message, "rate_limit_error")),
        // The scheduler's waiting queue is full (continuous batching): the same answer the server's own queue gives.
        SchedulerQueueFullException => WithRetryAfter(HartsyInferenceServiceExtensions.Problem(StatusCodes.Status429TooManyRequests, ex.Message, "rate_limit_error")),
        // The scheduler stopped (the model was unloaded or reloaded) while this request was queued or running: retryable, not a server fault.
        SchedulerStoppedException => HartsyInferenceServiceExtensions.Problem(StatusCodes.Status503ServiceUnavailable, ex.Message, "server_error"),
        // No checkpoint resolved for the requested model — a client input problem, not a server fault. The image
        // path throws FileNotFoundException for this; the text path throws HartsyInferenceException for the same
        // condition (see TextService.LoadInto) — both are "you asked for a model with no resolvable checkpoint".
        FileNotFoundException => HartsyInferenceServiceExtensions.Problem(StatusCodes.Status400BadRequest, ex.Message, "invalid_request_error"),
        HartsyInferenceException => HartsyInferenceServiceExtensions.Problem(StatusCodes.Status400BadRequest, ex.Message, "invalid_request_error"),
        // Checkpoint resolved but no recipe is lifted into the Engine for that family yet.
        NotSupportedException => HartsyInferenceServiceExtensions.Problem(StatusCodes.Status501NotImplemented, ex.Message, "invalid_request_error"),
        // A malformed request.settings block: unknown id, unparsable value, or a setting that binds at model load
        // and so cannot be changed per request. All three are the caller's to fix.
        ArgumentException => HartsyInferenceServiceExtensions.Problem(StatusCodes.Status400BadRequest, ex.Message, "invalid_request_error"),
        _ => HartsyInferenceServiceExtensions.Problem(StatusCodes.Status500InternalServerError, ex.Message, "server_error"),
    };

    /// <summary>How long a client should wait before retrying a full queue, in seconds.</summary>
    private const int RetryAfterSeconds = 1;

    /// <summary>Adds a <c>Retry-After</c> header to an error result, so a client knows roughly when a full queue may have room.</summary>
    private static IResult WithRetryAfter(IResult inner) => new RetryAfterResult(inner);

    private sealed class RetryAfterResult(IResult inner) : IResult
    {
        public Task ExecuteAsync(HttpContext httpContext)
        {
            httpContext.Response.Headers["Retry-After"] = RetryAfterSeconds.ToString(CultureInfo.InvariantCulture);
            return inner.ExecuteAsync(httpContext);
        }
    }
}
