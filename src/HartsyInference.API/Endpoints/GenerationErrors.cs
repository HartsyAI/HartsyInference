using HartsyInference.Core.Exceptions;
using HartsyInference.Engine;

namespace HartsyInference.API.Endpoints;

/// <summary>Maps the exceptions <see cref="HartsyInference.Engine.Services.IInferenceEngine"/>'s typed services
/// can throw to HTTP status codes, shared by every generation route's non-streaming path.</summary>
internal static class GenerationErrors
{
    public static IResult Map(Exception ex) => ex switch
    {
        QueueFullException => HartsyInferenceServiceExtensions.Problem(StatusCodes.Status429TooManyRequests, ex.Message, "rate_limit_error"),
        // No checkpoint resolved for the requested model — a client input problem, not a server fault. The image
        // path throws FileNotFoundException for this; the text path throws HartsyInferenceException for the same
        // condition (see TextService.LoadInto) — both are "you asked for a model with no resolvable checkpoint".
        FileNotFoundException => HartsyInferenceServiceExtensions.Problem(StatusCodes.Status400BadRequest, ex.Message, "invalid_request_error"),
        HartsyInferenceException => HartsyInferenceServiceExtensions.Problem(StatusCodes.Status400BadRequest, ex.Message, "invalid_request_error"),
        // Checkpoint resolved but no recipe is lifted into the Engine for that family yet.
        NotSupportedException => HartsyInferenceServiceExtensions.Problem(StatusCodes.Status501NotImplemented, ex.Message, "invalid_request_error"),
        _ => HartsyInferenceServiceExtensions.Problem(StatusCodes.Status500InternalServerError, ex.Message, "server_error"),
    };
}
