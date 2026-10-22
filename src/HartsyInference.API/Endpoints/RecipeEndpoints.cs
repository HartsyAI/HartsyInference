using HartsyInference.Engine;
using HartsyInference.Engine.Catalog;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Services;

namespace HartsyInference.API.Endpoints;

/// <summary>Native recipe catalog: the defaults and capabilities each registered family declares, answered without
/// loading weights or constructing a pipeline. Hosts query these routes instead of re-declaring the values.</summary>
public static class RecipeEndpoints
{
    /// <summary>Maps <c>/v1/native/recipes</c>.</summary>
    public static void MapRecipeEndpoints(this WebApplication app)
    {
        app.MapGet("/v1/native/recipes", IResult (string? modality, IInferenceEngine engine) =>
        {
            Modality? filter = null;
            if (modality is not null)
            {
                if (!TryParseModality(modality, out Modality parsed))
                {
                    return InvalidModality(modality);
                }
                filter = parsed;
            }
            return Results.Json<IReadOnlyList<ModelDescriptor>>(engine.Recipes.All(filter));
        });

        app.MapGet("/v1/native/recipes/{modality}/{familyId}", IResult (string modality, string familyId, string? variant,
            IInferenceEngine engine) =>
        {
            if (!TryParseModality(modality, out Modality parsed))
            {
                return InvalidModality(modality);
            }
            try
            {
                return Results.Json<ModelDescriptor>(engine.Recipes.DescribeFamily(parsed, familyId, variant));
            }
            catch (RecipeNotRegisteredException ex)
            {
                return HartsyInferenceServiceExtensions.Problem(StatusCodes.Status404NotFound, ex.Message, "not_found_error");
            }
            catch (ArgumentException ex)
            {
                return HartsyInferenceServiceExtensions.Problem(StatusCodes.Status400BadRequest, ex.Message, "invalid_request_error");
            }
        });

        app.MapPost("/v1/native/recipes/describe", IResult (NativeRecipeDescribeRequest req, IInferenceEngine engine) =>
        {
            if (!TryParseModality(req.Modality, out Modality modality))
            {
                return InvalidModality(req.Modality);
            }
            try
            {
                ModelSpec spec = ModelResolver.Resolve(req.Model, req.ModelPath, modality);
                return Results.Json<ModelDescriptor>(engine.Recipes.Describe(spec));
            }
            // Ahead of the input-failure filter: it keeps its own issue code even if its base type changes.
            catch (RecipeNotRegisteredException ex)
            {
                return Results.Json(UnregisteredFamily(ex), statusCode: StatusCodes.Status422UnprocessableEntity);
            }
            catch (Exception ex) when (VideoEndpoints.IsPlanningInputFailure(ex))
            {
                return Results.Json(ModelFailure(ex), statusCode: StatusCodes.Status422UnprocessableEntity);
            }
        });
    }

    /// <summary>Accepts the two modalities the catalog describes, case-insensitively.</summary>
    private static bool TryParseModality(string value, out Modality modality)
    {
        if (string.Equals(value, "image", StringComparison.OrdinalIgnoreCase))
        {
            modality = Modality.Image;
            return true;
        }
        if (string.Equals(value, "video", StringComparison.OrdinalIgnoreCase))
        {
            modality = Modality.Video;
            return true;
        }
        modality = default;
        return false;
    }

    private static IResult InvalidModality(string value) =>
        HartsyInferenceServiceExtensions.Problem(StatusCodes.Status400BadRequest,
            $"Unknown modality '{value}'; use 'image' or 'video'.", "invalid_request_error");

    /// <summary>The typed 422 for a family that resolves but has no recipe for the requested modality.</summary>
    private static NativeRecipeProblem UnregisteredFamily(RecipeNotRegisteredException exception) => new()
    {
        Issues =
        [
            NativeVideoPlanResponse.SanitizeIssue(new VideoPlanIssue
            {
                Code = "recipe.family.unregistered",
                Severity = VideoPlanIssueSeverity.Error,
                Message = exception.Message,
                Field = nameof(ModelSpec.Requested),
            }),
        ],
    };

    /// <summary>The typed 422 for a model whose checkpoint cannot be read, with its paths removed from the message.</summary>
    private static NativeRecipeProblem ModelFailure(Exception exception) => new()
    {
        Issues =
        [
            NativeVideoPlanResponse.SanitizeIssue(new VideoPlanIssue
            {
                Code = "recipe.model_unresolvable",
                Severity = VideoPlanIssueSeverity.Error,
                Message = exception.Message,
                Field = nameof(NativeRecipeDescribeRequest.Model),
            }),
        ],
    };
}
