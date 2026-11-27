using System.Text.Json;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.IO;
using HartsyInference.Engine;
using HartsyInference.Engine.Recipes.Video;

namespace HartsyInference.API.Endpoints;

/// <summary>Over HTTP a Duo run folder must lie under the models root, the rule <see cref="PackageListing"/> applies to the directories a
/// client may name. The engine and the command line take any folder.</summary>
internal static class DuoRunFolderGate
{
    /// <summary>Replaces the request's run folder with its full path under the models root (a relative one is taken under it). Returns the
    /// 400 to answer with when the value is not a folder name under that root, else null. The message repeats nothing of the server path.</summary>
    public static IResult? Confine(NativeVideoRequest request)
    {
        if (request.Request?.Extra is null || !request.Request.Extra.TryGetValue(DuoRecipePipeline.RunFolderKey, out object? value))
        {
            return null;
        }
        string? folder = value switch
        {
            string text => text,
            JsonElement { ValueKind: JsonValueKind.String } element => element.GetString(),
            _ => null,
        };
        if (string.IsNullOrWhiteSpace(folder))
        {
            return Refuse("must be a folder name.");
        }
        string modelsRoot = Path.GetFullPath(RepoPaths.ModelsRoot());
        string full;
        try
        {
            full = Path.GetFullPath(folder, modelsRoot);
        }
        catch (ArgumentException)
        {
            return Refuse("is not a valid path.");
        }
        if (!PackageListing.IsUnder(full, modelsRoot))
        {
            return Refuse("is outside the models root; over HTTP a run folder must be under it.");
        }
        Dictionary<string, object> extra = new(request.Request.Extra, StringComparer.Ordinal)
        {
            [DuoRecipePipeline.RunFolderKey] = full,
        };
        request.Request = request.Request with { Extra = extra };
        return null;
    }

    private static IResult Refuse(string reason) => HartsyInferenceServiceExtensions.Problem(StatusCodes.Status400BadRequest,
        $"Extra[\"{DuoRecipePipeline.RunFolderKey}\"] {reason}", "invalid_request_error");
}
