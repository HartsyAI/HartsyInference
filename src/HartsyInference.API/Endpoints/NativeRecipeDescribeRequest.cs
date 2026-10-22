namespace HartsyInference.API.Endpoints;

/// <summary>Body of <c>POST /v1/native/recipes/describe</c>: names a model the way the video plan route does.</summary>
public sealed class NativeRecipeDescribeRequest
{
    /// <summary>The domain to describe: <c>image</c> or <c>video</c>.</summary>
    public required string Modality { get; set; }

    /// <summary>Catalog id, local path, or HuggingFace repo id.</summary>
    public required string Model { get; set; }

    /// <summary>Explicit checkpoint path override, resolved the same way the video plan route resolves it.</summary>
    public string? ModelPath { get; set; }
}
