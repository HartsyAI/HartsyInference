namespace HartsyInference.Engine.Planning;

/// <summary>One catalog file a model needs that is not on disk.</summary>
/// <param name="Role">What the file is (<c>structure flow</c>, <c>encoder (gated)</c>).</param>
/// <param name="RelativePath">Where the engine looks for it, relative to the models root, never an absolute path.</param>
/// <param name="Required">False when only an optional feature (the texture, the matting model) needs it.</param>
public readonly record struct MeshMissingArtifact(string Role, string RelativePath, bool Required);
