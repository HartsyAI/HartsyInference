namespace HartsyInference.Engine.Planning;

/// <summary>A file the engine cannot fetch, with what the user must do about it, so a host can show instructions instead of an exception.</summary>
/// <param name="Role">What the file is.</param>
/// <param name="Kind">Why it cannot be fetched.</param>
/// <param name="Message">What to do, in one or two sentences; no server paths.</param>
public readonly record struct MeshAssetBlocker(string Role, MeshAssetBlockerKind Kind, string Message);
