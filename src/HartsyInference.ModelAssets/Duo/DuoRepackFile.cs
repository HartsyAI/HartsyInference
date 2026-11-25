namespace HartsyInference.ModelAssets.Duo;

/// <summary>One file a repack wrote: its path relative to the output folder, size and SHA-256.</summary>
public readonly record struct DuoRepackFile(string RelativePath, long Bytes, string Sha256);
