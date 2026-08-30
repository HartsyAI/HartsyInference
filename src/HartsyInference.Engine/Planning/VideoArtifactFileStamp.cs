namespace HartsyInference.Engine.Planning;

/// <summary>Identity of one resolved construction artifact at planning time.</summary>
internal readonly record struct VideoArtifactFileStamp(string CanonicalPath, long Length, long LastWriteUtcTicks);
