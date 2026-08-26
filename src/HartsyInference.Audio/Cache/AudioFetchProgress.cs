namespace HartsyInference.Audio.Cache;

/// <summary>Progress of a multi-file model fetch, reported as each file lands.</summary>
/// <param name="File">Repo-relative name of the file that just completed.</param>
/// <param name="FilesDone">Files resolved so far, including ones already cached.</param>
/// <param name="FileCount">Total files in the fetch list, including optional ones that may turn out to be absent.</param>
public readonly record struct AudioFetchProgress(string File, int FilesDone, int FileCount);
