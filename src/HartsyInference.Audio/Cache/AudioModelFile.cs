namespace HartsyInference.Audio.Cache;

/// <summary>One file a model pulls from its repo.</summary>
/// <param name="Name">Path within the repo, e.g. <c>"model.safetensors"</c> or <c>"snac/model.safetensors"</c>.</param>
/// <param name="Required">When false the file is fetched if present and skipped if the repo does not have it — several checkpoint families ship tokenizer extras only for some variants.</param>
/// <param name="Sha256">Expected digest of the downloaded file, when the revision is pinned. Verified after download; null skips verification.</param>
public readonly record struct AudioModelFile(string Name, bool Required = true, string? Sha256 = null);
