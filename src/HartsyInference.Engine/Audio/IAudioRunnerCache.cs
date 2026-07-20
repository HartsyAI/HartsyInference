namespace HartsyInference.Engine.Audio;

/// <summary>A per-category cache of resident audio pipelines that <see cref="AudioRuntime"/> can drop when a model
/// switch would not fit in host RAM or VRAM.</summary>
internal interface IAudioRunnerCache
{
    /// <summary>Drops every resident runner except the one keyed <paramref name="keepKey"/> (the incoming model).</summary>
    void UnloadAllExcept(string? keepKey);
}
