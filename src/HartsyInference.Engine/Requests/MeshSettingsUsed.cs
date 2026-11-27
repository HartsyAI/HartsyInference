using HartsyInference.Engine.Planning;

namespace HartsyInference.Engine.Requests;

/// <summary>The settings a result was made with, as they ran: the resolved seed (a request's negative seed is drawn, so it is known afterwards), each setting's value and where it came from, and the engine that made it. The same description is embedded in the file.</summary>
public sealed record MeshSettingsUsed
{
    /// <summary>The catalog id of the model.</summary>
    public required string ModelId { get; init; }

    /// <summary>The seed the run used; null for a model with no seed (TripoSR, CoLa3D).</summary>
    public long? Seed { get; init; }

    /// <summary>Every setting the model reads: its value (null where the checkpoint chose it) and whether the caller or the model's default chose it. The seed is derived when it was drawn.</summary>
    public IReadOnlyDictionary<string, ResolvedParameter> Parameters { get; init; } = new Dictionary<string, ResolvedParameter>();

    /// <summary>The version of the engine assembly.</summary>
    public required string EngineVersion { get; init; }
}
