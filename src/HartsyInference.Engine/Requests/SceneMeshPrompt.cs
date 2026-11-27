namespace HartsyInference.Engine.Requests;

/// <summary>Which objects a scene-decomposition model extracts: one query per object, as masks in one view or as points in
/// the scene's units. Give exactly one of <see cref="Masks"/> and <see cref="Points"/>.</summary>
public sealed record SceneMeshPrompt
{
    /// <summary>One mask per object, all the same size, from a single view. Selects the prompt-mask checkpoint. The camera,
    /// image size and framing the checkpoint expects are not verified yet (see docs/Research/COLA3D_GEOMETRY_DECODER.md).</summary>
    public IReadOnlyList<ScenePromptMask>? Masks { get; init; }

    /// <summary>One point per object, in the scene mesh's units. Selects the prompt-point checkpoint.</summary>
    public IReadOnlyList<ScenePoint>? Points { get; init; }
}
