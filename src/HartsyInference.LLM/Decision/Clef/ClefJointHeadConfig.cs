namespace HartsyInference.LLM.Decision.Clef;

/// <summary>Shape of the joint schema head (<c>joint_head_config.json</c>).</summary>
public sealed record ClefJointHeadConfig
{
    public int HiddenSize { get; init; } = 4_096;

    public int Width { get; init; } = 1_024;

    public int RoutingLayers { get; init; } = 2;

    public int Layers { get; init; } = 4;

    public int Heads { get; init; } = 16;

    public int Feedforward { get; init; } = 4_096;
}
