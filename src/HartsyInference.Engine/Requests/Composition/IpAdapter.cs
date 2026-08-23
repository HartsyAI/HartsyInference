namespace HartsyInference.Engine.Requests;

/// <summary>Image-prompt (IP-Adapter / Redux / FaceID) conditioning: one or more reference images plus grouping and face-weight controls.</summary>
public sealed record IpAdapter
{
    /// <summary>The reference / prompt images driving the adapter.</summary>
    public required IReadOnlyList<ImageData> PromptImages { get; init; }

    /// <summary>Whether the prompt images are combined as a single grouped conditioning rather than applied separately.</summary>
    public bool Grouping { get; init; }

    /// <summary>FaceID-v2 blend weight, when a FaceID adapter is in use; null otherwise.</summary>
    public double? FaceIdV2Weight { get; init; }
}
