namespace HartsyInference.Engine.Requests;

/// <summary>One object's mask in a scene prompt: row-major coverage, 0 (outside) to 255 (inside).</summary>
public sealed record ScenePromptMask
{
    /// <summary>Coverage bytes, <see cref="Width"/> * <see cref="Height"/> long.</summary>
    public required byte[] Coverage { get; init; }

    /// <summary>Pixel width.</summary>
    public required int Width { get; init; }

    /// <summary>Pixel height.</summary>
    public required int Height { get; init; }
}
