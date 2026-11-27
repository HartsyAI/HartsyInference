namespace HartsyInference.Engine.Requests;

/// <summary>A rendered thumbnail of a model file.</summary>
public sealed record MeshThumbnail
{
    /// <summary>The picture as a PNG: RGBA when the background is transparent, RGB otherwise.</summary>
    public required byte[] Png { get; init; }

    /// <summary>Pixel width.</summary>
    public required int Width { get; init; }

    /// <summary>Pixel height.</summary>
    public required int Height { get; init; }

    /// <summary>What was drawn: <c>mesh</c> (triangles) or <c>points</c> (a splat or point cloud's centres).</summary>
    public required string Kind { get; init; }

    /// <summary>What was not drawn as the file describes it (a texture that could not be decoded, faint splats left out).</summary>
    public IReadOnlyList<string> Notes { get; init; } = [];
}
