namespace HartsyInference.Engine.Requests;

/// <summary>A region of the init image in its own pixels: rows <see cref="YMin"/> up to <see cref="YMax"/> and columns <see cref="XMin"/> up to
/// <see cref="XMax"/>, the ends excluded. A box past the image's far edges is clipped to the image.</summary>
public sealed record VideoSpeakerBox
{
    /// <summary>First row.</summary>
    public required int YMin { get; init; }

    /// <summary>First column.</summary>
    public required int XMin { get; init; }

    /// <summary>The row after the last.</summary>
    public required int YMax { get; init; }

    /// <summary>The column after the last.</summary>
    public required int XMax { get; init; }
}
