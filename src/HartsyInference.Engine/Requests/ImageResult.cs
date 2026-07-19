namespace HartsyInference.Engine.Requests;

/// <summary>The result of an image generation: the decoded RGB pixels plus the seed actually used and free-form
/// metadata. One per generated image (a batch yields a list of these).</summary>
public sealed record ImageResult
{
    /// <summary>Row-major, top-to-bottom RGB24 bytes; length is <see cref="Width"/> * <see cref="Height"/> * 3.</summary>
    public required byte[] Rgb { get; init; }

    /// <summary>Pixel width.</summary>
    public required int Width { get; init; }

    /// <summary>Pixel height.</summary>
    public required int Height { get; init; }

    /// <summary>The seed actually used (resolved from a random seed when the request asked for -1).</summary>
    public long Seed { get; init; }

    /// <summary>Free-form metadata surfaced to the caller (steps, sampler, timing, arch).</summary>
    public IReadOnlyDictionary<string, string> Meta { get; init; } = new Dictionary<string, string>();
}
