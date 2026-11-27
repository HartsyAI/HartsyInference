namespace HartsyInference.Engine.Recipes.Video;

/// <summary>A chain request the LongCat-Video-Avatar recipe refuses. The <see cref="Code"/> is stable (<c>video.chain.*</c>), so a caller can match on
/// it without parsing the message.</summary>
public sealed class VideoChainException(string code, string message, string paramName) : ArgumentException(message, paramName)
{
    /// <summary>The stable code: <c>video.chain.budget</c>, <c>video.chain.segment_frames</c> or <c>video.chain.fps</c>.</summary>
    public string Code { get; } = code;
}
