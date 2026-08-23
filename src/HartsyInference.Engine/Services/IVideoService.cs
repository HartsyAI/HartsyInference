using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Typed video-generation surface: returns the decoded frames together with the soundtrack that belongs with them.</summary>
public interface IVideoService
{
    /// <summary>Generates a video for <paramref name="request"/>, including any audio the model produced or the request supplied for mux.</summary>
    Task<VideoGenerationResult> GenerateAsync(ModelSpec spec, VideoRequest request, IProgress<StepPreview>? progress = null, CancellationToken cancel = default);

    /// <summary>Streams frames as they decode instead of buffering the whole clip (Tier 3.5) — lower peak memory and time-to-first-frame for the subset of requests this applies to. No audio and no trim/boomerang: those need the full clip (final frame count / random access), so only call this for a request with neither set; use <see cref="GenerateAsync"/> otherwise. Throws <see cref="NotSupportedException"/> if the resolved family/variant can't stream at all (check with a request shape it can actually serve, or catch and fall back to <see cref="GenerateAsync"/>).</summary>
    IAsyncEnumerable<VideoFrame> GenerateFramesAsync(ModelSpec spec, VideoRequest request, IProgress<StepPreview>? progress = null, CancellationToken cancel = default);
}
