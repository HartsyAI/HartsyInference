using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.Engine.Recipes;

/// <summary>A constructed, ready-to-run video pipeline for one architecture family. Owns the loaded components and
/// drives that family's bespoke encode + denoise + decode, returning the decoded frames and any soundtrack that
/// belongs with them. Cached per loaded model and reused across requests.</summary>
public interface IVideoRecipePipeline : IDisposable
{
    /// <summary>Generates for <paramref name="request"/>, reporting step progress. Attach audio to the result only when it is meant to be heard — a family that consumes audio purely as conditioning leaves it null.</summary>
    VideoGenerationResult Generate(VideoRequest request, IProgress<StepPreview>? progress, CancellationToken cancel);

    /// <summary>True when this family can stream frames as they decode instead of buffering the whole clip (Tier
    /// 3.6/3.5 backlog — as of 2026-08-11 only Wan's plain T2V/I2V-TI2V path does; the concat-I2V variant and
    /// every other family still throw from <see cref="GenerateFramesAsync"/>). Default false so adding streaming
    /// to a family is opt-in, not a silent expectation on every implementer.</summary>
    bool SupportsStreaming => false;

    /// <summary>Streaming sibling of <see cref="Generate"/>: yields frames as they decode. No audio track and no
    /// request-level frame edits (trim/boomerang) — those need the full buffered clip (final frame count / random
    /// access), so <see cref="VideoService"/> only calls this for edit-free, audio-free requests and falls back to
    /// <see cref="Generate"/> otherwise. A family/variant that can't stream (or a request shape it can't stream —
    /// e.g. Wan's concat-I2V) throws <see cref="NotSupportedException"/> with a message naming the reason.</summary>
    IAsyncEnumerable<VideoFrame> GenerateFramesAsync(VideoRequest request, IProgress<StepPreview>? progress, CancellationToken cancel) =>
        throw new NotSupportedException("This video family does not support streaming generation.");
}
