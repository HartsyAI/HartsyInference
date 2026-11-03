using HartsyInference.Engine.Requests;

namespace HartsyInference.API.Endpoints;

/// <summary>Envelope for <c>/v1/native/vision</c>. <see cref="VisionRequest.Image"/> is raw RGB24 (<see cref="ImageData"/>), matching the native contract exactly — no PNG/JPEG decode convenience layer, same as every other native route's pass-through philosophy.</summary>
public sealed class NativeVisionRequest
{
    /// <summary>Catalog id, local path, or HuggingFace repo id.</summary>
    public required string Model { get; set; }

    /// <summary>Explicit checkpoint path override; wins over catalog/HF resolution. Must be an absolute path — a relative one resolves against the server process's working directory, not anything an HTTP client can know. Rarely needed: a plain catalog id in <c>model</c> resolves correctly on its own.</summary>
    public string? ModelPath { get; set; }

    /// <summary>The native vision request, unmodified.</summary>
    public required VisionRequest Request { get; set; }
}

/// <summary>Envelope for <c>/v1/native/vision/sequence</c>: several views of one scene (answered as one JSON result with
/// <c>views</c>) or the frames of a video (answered as a <c>text/event-stream</c> of <c>frame</c> events, then
/// <c>complete</c>). Images are raw RGB24, as on <c>/v1/native/vision</c>.</summary>
public sealed class NativeVisionSequenceRequest
{
    /// <summary>Catalog id, local path, or HuggingFace repo id.</summary>
    public required string Model { get; set; }

    /// <summary>Explicit absolute checkpoint path override; wins over catalog/HF resolution.</summary>
    public string? ModelPath { get; set; }

    /// <summary>The native batch request, unmodified.</summary>
    public required VisionBatchRequest Request { get; set; }
}

/// <summary>One <c>frame</c> event on <c>/v1/native/vision/sequence</c>.</summary>
public sealed class NativeVisionFrameEvent
{
    /// <summary>The output frame's index in the stream.</summary>
    public required int Index { get; init; }

    /// <summary>The frame's presentation time in seconds.</summary>
    public required double TimestampSeconds { get; init; }

    /// <summary>The input frame this output copies or describes; null for a synthesized frame.</summary>
    public int? SourceIndex { get; init; }

    /// <summary>The frame's result, in the <c>/v1/native/vision</c> response shape.</summary>
    public required object Result { get; init; }
}

/// <summary>The closing <c>complete</c> event on <c>/v1/native/vision/sequence</c>.</summary>
public sealed class NativeVisionSequenceCompleteEvent
{
    /// <summary>How many frame events were sent.</summary>
    public required int Frames { get; init; }
}

/// <summary>Envelope for <c>/v1/native/mesh</c>.</summary>
public sealed class NativeMeshRequest : NativeArtifactRequest
{
    /// <summary>Catalog id, local path, or HuggingFace repo id.</summary>
    public required string Model { get; set; }

    /// <summary>Explicit checkpoint path override; wins over catalog/HF resolution. Must be an absolute path — a relative one resolves against the server process's working directory, not anything an HTTP client can know. Rarely needed: a plain catalog id in <c>model</c> resolves correctly on its own.</summary>
    public string? ModelPath { get; set; }

    /// <summary>The native mesh request, unmodified.</summary>
    public required MeshRequest Request { get; set; }
}
