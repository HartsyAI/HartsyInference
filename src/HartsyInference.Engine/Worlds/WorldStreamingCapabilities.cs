namespace HartsyInference.Engine.Worlds;

/// <summary>What a world model that generates chunk by chunk supports, declared on its <see cref="WorldCapabilities"/>. A
/// streaming session delivers each chunk's frames through <see cref="Requests.IWorldSession.StreamAsync"/> as soon as the
/// chunk is decoded, and actions sent between chunks steer the chunks that follow. The frame grid, defaults and camera rules
/// of a streaming request are the model's ordinary capabilities.</summary>
public sealed record WorldStreamingCapabilities
{
    /// <summary>Whether streaming is the model's only mode: it has no whole-clip batch run, so a request that names no mode
    /// streams and a request for a batch run is refused.</summary>
    public bool Only { get; init; }

    /// <summary>Frames the first chunk delivers, the seed frame included.</summary>
    public required int FirstChunkFrames { get; init; }

    /// <summary>Frames every later chunk delivers.</summary>
    public required int ChunkFrames { get; init; }

    /// <summary>Whether a stream may run with no frame count, until its session is disposed or cancelled.</summary>
    public bool SupportsUnbounded { get; init; }

    /// <summary>Whether actions sent while the stream runs steer it. Such a model takes its camera from live actions: a request
    /// needs no camera path, and an action script on the request is the stream's opening move rather than its whole path.</summary>
    public bool SupportsLiveActions { get; init; }
}
