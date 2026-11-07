namespace HartsyInference.Core.Backends;

// Backend-agnostic capture and replay of an op sequence as one graph handle.
public partial interface IBackend
{
    // Callers outside HartsyInference.Cuda (e.g. TextGenerationPipeline, which must stay backend-agnostic —
    // CPU builds link against IBackend only) capture/replay through this opaque-handle indirection instead of
    // referencing CudaGraph directly. Default: unsupported: CaptureGraph returns null, meaning "not captured,
    // call recordWork() yourself every time" — callers must treat a null handle as the eager fallback, not an error.

    /// <summary>Captures the device work recordWork issues into a graph; returns a handle for <see cref="LaunchGraph"/>, or null.</summary>
    object? CaptureGraph(Action recordWork) => null;

    /// <summary>Replays a graph captured by <see cref="CaptureGraph"/> with a single launch.</summary>
    void LaunchGraph(object graphHandle) => throw new NotSupportedException("LaunchGraph requires a handle from a backend whose CaptureGraph returned non-null.");

    /// <summary>Disposes a graph handle from <see cref="CaptureGraph"/>.</summary>
    void DisposeGraph(object graphHandle) { }
}
