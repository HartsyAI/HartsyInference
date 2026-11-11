using HartsyInference.Core.Backends;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>Estimates a foreground matte for an opaque image.</summary>
public interface IForegroundMatter
{
    /// <summary>Whether <see cref="Alpha"/> can run right now. A matter that finds its checkpoint lazily reports it here, so a request is
    /// rejected up front instead of failing partway through.</summary>
    bool IsAvailable => true;

    /// <summary>Foreground coverage in <c>[0, 1]</c> per pixel of interleaved RGB24 <paramref name="rgb"/>, row-major.</summary>
    float[] Alpha(IBackend backend, ReadOnlySpan<byte> rgb, int width, int height);
}
