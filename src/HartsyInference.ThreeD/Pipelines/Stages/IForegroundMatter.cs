using HartsyInference.Core.Backends;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>Estimates a foreground matte for an opaque image.</summary>
public interface IForegroundMatter
{
    /// <summary>Foreground coverage in <c>[0, 1]</c> per pixel of interleaved RGB24 <paramref name="rgb"/>, row-major <c>height × width</c>.</summary>
    float[] Alpha(IBackend backend, ReadOnlySpan<byte> rgb, int width, int height);
}
