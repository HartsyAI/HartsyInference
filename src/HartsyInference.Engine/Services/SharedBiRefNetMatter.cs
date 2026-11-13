using HartsyInference.Core.Backends;
using HartsyInference.ThreeD.Pipelines.Stages;

namespace HartsyInference.Engine.Services;

/// <summary>The BiRefNet-matting checkpoint as the shared matter: found and read on first use, released with <see cref="Dispose"/>.</summary>
/// <param name="matter">The lazy matter this wraps; this instance owns it.</param>
internal sealed class SharedBiRefNetMatter(Trellis2LazyMatter matter) : ISharedMatter
{
    /// <inheritdoc/>
    public string? LoadedFrom => matter.LoadedFrom;

    /// <inheritdoc/>
    public bool IsAvailable => matter.IsAvailable;

    /// <inheritdoc/>
    public float[] Alpha(IBackend backend, ReadOnlySpan<byte> rgb, int width, int height) => matter.Alpha(backend, rgb, width, height);

    /// <inheritdoc/>
    public void Dispose() => matter.Dispose();
}
