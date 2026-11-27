using HartsyInference.Core.Backends;
using HartsyInference.Vision.Matting.BiRefNet;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>Matting through <see cref="BiRefNetMatter"/>. The caller keeps ownership of the matter. TRELLIS.2 itself uses RMBG-2.0, a
/// BiRefNet with other weights; this adapter serves whatever BiRefNet checkpoint the matter was loaded with.</summary>
public sealed class BiRefNetForegroundMatter : IForegroundMatter
{
    private readonly BiRefNetMatter _matter;

    public BiRefNetForegroundMatter(BiRefNetMatter matter) => _matter = matter ?? throw new ArgumentNullException(nameof(matter));

    /// <inheritdoc/>
    public float[] Alpha(IBackend backend, ReadOnlySpan<byte> rgb, int width, int height) => _matter.Matte(backend, rgb, width, height);
}
