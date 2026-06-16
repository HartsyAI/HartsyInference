using HartsyInference.Core.Backends;
using HartsyInference.Core.Pipelines;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Segmentation;

/// <summary>Stub for the CLIPSeg text-prompted segmentation pipeline. Reserves the type and the
/// <see cref="ITextSegmenter"/> API shape so callers can wire <c>&lt;segment:text&gt;</c> to
/// <c>RegionMask.FromBitmap</c> without future churn; the implementation is deferred because YOLO
/// class-prompted segmentation already covers the common case. The real pipeline will pair a CLIP
/// ViT-B/16 backbone with the small FiLM-conditioned transformer decoder that emits the mask.
/// <para>Throws <see cref="NotImplementedException"/> on every operation. Next steps: write a
/// checkpoint converter for the CIDAS CLIPSeg weights and validate the decoder against the Python
/// reference within documented tolerances.</para></summary>
public sealed class ClipSegPipeline : IVisionPipeline, ITextSegmenter
{
    /// <inheritdoc/>
    public string ModelName => "clipseg-stub";

    /// <summary>Placeholder constructor. The real pipeline will take a CLIP backbone + CLIPSeg decoder plus an <see cref="IBackend"/>.</summary>
    public ClipSegPipeline() => throw new NotImplementedException(
        "CLIPSeg text-prompted segmentation is not yet implemented. Use YOLO <segment:> (Detection) for the common case; " +
        "this seam exists so CLIPSeg can be added without API churn — its float[] mask feeds RegionMask.FromBitmap.");

    /// <inheritdoc/>
    public float[] Segment(IBackend backend, Tensor image, string prompt, out int width, out int height) =>
        throw new NotImplementedException("CLIPSeg segmentation is not yet implemented.");

    /// <inheritdoc/>
    public void Dispose() { }
}
