namespace HartsyInference.Vision.Segmentation;

/// <summary>Stub for the SAM / SAM 2 mask decoder — a two-way transformer that takes the image
/// encoder output plus prompt tokens (points, boxes, optional dense mask) and predicts up to
/// 3 candidate masks per query with associated IoU confidence scores. Not yet implemented; see
/// <see cref="SamPipeline"/> for the parent stub and the Phase 6 checklist for the planned layout.</summary>
public sealed class SamMaskDecoder
{
    /// <summary>Placeholder. The real implementation will run the two-way transformer + multi-mask head.</summary>
    public SamMaskDecoder() => throw new NotImplementedException(
        "SAM mask decoder scaffolding only — two-way transformer + multi-mask head not yet built.");
}
