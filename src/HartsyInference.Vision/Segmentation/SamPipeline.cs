using HartsyInference.Core.Pipelines;

namespace HartsyInference.Vision.Segmentation;

/// <summary>Stub for the Segment Anything Model (SAM / SAM 2) pipeline. Phase 6 reserves the
/// type and API shape; the implementation is deferred to a future slice because SAM brings two
/// non-trivial pieces of new infrastructure:
/// <list type="bullet">
///   <item><b>Windowed/relative-position attention</b> in the image encoder (Hiera backbone for SAM 2).</item>
///   <item><b>Mask decoder</b> — a two-way transformer that consumes prompt tokens (points, boxes, masks) and produces multi-mask predictions with IoU scores.</item>
/// </list>
/// <para>Throws <see cref="NotImplementedException"/> on every operation. Suggested next steps:
/// research the SAM 2 image-encoder + mask-decoder layout, write a checkpoint converter for the
/// Meta SAM 2 weights, and decide whether to support video tracking (the "memory attention"
/// path) in v1 or defer to a follow-up.</para></summary>
public sealed class SamPipeline : IVisionPipeline
{
    /// <inheritdoc/>
    public string ModelName => "sam2-stub";

    /// <summary>Placeholder constructor. The real pipeline will take a backbone + mask decoder
    /// pair, plus an <see cref="HartsyInference.Core.Backends.IBackend"/>.</summary>
    public SamPipeline() => throw new NotImplementedException(
        "SAM pipeline scaffolding only — image encoder (Hiera) and mask decoder are not yet implemented. " +
        "See PHASE_6_VISION.md § 5 for the planned API surface.");

    /// <inheritdoc/>
    public void Dispose() { }
}
