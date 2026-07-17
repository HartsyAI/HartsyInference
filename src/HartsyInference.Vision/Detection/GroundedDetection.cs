using HartsyInference.Core.Pipelines;

namespace HartsyInference.Vision.Detection;

/// <summary>One open-vocabulary detection: a bounding box plus the caption phrase it was grounded to. Unlike a
/// closed-set <see cref="DetectionResult"/> (which carries a fixed class index / label), the label here is a free
/// phrase from the caption, and <see cref="Score"/> is the phrase-aggregated contrastive confidence.</summary>
public sealed record GroundedDetection
{
    /// <summary>Normalized xywh box + confidence; <see cref="DetectionResult.Label"/> is set to <see cref="Phrase"/>.</summary>
    public required DetectionResult Box { get; init; }

    /// <summary>The caption phrase this box was grounded to.</summary>
    public required string Phrase { get; init; }

    /// <summary>Phrase-aggregated contrastive score (post-sigmoid) that selected this phrase for this query.</summary>
    public float Score { get; init; }
}
