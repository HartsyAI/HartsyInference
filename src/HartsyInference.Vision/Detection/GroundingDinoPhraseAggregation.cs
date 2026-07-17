namespace HartsyInference.Vision.Detection;

/// <summary>How a phrase's per-sub-token contrastive scores collapse into one phrase score. Grounding DINO's
/// reference post-processing takes the max over a phrase's token span; mean is offered as an alternative.</summary>
public enum GroundingDinoPhraseAggregation
{
    /// <summary>Maximum score over the phrase's token span (Grounding DINO default).</summary>
    Max,

    /// <summary>Mean score over the phrase's token span.</summary>
    Mean,
}
