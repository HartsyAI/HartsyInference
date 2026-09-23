namespace HartsyInference.Engine.Planning.Memory;

/// <summary>How much of an estimate came from the model's own formulas versus a generic allowance.</summary>
public enum MemoryEstimateAccuracy
{
    /// <summary>The family's recipe described its activations with the same formulas its pipeline budgets with.</summary>
    Recipe,

    /// <summary>Weight bytes come from the checkpoint header; activations are a generic pixel-scaled allowance
    /// because the family has not described its own. Trust the weights, treat the activations as a guess.</summary>
    HeaderOnly,
}
