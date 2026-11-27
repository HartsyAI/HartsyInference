namespace HartsyInference.Core.Configuration;

/// <summary>LoRA application: what a stack does with a unit it cannot apply.</summary>
public static partial class EngineKnobs
{
    /// <summary>What a LoRA stack does with a unit it cannot apply (no matching weight, or a shape from another build).</summary>
    public static readonly Knob<string?> LoraPartialApplication =
        Str("lora.partialApplication", "allow", KnobScope.Runtime, KnobDomain.Lora,
            "What a LoRA stack does with a unit it cannot apply: allow applies what fits and reports the rest, refuse rejects the request by name.");
}
