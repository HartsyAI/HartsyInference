namespace HartsyInference.LLM.Transformer;

/// <summary>Where a latent MoE block (<see cref="MoeConfig.LatentSize"/>) applies its latent RMSNorm.</summary>
public enum MoeLatentNorm
{
    /// <summary>On the projected input, before the experts read it (Muse Spark <c>pre_expert_norm</c>).</summary>
    BeforeExperts,

    /// <summary>On the weighted expert sum, before the exit projection (ERNIE-5.5 <c>latent_norm</c>).</summary>
    AfterCombine,

    /// <summary>No latent norm.</summary>
    None,
}
