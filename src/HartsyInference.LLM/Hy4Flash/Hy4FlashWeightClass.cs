namespace HartsyInference.LLM.Hy4Flash;

/// <summary>How a Hy4 Flash tensor is read, which decides where it lives on a small card.</summary>
public enum Hy4FlashWeightClass
{
    /// <summary>Every token reads it: attention, router, shared experts, dense FFN, norms, projections, embedding and head.</summary>
    PerToken,

    /// <summary>The fused routed-expert stacks and their block scales; a token reads 16 of 512 experts.</summary>
    RoutedExperts,

    /// <summary>The n-gram memory tables; a token reads a few rows.</summary>
    EngramTable,

    /// <summary>Vision tower and projector.</summary>
    Vision,

    /// <summary>The speculative-decoding drafter; not run by the first series.</summary>
    Drafter,
}
