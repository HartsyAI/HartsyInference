namespace HartsyInference.LLM.Transformer.Mhc;

/// <summary>Where the RMS epsilon sits in a hyper-connection's stream statistic.</summary>
public enum RmsEpsPlacement
{
    /// <summary>Scale <c>rsqrt(mean(x^2) + eps)</c>. DeepSeek-V4.1 uses it for the mixes.</summary>
    InsideSqrt,

    /// <summary>Scale <c>1 / (sqrt(mean(x^2)) + eps)</c>. ERNIE-5.5 uses it for the mixes; its reference test asserts the two forms differ.</summary>
    OutsideSqrt,
}
