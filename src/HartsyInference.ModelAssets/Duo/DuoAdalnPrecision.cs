namespace HartsyInference.ModelAssets.Duo;

/// <summary>How an AdaLN table is rounded.</summary>
public enum DuoAdalnPrecision
{
    /// <summary>F32 throughout, which is what the engine's H3 path computes.</summary>
    F32,

    /// <summary>The upstream numerics: the SiLU of the time embedding rounds to BF16 before the BF16 <c>adaln_proj</c>, and its output rounds to BF16.</summary>
    Bf16Reference,
}
