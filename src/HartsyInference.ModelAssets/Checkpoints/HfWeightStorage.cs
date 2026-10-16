namespace HartsyInference.ModelAssets.Checkpoints;

/// <summary>How a Hugging Face checkpoint stores one logical weight.</summary>
public enum HfWeightStorage
{
    /// <summary>A float tensor (F32, F16 or BF16) as is.</summary>
    Plain,

    /// <summary>F8 E4M3 values with an F32 <c>weight_scale_inv</c> block grid beside them.</summary>
    Fp8Block,

    /// <summary>Symmetric 4-bit GPTQ: <c>qweight</c>, <c>scales</c>, <c>qzeros</c> and an optional <c>g_idx</c>.</summary>
    GptqInt4,
}
