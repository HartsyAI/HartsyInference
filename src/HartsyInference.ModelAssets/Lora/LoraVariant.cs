namespace HartsyInference.ModelAssets.Lora;

/// <summary>Which decomposition a LoRA layer's delta came from. The math for each lives in the matching <see cref="LoraDelta"/> subclass.</summary>
public enum LoraVariant
{
    /// <summary>Standard low-rank adaptation: delta = scale * (B @ A).</summary>
    StandardLora,

    /// <summary>LyCORIS LoHa: delta = (W1a @ W1b) ⊙ (W2a @ W2b), optionally rebuilt through CP/Tucker cores.</summary>
    LoHa,

    /// <summary>LyCORIS LoKr: delta = W1 ⊗ W2 (Kronecker product), either factor optionally low-rank.</summary>
    LoKr,

    /// <summary>Weight-decomposed low-rank adaptation: a standard pair plus a magnitude vector that rescales the LoRA'd weight (<see cref="LoraDoraDecompose"/>). A LoHa or LoKr file can carry the same vector, in which case it keeps its own variant and exposes the vector through <see cref="LoraDelta.DoraScale"/>.</summary>
    DoRA,
}
