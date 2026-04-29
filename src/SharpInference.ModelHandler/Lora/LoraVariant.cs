namespace SharpInference.ModelHandler.Lora;

/// <summary>LoRA decomposition variant. Only StandardLora is implemented in v1; LoHa and LoKr are recognized but rejected at load.</summary>
public enum LoraVariant
{
    /// <summary>Standard low-rank adaptation: delta = scale * (B @ A).</summary>
    StandardLora,

    /// <summary>LyCORIS LoHa: delta = (W1b @ W1a) ⊙ (W2b @ W2a). Not implemented in v1.</summary>
    LoHa,

    /// <summary>LyCORIS LoKr: delta = W1 ⊗ W2 (Kronecker product). Not implemented in v1.</summary>
    LoKr,
}
