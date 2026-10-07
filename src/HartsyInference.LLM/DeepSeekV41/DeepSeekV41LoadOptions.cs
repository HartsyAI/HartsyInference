namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>Sizing choices for loading a V4.1 checkpoint into the host reference model.</summary>
/// <param name="MaxTokens">Longest sequence the model will be asked to hold; sizes the rope tables and every sequence state.</param>
/// <param name="ExpertCacheCapacity">Dequantized routed experts kept across all layers (each is three F32 matrices, roughly 140 MB on the official checkpoint).</param>
/// <param name="EngramBudgetBytes">Host memory each Engram table may use for cached rows; the tables themselves stay on disk.</param>
public sealed record DeepSeekV41LoadOptions(int MaxTokens, int ExpertCacheCapacity = 16, long EngramBudgetBytes = 1L << 30)
{
    /// <summary>Checks every field is usable.</summary>
    public void Validate()
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(MaxTokens, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(ExpertCacheCapacity, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(EngramBudgetBytes, 1L);
    }
}
