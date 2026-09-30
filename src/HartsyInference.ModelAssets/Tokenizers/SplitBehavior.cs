namespace HartsyInference.ModelAssets.Tokenizers;

/// <summary>HuggingFace <c>SplitDelimiterBehavior</c>: what a <c>Split</c> pre-tokenizer stage does with the regex-matched spans.</summary>
public enum SplitBehavior
{
    Removed,
    Isolated,
    MergedWithPrevious,
    MergedWithNext,
    Contiguous,
}
