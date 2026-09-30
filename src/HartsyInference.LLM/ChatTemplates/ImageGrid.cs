namespace HartsyInference.LLM.ChatTemplates;

/// <summary>Post-merge LLM-side patch grid of one image (columns x rows), as produced by the image processor.</summary>
public readonly record struct ImageGrid(int Width, int Height)
{
    /// <summary>Prompt positions the image occupies: IMAGE_START, one NEWLINE-terminated row per grid row, IMAGE_END.</summary>
    public int TokenCount => Height * (Width + 1) + 2;
}
