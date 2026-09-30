namespace HartsyInference.LLM.ChatTemplates;

/// <summary>Where one image sits in the encoded prompt: <paramref name="Length"/> positions from <paramref name="Start"/>, all carrying the image token id.</summary>
public readonly record struct ImageSpan(int ImageIndex, int Start, int Length, ImageGrid Grid);
