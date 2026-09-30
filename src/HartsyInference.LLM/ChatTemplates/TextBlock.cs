namespace HartsyInference.LLM.ChatTemplates;

/// <summary>A run of text inside a structured message body.</summary>
public sealed record TextBlock(string Text) : ContentBlock;
