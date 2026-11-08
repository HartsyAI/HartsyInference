namespace HartsyInference.Vision.Segmentation;

/// <summary>Instances found for one text prompt over one image, highest score first.</summary>
public sealed record TextGroundedPromptResult(string Prompt, int Width, int Height, IReadOnlyList<TextGroundedInstance> Instances);
