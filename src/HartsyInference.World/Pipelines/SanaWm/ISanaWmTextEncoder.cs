namespace HartsyInference.World.Pipelines.SanaWm;

/// <summary>The SANA-WM text path: the CHI-prefixed prompt and the plain negative prompt through Gemma-2.</summary>
public interface ISanaWmTextEncoder
{
    /// <summary>Encodes both prompts. The caller owns the result.</summary>
    SanaWmTextConditioning Encode(string prompt, string negativePrompt);
}
