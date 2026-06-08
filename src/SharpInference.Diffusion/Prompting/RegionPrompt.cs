namespace SharpInference.Diffusion.Prompting;

/// <summary>A per-region prompt + pixel-space mask, for models that condition on <c>(prompt, mask)</c> regions via attention coupling (as opposed to Ideogram's native JSON bbox control).</summary>
public readonly record struct RegionPrompt(string Prompt, RectMask Mask, float Weight);
