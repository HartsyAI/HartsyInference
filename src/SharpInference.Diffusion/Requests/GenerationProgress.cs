namespace SharpInference.Diffusion.Requests;

/// <summary>Progress report during image generation. Emitted after each denoising step.</summary>
public readonly record struct GenerationProgress(int Step, int TotalSteps, double ElapsedMs);
