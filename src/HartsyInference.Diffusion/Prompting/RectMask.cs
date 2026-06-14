namespace HartsyInference.Diffusion.Prompting;

/// <summary>Pixel-space rectangle for regional prompting: top-left <c>(X, Y)</c> plus <c>Width</c>/<c>Height</c>.</summary>
public readonly record struct RectMask(int X, int Y, int Width, int Height);
