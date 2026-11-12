namespace HartsyInference.Diffusion.Pipelines;

/// <summary>An Iris-Z generation.</summary>
/// <param name="Rgb">Row-major RGB24 pixels.</param>
/// <param name="LogDepth">Relative log depth per pixel in <c>[-1, 1]</c> (near is low, sky is +1), or null when the
/// run produced no depth.</param>
/// <param name="Width">Pixel width.</param>
/// <param name="Height">Pixel height.</param>
/// <param name="Seed">The seed used.</param>
public sealed record IrisZImage(byte[] Rgb, float[]? LogDepth, int Width, int Height, int Seed);
