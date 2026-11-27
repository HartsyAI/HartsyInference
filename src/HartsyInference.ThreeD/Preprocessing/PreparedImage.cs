namespace HartsyInference.ThreeD.Preprocessing;

/// <summary>An image after a model's input preparation: interleaved RGB24 rows, the size they are, and the foreground coverage that went into them when the model keeps one.</summary>
/// <param name="Rgb">Row-major interleaved RGB24, <c>Width * Height * 3</c> bytes.</param>
/// <param name="Width">Pixel width.</param>
/// <param name="Height">Pixel height.</param>
/// <param name="Alpha">Row-major coverage, <c>Width * Height</c> bytes, or null when the preparation composites it away.</param>
public sealed record PreparedImage(byte[] Rgb, int Width, int Height, byte[]? Alpha = null);
