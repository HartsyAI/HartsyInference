namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>The preprocessed conditioning image: an RGB24 square, foreground premultiplied by alpha.</summary>
/// <param name="Rgb">Interleaved RGB24, row-major, <c>Width · Height · 3</c> bytes.</param>
public readonly record struct Trellis2PreparedImage(byte[] Rgb, int Width, int Height);
