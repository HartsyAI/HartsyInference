namespace HartsyInference.ThreeD.Geometry;

/// <summary>
/// One hair card in the reference card format: <paramref name="Sequence"/> holds N rows of (x, y, z, width, thickness,
/// root width direction x, y, z), row-major, metres. <paramref name="ShapeClass"/> is 0 (square) or 1 (triangle).
/// <paramref name="Colors"/> are B linear RGB colours (B * 3 floats) along the card from root to tip, or null.
/// </summary>
public sealed record HairCard(double[] Sequence, int ShapeClass, float[]? Colors);
