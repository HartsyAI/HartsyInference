namespace HartsyInference.ThreeD.Models.HairDeck.Preprocessing;

/// <summary>The square head crop in source pixels, <c>x1 - x0</c> wide, possibly extending past the image (rounded to 0.1 px).</summary>
public readonly record struct HairDeckBox(double X0, double Y0, double X1, double Y1);
