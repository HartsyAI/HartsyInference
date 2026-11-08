namespace HartsyInference.ThreeD.Models.HairDeck;

/// <summary>One decoded hair card in the model's head frame (y up, face toward +z, metres).</summary>
/// <param name="Centerline">Row-major xyz of <paramref name="Points"/> control points, root first (length <c>Points * 3</c>).</param>
/// <param name="Points">Number of control points of the card (at least 3).</param>
/// <param name="Width">Mean width over the points.</param>
/// <param name="Thickness">Mean thickness over the points.</param>
/// <param name="ShapeClass">0 for a square cross-section, 1 for a triangle.</param>
/// <param name="Color">Sixteen RGB colours along the card, root to tip, linear, row-major (length 48); null when colours are disabled.</param>
public sealed record DecodedCard(float[] Centerline, int Points, float Width, float Thickness, int ShapeClass, float[] Color)
{
    /// <summary>Width at each of the <see cref="Points"/> control points; the mesh needs this profile, not <see cref="Width"/>.</summary>
    public float[] WidthProfile { get; init; } = [];

    /// <summary>Thickness at each of the <see cref="Points"/> control points.</summary>
    public float[] ThicknessProfile { get; init; } = [];
}

/// <summary>One spring-bone chain in the model's head frame. <see cref="Joints"/> are the joints interpolated along the chain.</summary>
/// <param name="Joints">Row-major xyz of <paramref name="JointCount"/> joints, root first (length <c>JointCount * 3</c>).</param>
/// <param name="JointCount">Number of joints, clamped to 2..20.</param>
/// <param name="Stiffness">Chain stiffness clamped to 0..1.</param>
public sealed record DecodedChain(float[] Joints, int JointCount, float Stiffness)
{
    /// <summary>The eight arc-length samples of the chain path (row-major xyz, length 24) the joints are interpolated from.</summary>
    public float[] Path { get; init; } = [];
}

/// <summary>Decoded output of one HairDeck sample: its cards and chains, and the seed of the initial noise.</summary>
/// <param name="Cards">Cards in the order of their slots in the card token set.</param>
/// <param name="Chains">Chains in the order of their slots; empty when chains were not sampled.</param>
/// <param name="Seed">Seed of the initial noise.</param>
public sealed record HairDeckResult(IReadOnlyList<DecodedCard> Cards, IReadOnlyList<DecodedChain> Chains, int Seed);
