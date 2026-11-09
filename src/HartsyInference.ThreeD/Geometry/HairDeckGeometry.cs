using HartsyInference.ThreeD.Models.HairDeck;

namespace HartsyInference.ThreeD.Geometry;

/// <summary>Hair cards and spring chains of one decoded HairDeck sample, in the model's head frame (metres).</summary>
public static class HairDeckGeometry
{
    private const int MinPoints = 3;
    private const int RowStride = 8;

    /// <summary>Hair cards of <paramref name="result"/> (null when no card has a triangle); see <see cref="HairCardMesher.Build"/>.</summary>
    public static HairCardMesh? BuildCards(HairDeckResult result, float[] hairColorSrgb)
    {
        ArgumentNullException.ThrowIfNull(result);
        List<HairCard> cards = [];
        foreach (DecodedCard card in result.Cards)
        {
            ValidateCard(card);
            if (card.Points >= MinPoints)
                cards.Add(new HairCard(ToSequence(card), card.ShapeClass, card.Color));
        }
        return HairCardMesher.Build(cards, hairColorSrgb);
    }

    /// <summary>Spring chains of <paramref name="result"/> as static tubes and joint spheres; null when there are none.</summary>
    public static Mesh? BuildChains(HairDeckResult result)
    {
        ArgumentNullException.ThrowIfNull(result);
        List<float[]> polylines = new(result.Chains.Count);
        foreach (DecodedChain chain in result.Chains)
        {
            ArgumentNullException.ThrowIfNull(chain.Joints);
            if (chain.JointCount < 1 || chain.JointCount * 3 > chain.Joints.Length)
                throw new ArgumentException("A chain's joint count must match its joint array.", nameof(result));
            polylines.Add(chain.Joints.AsSpan(0, chain.JointCount * 3).ToArray());
        }
        return ChainTubeMesher.Build(polylines);
    }

    private static void ValidateCard(DecodedCard card)
    {
        ArgumentNullException.ThrowIfNull(card);
        ArgumentNullException.ThrowIfNull(card.Centerline);
        if (card.Points < 1 || card.Centerline.Length != card.Points * 3)
            throw new ArgumentException("A card's centreline must hold Points xyz triples.", nameof(card));
        RequireProfile(card.WidthProfile, card.Points, "width", nonNegative: true);
        RequireProfile(card.ThicknessProfile, card.Points, "thickness", nonNegative: true);
        if (card.RootNormal is not { Length: 3 } normal || normal.Any(v => !float.IsFinite(v)))
            throw new ArgumentException("A card's root normal must be three finite values.", nameof(card));
        if (normal[0] == 0f && normal[1] == 0f && normal[2] == 0f)
            throw new ArgumentException("A card's root normal must not be zero.", nameof(card));
        if (card.ShapeClass is not (0 or 1))
            throw new ArgumentException("A card's shape class must be 0 (square) or 1 (triangle).", nameof(card));
    }

    private static void RequireProfile(float[] profile, int points, string name, bool nonNegative)
    {
        if (profile.Length != points)
            throw new ArgumentException($"A card's {name} profile must hold one value per point.");
        foreach (float v in profile)
            if (!float.IsFinite(v) || (nonNegative && v < 0f))
                throw new ArgumentException($"A card's {name} profile must be finite and non-negative.");
    }

    /// <summary>Card rows (x, y, z, width, thickness, unit root width direction) in the reference card format.</summary>
    private static double[] ToSequence(DecodedCard card)
    {
        double nx = card.RootNormal[0];
        double ny = card.RootNormal[1];
        double nz = card.RootNormal[2];
        double length = Math.Sqrt(nx * nx + ny * ny + nz * nz);
        double[] sequence = new double[card.Points * RowStride];
        for (int i = 0; i < card.Points; i++)
        {
            sequence[RowStride * i] = card.Centerline[3 * i];
            sequence[RowStride * i + 1] = card.Centerline[3 * i + 1];
            sequence[RowStride * i + 2] = card.Centerline[3 * i + 2];
            sequence[RowStride * i + 3] = card.WidthProfile[i];
            sequence[RowStride * i + 4] = card.ThicknessProfile[i];
            sequence[RowStride * i + 5] = nx / length;
            sequence[RowStride * i + 6] = ny / length;
            sequence[RowStride * i + 7] = nz / length;
        }
        return sequence;
    }
}
