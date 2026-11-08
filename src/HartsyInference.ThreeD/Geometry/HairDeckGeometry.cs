using HartsyInference.ThreeD.Models.HairDeck;

namespace HartsyInference.ThreeD.Geometry;

/// <summary>
/// Hair cards and spring chains of one decoded HairDeck sample. A decoded card has one width and one thickness and no root
/// width direction, so the direction is the smallest principal axis of the card's first three points, signed away from the
/// mean root of the hairstyle (hairdeck.cards.codec, the rule for cards from another model).
/// </summary>
public static class HairDeckGeometry
{
    private const int MinPoints = 3;
    private const int RowStride = 8;
    private const int JacobiSweeps = 50;
    private const double JacobiTolerance = 1e-30;

    /// <summary>Hair cards of <paramref name="result"/> (null when no card has a triangle); see <see cref="HairCardMesher.Build"/>.</summary>
    public static HairCardMesh? BuildCards(HairDeckResult result, float[] hairColorSrgb)
    {
        ArgumentNullException.ThrowIfNull(result);
        List<DecodedCard> meshable = [];
        foreach (DecodedCard card in result.Cards)
        {
            ValidateCard(card);
            if (card.Points >= MinPoints)
                meshable.Add(card);
        }
        if (meshable.Count == 0)
            return null;

        Vector3d mean = new(0, 0, 0);
        foreach (DecodedCard card in meshable)
            mean += Root(card.Centerline);
        mean /= meshable.Count;

        List<HairCard> cards = new(meshable.Count);
        foreach (DecodedCard card in meshable)
        {
            Vector3d direction = RootWidthDirection(card.Centerline);
            if (Vector3d.Dot(direction, Root(card.Centerline) - mean) < 0)
                direction = new Vector3d(0, 0, 0) - direction;
            cards.Add(new HairCard(ToSequence(card, direction), card.ShapeClass, card.Color));
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
        if (!float.IsFinite(card.Width) || !float.IsFinite(card.Thickness) || card.Width < 0 || card.Thickness < 0)
            throw new ArgumentException("A card's width and thickness must be finite and non-negative.", nameof(card));
        if (card.ShapeClass is not (0 or 1))
            throw new ArgumentException("A card's shape class must be 0 (square) or 1 (triangle).", nameof(card));
    }

    /// <summary>Card rows (x, y, z, width, thickness, root width direction) in the reference card format.</summary>
    private static double[] ToSequence(DecodedCard card, Vector3d direction)
    {
        double[] sequence = new double[card.Points * RowStride];
        for (int i = 0; i < card.Points; i++)
        {
            sequence[RowStride * i] = card.Centerline[3 * i];
            sequence[RowStride * i + 1] = card.Centerline[3 * i + 1];
            sequence[RowStride * i + 2] = card.Centerline[3 * i + 2];
            sequence[RowStride * i + 3] = card.Width;
            sequence[RowStride * i + 4] = card.Thickness;
            sequence[RowStride * i + 5] = direction.X;
            sequence[RowStride * i + 6] = direction.Y;
            sequence[RowStride * i + 7] = direction.Z;
        }
        return sequence;
    }

    private static Vector3d Root(float[] centerline) => new(centerline[0], centerline[1], centerline[2]);

    /// <summary>Unit eigenvector of the smallest eigenvalue of the covariance of the first three points (sign unresolved).</summary>
    private static Vector3d RootWidthDirection(float[] centerline)
    {
        Vector3d[] points = [Root(centerline), new(centerline[3], centerline[4], centerline[5]),
            new(centerline[6], centerline[7], centerline[8])];
        Vector3d centre = (points[0] + points[1] + points[2]) / 3.0;
        double[,] covariance = new double[3, 3];
        foreach (Vector3d p in points)
        {
            double[] d = [p.X - centre.X, p.Y - centre.Y, p.Z - centre.Z];
            for (int a = 0; a < 3; a++)
                for (int b = 0; b < 3; b++)
                    covariance[a, b] += d[a] * d[b] / 2.0;
        }
        return SmallestEigenvector(covariance);
    }

    /// <summary>Cyclic Jacobi eigen-decomposition of a symmetric 3x3 matrix; returns the eigenvector of the smallest eigenvalue.</summary>
    private static Vector3d SmallestEigenvector(double[,] symmetric)
    {
        double[,] a = (double[,])symmetric.Clone();
        double[,] v = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
        int[][] pairs = [[0, 1], [0, 2], [1, 2]];
        for (int sweep = 0; sweep < JacobiSweeps; sweep++)
        {
            double off = a[0, 1] * a[0, 1] + a[0, 2] * a[0, 2] + a[1, 2] * a[1, 2];
            if (off < JacobiTolerance)
                break;
            foreach (int[] pq in pairs)
            {
                int p = pq[0];
                int q = pq[1];
                if (a[p, q] == 0.0)
                    continue;
                double phi = 0.5 * Math.Atan2(2.0 * a[p, q], a[q, q] - a[p, p]);
                double c = Math.Cos(phi);
                double s = Math.Sin(phi);
                // A <- J^T A J and V <- V J with J = [[c, s], [-s, c]] in the (p, q) plane.
                for (int k = 0; k < 3; k++)
                {
                    double akp = a[k, p];
                    double akq = a[k, q];
                    a[k, p] = c * akp - s * akq;
                    a[k, q] = s * akp + c * akq;
                }
                for (int k = 0; k < 3; k++)
                {
                    double apk = a[p, k];
                    double aqk = a[q, k];
                    a[p, k] = c * apk - s * aqk;
                    a[q, k] = s * apk + c * aqk;
                }
                for (int k = 0; k < 3; k++)
                {
                    double vkp = v[k, p];
                    double vkq = v[k, q];
                    v[k, p] = c * vkp - s * vkq;
                    v[k, q] = s * vkp + c * vkq;
                }
            }
        }
        int smallest = 0;
        for (int i = 1; i < 3; i++)
            if (a[i, i] < a[smallest, smallest])
                smallest = i;
        Vector3d direction = new(v[0, smallest], v[1, smallest], v[2, smallest]);
        return direction / (direction.Length + 1e-12);
    }
}
